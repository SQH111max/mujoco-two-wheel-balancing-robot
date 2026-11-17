# 模块 5 · PPO 学习辅助 LQR 的共同扭矩，用于直线行驶抗扰。
import argparse
from importlib import import_module
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from controllers import solve_lqr
from two_wheel import Fall_Pitch_Deg, linearize, load_plant, make_speed_reference, playback, setup_line_viewer


Residual_Max_Torque_Nm = 0.025  # 单轮共同扭矩的补偿上限。
Position_Gain_Adjustment = 0.35  # (N·m)/m。PPO 的动作改变位置误差对应的反馈力度。
Residual_Enable_S = 4.0  # LQR 完成启驶后，才进入 PPO 训练所覆盖的匀速跟踪阶段。
Action_Hold_Steps = 5  # PPO 每 5 拍更新动作，基础 LQR 仍每拍更新。
Episode_Duration_S = 3.5  # 单个训练回合的时长。评估时长取 04 号双推力工况的设置。
Training_Steps = 65_536
Training_Seed = 7
Observation_Scales = np.array([0.05, 0.15, 0.08, 0.7])  # 依次为 m、m/s、rad、rad/s。
Error_Weights = np.array([3.0, 1.0, 0.6, 0.1])
Residual_Weight = 0.05
Checkpoint = Path(__file__).with_name("ppo_residual.zip")


def observation(error):
    # 网络使用归一化后的四个误差。从车身状态计算，不读取外力值和施力时序。
    # 策略选择的是反馈力度。将前后反号的误差折到位置特征非负的一侧。
    # 在位置误差等于零时也沿用已训练策略对其余特征的符号约定。
    canonical = error if error[0] > 0.0 else -error
    return np.clip(canonical / Observation_Scales, -8.0, 8.0).astype(np.float32)


def combined_torque(gain, error, action):
    # 误差按参考值减实际值。LQR 的基础扭矩为 K @ error，PPO 调节位置反馈。
    # 补偿扭矩随位置误差趋零而消失，单轮实际扭矩的物理限幅由 BalancePlant 执行。
    gain_adjustment = Position_Gain_Adjustment * float(np.clip(action[0], -1.0, 1.0))
    residual = float(np.clip(-gain_adjustment * error[0], -Residual_Max_Torque_Nm, Residual_Max_Torque_Nm))
    return (gain @ error).item() + residual, residual


class ResidualTorqueEnv(gym.Env):
    """每个回合从匀速行驶开始，随机施加一次纵向推力，训练恢复策略。"""

    def __init__(self, plant, gain, disturbance, target_speed, fall_pitch):
        super().__init__()
        self.plant = plant
        self.gain = gain
        self.disturbance = disturbance
        self.target_speed = target_speed
        self.fall_pitch = np.deg2rad(fall_pitch)
        self.body_id = plant.model.body("chassis").id
        self.steps = round(Episode_Duration_S / plant.dt)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-8.0, 8.0, shape=(4,), dtype=np.float32)

    def error(self):
        target_position = self.step_index * self.plant.dt * self.target_speed
        target = (target_position, self.target_speed, 0.0, 0.0)
        return target - self.state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        self.state = self.plant.reset((0.0, self.target_speed, 0.0, 0.0))
        sign = self.np_random.choice((-1.0, 1.0))
        force = sign * self.np_random.uniform(0.30, 1.20)
        if self.np_random.random() < 0.20:
            force = 0.0  # 一部分回合保持匀速，用于练习平稳巡航时少补偿。
        self.forces, _, _ = self.disturbance.make_pulse(
            self.steps, self.plant.dt, self.np_random.uniform(0.25, 1.10),
            self.np_random.uniform(0.12, 0.32), force,
        )
        return observation(self.error()), {}

    def step(self, action):
        # 一个 PPO 动作保持五个物理控制周期，每拍仍用新状态更新基础扭矩和补偿。
        reward = 0.0
        error = self.error()
        for _ in range(Action_Hold_Steps):
            torque, residual = combined_torque(self.gain, error, action)
            self.plant.data.xfrc_applied[self.body_id, 0] = self.forces[self.step_index]
            self.state = self.plant.step(torque)
            self.step_index += 1
            error = self.error()
            scaled_error = error / Observation_Scales
            cost = Error_Weights @ (scaled_error ** 2) + Residual_Weight * (residual / Residual_Max_Torque_Nm) ** 2
            reward -= float(cost) / Action_Hold_Steps
            # 失衡与达到时长分别用 terminated、truncated 表示。
            if abs(self.state[2]) >= self.fall_pitch:
                return observation(error), reward - 20.0, True, False, {}
            if self.step_index >= self.steps:
                return observation(error), reward, False, True, {}
        return observation(error), reward, False, False, {}


def train(plant, gain, disturbance, reference, steps):
    env = ResidualTorqueEnv(plant, gain, disturbance, reference[-1, 1], Fall_Pitch_Deg)
    model = PPO(
        "MlpPolicy", env, seed=Training_Seed, device="cpu", verbose=1,
        n_steps=1024, batch_size=128, n_epochs=8, gamma=0.985, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[32, 32], log_std_init=-0.7),
    )
    model.learn(total_timesteps=steps, log_interval=10)
    model.save(Checkpoint)
    print(f"策略文件：{Checkpoint}｜训练步数 {model.num_timesteps}")
    return model


def make_policy(model, plant, gain):
    enable_step = round(Residual_Enable_S / plant.dt)
    action = np.zeros(1, dtype=np.float32)

    def policy(state, target_speed):
        nonlocal action
        # 路径控制以当拍参考点为局部原点。实测位置、速度均沿当前车头方向投影。
        # 参考纵向位置、俯仰角和俯仰角速度都是零，误差仍按参考减实际。
        error = (0.0, target_speed, 0.0, 0.0) - state
        elapsed_steps = round(plant.data.time / plant.dt) - enable_step
        if elapsed_steps < 0:
            return (gain @ error).item()
        if elapsed_steps % Action_Hold_Steps == 0:
            action, _ = model.predict(observation(error), deterministic=True)
        torque, _ = combined_torque(gain, error, action)
        return torque

    return policy


def evaluate(plant, gain, disturbance, reference, drive_start, model, show_viewer, xml):
    path = import_module("03_path_tracking")
    total_steps = len(reference) - 1
    # 两组都复用 04 号的外力时序、作用点和转向 PID。PPO 调节两轮的共同扭矩。
    forces, windows = disturbance.make_force_schedule(total_steps, plant.dt)
    disturbance.print_force_schedule(forces, windows, plant.dt)
    policy = make_policy(model, plant, gain)
    trials = []
    with playback(xml, plant.data, show_viewer) as (window, show_frame):
        if window is not None and window.is_running():
            setup_line_viewer(window, reference[-1, 0])
        for label, color, controller in (("LQR", "#007872", None), ("LQR + PPO", "#73459E", policy)):
            if window is not None and window.is_running():
                window.set_texts((None, None, "Control", f"{label} + Yaw PID | Two pushes"))
            result, reason = disturbance.run_planar_trial(
                plant, gain, path, reference, drive_start, forces, windows[1], show_frame,
                common_torque_policy=controller,
            )
            errors = disturbance.planar_errors(path, result, reference)
            trials.append((label, color, result, reason, errors))
    disturbance.print_planar_results(trials, windows, total_steps, plant.dt)
    disturbance.plot_planar_results(trials, forces, windows, plant.dt, filename="05_disturbance_lqr_vs_ppo.png")


def main():
    parser = argparse.ArgumentParser(description="训练 PPO 扭矩补偿并在两次推力下对照抗扰效果。")
    parser.add_argument("--train", action="store_true", help="从头训练并保存策略。直接运行时加载已有策略评估。")
    parser.add_argument("--steps", type=int, default=Training_Steps, help="训练步数，按 PPO 的采样批次取整。")
    parser.add_argument("--viewer", action="store_true", help="回放双推力工况下的 LQR 和 LQR + PPO。")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("训练步数必须大于零。")
    if not args.train and not Checkpoint.is_file():
        parser.error("项目中找不到策略文件，请先用 --train 训练。")
    torch.set_num_threads(1)
    disturbance = import_module("04_disturbance")
    xml, plant = load_plant()
    gain = solve_lqr(*linearize(plant))
    reference, drive_start, _ = make_speed_reference(disturbance.Disturbance_Duration_S, plant.dt, stop=False)
    if args.train:
        model = train(plant, gain, disturbance, reference, args.steps)
    else:
        model = PPO.load(Checkpoint, device="cpu")
    evaluate(plant, gain, disturbance, reference, drive_start, model, args.viewer, xml)


if __name__ == "__main__":
    main()
