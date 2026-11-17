# 模块 1 · 双轮模型、开环验证和 LQR 平衡对照。
import argparse
from math import atan2, degrees, hypot, radians
from time import perf_counter

import mujoco
import numpy as np

from controllers import solve_lqr
from two_wheel import (
    Duration_S, Fall_Pitch_Deg, Hold_Duration_S, Initial_Pitch_Deg,
    Pitch_Tolerance_Deg, Position_Tolerance_M, configure_plot, curve_output_path,
    linearize, load_plant, playback, prepare_contacts, read_model_xml,
)

Test_Torque_Nm = 0.025
Trial_Duration_S = 0.15
Viewer_Speed = 0.15
Viewer_Hold_S = 0.7


def run_open_loop_trial(model, data, torques, show_frame=None):
    # 每组都回到 XML 定义的初始位置，车身竖直、两轮静止。
    prepare_contacts(model, data)
    dt = model.opt.timestep

    chassis = data.body("chassis")
    mujoco.mj_forward(model, data)
    start_x = chassis.xpos[0]
    if show_frame is not None:
        show_frame(Viewer_Hold_S)

    # 开环输入就是 torques = (左轮扭矩, 右轮扭矩)，单位 N·m，顺序对应 XML 的两个电机。
    # 一开始设好扭矩，MuJoCo 根据动力学和地面摩擦计算这一小段时间内的运动。
    data.ctrl[:] = torques
    for _ in range(round(Trial_Duration_S / dt)):
        mujoco.mj_step(model, data)
        if show_frame is not None:
            show_frame(dt / Viewer_Speed)

    # mj_forward 更新最后一步的车身位置和姿态，仿真时间停留在这一刻。
    mujoco.mj_forward(model, data)
    # 旋转矩阵第一列是车身的 +x 轴在世界坐标中的方向，也就是鸭嘴的朝向。
    forward = chassis.xmat.reshape(3, 3)[:, 0]
    forward_displacement = chassis.xpos[0] - start_x
    # 俯仰角描述前倾或后仰，正值为前倾。偏航角描述水平转向，正值为左转。
    pitch = degrees(atan2(-forward[2], hypot(forward[0], forward[1])))
    yaw = degrees(atan2(forward[1], forward[0]))
    if show_frame is not None:
        show_frame(Viewer_Hold_S)
    return forward_displacement, pitch, yaw


def open_loop(case, viewer):
    xml = read_model_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 每一行给出命令行名称、结果名称和两只轮子的固定扭矩。
    trials = (
        ("zero", "零扭矩", (0.0, 0.0)),
        ("same", "同向扭矩", (Test_Torque_Nm, Test_Torque_Nm)),
        ("opposite", "反向扭矩", (-Test_Torque_Nm, Test_Torque_Nm)),
    )
    with playback(xml, data, viewer) as (window, show_frame):
        for case, label, torques in trials:
            if case not in ("all", case):
                continue
            if window is not None and window.is_running():
                window.set_texts((None, None, "Test", case))
            dx, pitch, yaw = run_open_loop_trial(model, data, torques, show_frame)
            print(f"{label} | 左轮 {torques[0]:+.3f} N·m，右轮 {torques[1]:+.3f} N·m")
            print(f"  {Trial_Duration_S:.2f} s 后，沿 x 位移 {dx * 100:+.2f} cm，"
                  f"俯仰角 {pitch:+.2f}°，偏航角 {yaw:+.2f}°")


def run_balance_trial(plant, gain=None, show_frame=None):
    # 给两组相同的前倾角和零初速度，比较开环和闭环从同一起点出发会怎样。
    state = plant.reset((0.0, 0.0, radians(Initial_Pitch_Deg), 0.0))
    reference = np.zeros(4)  # 目标为车身竖直并停在起点。
    times = [0.0]
    states = [state]
    torques = []
    reason = "达到实验时长"
    if show_frame is not None:
        show_frame(0.7)
    start = perf_counter()
    # 开环很快倒下，放慢播放。两组的仿真时间按同一个控制周期记录。
    playback_speed = 0.25 if gain is None else 1.0

    for _ in range(round(Duration_S / plant.dt)):
        # 误差统一为参考值减实际值。闭环扭矩是 K @ error，开环扭矩为零。
        error = reference - state
        requested = 0.0 if gain is None else (gain @ error).item()
        state = plant.step(requested)
        # 记录步末状态和这一步的实际执行器扭矩。两个相邻时刻之间对应一个扭矩。
        times.append(float(plant.data.time))
        states.append(state)
        torques.append(float(plant.data.actuator_force[0]))
        if show_frame is not None:
            show_frame(max(0.0, start + times[-1] / playback_speed - perf_counter()))
        if abs(degrees(state[2])) >= Fall_Pitch_Deg:
            # 到达指定倾角便结束这一组，记录检测到失衡的时刻。
            reason = "达到俯仰角边界"
            break
    if show_frame is not None:
        show_frame(0.7)
    return np.array(times), np.array(states), torques, reason


def settling_time(times, states):
    # 为每个采样点判断两件事：位置是否接近起点，俯仰角是否接近竖直。
    inside = (np.abs(states[:, 0]) <= Position_Tolerance_M) & (
        np.abs(np.rad2deg(states[:, 2])) <= Pitch_Tolerance_Deg
    )
    # 找最后一次越出容差带的索引。下一个采样点是持续留在带内的起点。
    outside = np.flatnonzero(~inside)
    start = outside[-1] + 1 if outside.size else 0
    if start >= len(times) or times[-1] - times[start] < Hold_Duration_S - 1e-9:
        return None
    return times[start]


def plot_trials(trials):
    # 三个子图共享同一组时间轴，便于对照扭矩和车身的反应。
    plt = configure_plot()
    fig, (pitch_ax, position_ax, torque_ax) = plt.subplots(
        3, 1, sharex=True, figsize=(8.4, 6.4), layout="constrained",
    )
    pitch_ax.set_title(f"双轮小黄鸭平衡实验 · 初始前倾 {Initial_Pitch_Deg:+g}°")
    pitch_ax.set_ylabel("俯仰角（°）")
    position_ax.set_ylabel("前后位置（cm）")
    torque_ax.set_ylabel("单轮扭矩（N·m）")
    torque_ax.set_xlabel("时间（秒）")
    for axis in (pitch_ax, position_ax, torque_ax):
        axis.axhline(0.0, color="#64748B", ls="--", lw=1)
        axis.grid(alpha=0.22)
    pitch_ax.axhspan(-Pitch_Tolerance_Deg, Pitch_Tolerance_Deg, color="#007872", alpha=0.09)
    position_ax.axhspan(-100 * Position_Tolerance_M, 100 * Position_Tolerance_M,
                       color="#007872", alpha=0.09)

    for label, color, (times, states, torques, reason) in trials:
        # 控制计算使用弧度和米，图上换成角度和厘米方便读数。
        angles = np.rad2deg(states[:, 2])
        positions = states[:, 0] * 100
        pitch_ax.plot(times, angles, color=color, lw=2, label=label)
        position_ax.plot(times, positions, color=color, lw=2)
        torque_ax.stairs(torques, times, color=color, lw=2, baseline=None)

        settled = settling_time(times, states)
        settled_text = "未达到" if settled is None else f"{settled:.2f} s"
        print(f"{label} | {reason} | 仿真 {times[-1]:.2f} s | 调节时间 {settled_text}")
        print(f"  结束俯仰角 {angles[-1]:+.3f}°，结束位置 {positions[-1]:+.3f} cm")
        print(f"  最大位移 {max(abs(positions)):.2f} cm，峰值单轮扭矩 {max(map(abs, torques)):.4f} N·m")

    pitch_ax.legend(loc="upper right", frameon=False)
    torque_ax.set_xlim(0, Duration_S)
    output = curve_output_path("01_balance_comparison.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def balance(viewer):
    xml, plant = load_plant()
    # 先用原始 MuJoCo 模型求出局部模型，再据此离线计算反馈增益。
    a, b = linearize(plant)
    gain = solve_lqr(a, b)
    with np.printoptions(precision=5, suppress=True):
        print("A =", a, sep="\n")
        print("B =", b, sep="\n")
        print("K =", gain)
    # 把 torque = -K @ s 代入局部模型，得到闭环矩阵 A - B @ K。
    # 最大特征值模长小于 1 时，这个离散线性模型的偏差会逐步衰减。
    radius = max(np.abs(np.linalg.eigvals(a - b @ gain)))
    print(f"控制周期 {plant.dt:.3f} s，线性闭环最大特征值模长 {radius:.5f}")

    controllers = (
        ("零扭矩", "#B65E2A", "Zero torque", None),
        ("LQR", "#007872", "LQR", gain),
    )
    trials = []
    with playback(xml, plant.data, viewer) as (window, show_frame):
        for label, color, viewer_label, policy in controllers:
            if window is not None and window.is_running():
                window.set_texts((None, None, "Balance", viewer_label))
            trials.append((label, color, run_balance_trial(plant, policy, show_frame)))
    plot_trials(trials)


def main():
    parser = argparse.ArgumentParser(description="核对开环运动方向，或者比较零扭矩与 LQR 平衡。")
    parser.add_argument("experiment", choices=("balance", "open-loop"), nargs="?", default="balance")
    parser.add_argument("--case", choices=("all", "zero", "same", "opposite"), default="all", help="选择开环工况。")
    parser.add_argument("--viewer", action="store_true", help="在 MuJoCo Viewer 回放。")
    args = parser.parse_args()
    if args.experiment == "open-loop":
        open_loop(args.case, args.viewer)
    else:
        balance(args.viewer)


if __name__ == "__main__":
    main()
