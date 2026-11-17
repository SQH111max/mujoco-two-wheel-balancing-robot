"""模块 6 · 操纵红球滚动，验证实时跟随和手动触发的随机抗扰。"""
from collections import deque
from importlib import import_module
from math import acos, atan2, cos, degrees, isfinite, radians, remainder, sin, sqrt, tau
from time import perf_counter, sleep
from xml.etree import ElementTree

import mujoco
import numpy as np

from controllers import Derivative_Filter_S, Yaw_Kd, Yaw_Ki, YawPid, read_yaw, solve_lqr
from two_wheel import BalancePlant, Fall_Pitch_Deg, Initial_Pitch_Deg, linearize, read_model_xml


Ball_Radius_M = 0.045
Ball_Mass_Kg = 0.09
Ball_Max_Speed_M_S = 0.40  # 按住方向键时的球速目标，单位 m/s。
Ball_Accel_M_S2 = 0.46  # 目标球速每秒最多变化多少，松开时也用这个速率减速。
Ball_Speed_Kp = 2.5  # 水平速度误差乘此系数得到作用于球心的力，单位 N/(m/s)。
Ball_Max_Force_N = 0.50
Follow_Gap_M = 0.10  # 球面到鸭子最近碰撞外缘的固定目标间隙。
Follow_Max_Speed_M_S = 0.35  # 小黄鸭的前进速度指令上限，单位 m/s。
Follow_Accel_M_S2 = 0.46  # 配合红球更快的启停，限制每拍速度指令的变化。
Gap_Kp = 0.90  # 把间隙误差换成追赶速度的系数，单位 1/s。
Gap_Brake_Preview_S = 0.35  # 接近时估算这段时间还会缩短多少间隙，给减速留出余量。
Position_Lead_Limit_M = 0.055  # 纵向局部参考点离轮轴最多这么远。
Heading_Kp = 2.6  # 把航向误差（rad）换成目标偏航角速度（rad/s）。
Follow_Max_Yaw_Rate_Rad_S = 0.90
Follow_Turn_First_Deg = 65.0  # 球偏离车头超过此角度时，先转向。
Push_Min_Force_N = 0.40
Push_Max_Force_N = 1.00
Push_Duration_S = 1.00
Push_Forward_Offset_M = 0.10  # 沿用 04 的施力点，位于车身质心前方 10 cm。
Recovery_Gap_Tolerance_M = 0.02
Recovery_Heading_Tolerance_Deg = 12.0
Recovery_Relative_Speed_Tolerance_M_S = 0.04
Recovery_Hold_S = 0.50  # 连续半秒满足三个条件才算恢复跟踪。
Arrow_Keys = ("Up", "Down", "Left", "Right")


def load_ball_plant():
    # 在内存里的场景加球，其他实验仍从原来的 two_wheel.xml 加载小黄鸭。
    scene = ElementTree.fromstring(read_model_xml())
    world = scene.find("worldbody")
    world.find("geom[@name='floor']").set("size", "0 0 0.1")  # 绕场地行驶时也一直显示地面。
    ball = ElementTree.SubElement(world, "body", name="red_ball", pos=f"0.48 0.16 {Ball_Radius_M}")
    ElementTree.SubElement(ball, "freejoint", name="ball_joint")
    ElementTree.SubElement(ball, "geom", name="ball_contact", type="sphere",
                           size=str(Ball_Radius_M), mass=str(Ball_Mass_Kg), rgba="0.90 0.04 0.07 1")
    # 表面小标记随球旋转，方便辨认滚动。接触和质量由红色球体计算。
    ElementTree.SubElement(ball, "geom", {"class": "decoration", "type": "sphere",
                                         "pos": f"0 0 {Ball_Radius_M - 0.002}", "size": "0.005",
                                         "rgba": "1 0.52 0.44 1"})
    model = mujoco.MjModel.from_xml_string(ElementTree.tostring(scene, encoding="unicode"))
    return BalancePlant(model)


class PpoTorqueCompensation:
    """复用 05 号的策略和扭矩公式。每 5 拍更新 a，每拍重新算补偿。"""

    def __init__(self):
        self.strategy = import_module("05_residual_ppo")
        if not self.strategy.Checkpoint.is_file():
            raise FileNotFoundError(self.strategy.Checkpoint)
        self.strategy.torch.set_num_threads(1)
        self.model = self.strategy.PPO.load(self.strategy.Checkpoint, device="cpu")
        self.reset()

    def reset(self):
        self.next_action_step = None
        self.action = None

    def torque(self, gain, error, step):
        if self.next_action_step is None or step >= self.next_action_step:
            self.action, _ = self.model.predict(self.strategy.observation(error), deterministic=True)
            self.next_action_step = step + self.strategy.Action_Hold_Steps
        _, residual = self.strategy.combined_torque(gain, error, self.action)
        return residual


class BallTracking:
    """每隔 0.01 秒推进一次红球和小黄鸭的闭环。"""

    def __init__(self, seed=None, compensation=None):
        self.plant = load_ball_plant()
        self.gain = solve_lqr(*linearize(self.plant))
        # 位置误差和速度误差都进 LQR。两项增益的比值表示正向位置误差相当于增加多少速度指令。
        self.speed_per_position_error = self.gain[0, 0] / self.gain[0, 1]
        self.data = self.plant.data
        self.duck = self.data.joint("base_joint")
        self.ball = self.data.joint("ball_joint")
        self.duck_body = self.data.body("chassis").id
        self.ball_body = self.data.body("red_ball").id
        self.ball_geom = self.plant.model.geom("ball_contact").id
        self.duck_geoms = tuple(self.plant.model.geom(name).id for name in (
            "body_box", "left_contact", "right_contact",
        ))
        self.random = np.random.default_rng(seed)
        self.compensation = compensation
        self.compensation_enabled = compensation is not None
        self.pending_pushes = deque()
        self.push_arm_local = np.array([Push_Forward_Offset_M, 0.0, 0.0])
        self.push_arm_world = np.empty(3)
        self.reset()

    def reset(self):
        self.plant.reset((0.0, 0.0, radians(Initial_Pitch_Deg), 0.0))
        self.yaw_pid = YawPid(self.plant.dt, Yaw_Ki, Yaw_Kd, Derivative_Filter_S)
        self.yaw_rate = 0.0
        self.ball_speed_reference = np.zeros(2)
        self.forward_speed_reference = 0.0
        self.fallen = False
        self.pending_pushes.clear()
        self.last_push = None
        self.push_steps_left = 0
        self.push_ended_at = self.within_recovery_since = self.recovery_s = None
        self.residual_torque = 0.0
        if self.compensation is not None:
            self.compensation.reset()
        self.minimum_gap = self.gap()
        self.peak_pitch_deg = Initial_Pitch_Deg

    @property
    def compensation_active(self):
        # 施力中和撤力后的恢复期都可以补偿，达到恢复判据后回到基础控制。
        recovering = self.push_ended_at is not None and self.recovery_s is None
        return self.compensation_enabled and not self.fallen and (self.push_steps_left > 0 or recovering)

    def queue_push(self, min_force, max_force, duration, vertical):
        # 每次按下都独立抽取一次力。已在施力时排队，前一段结束再开始下一段。
        step_count = duration / self.plant.dt
        if (not all(map(isfinite, (min_force, max_force, duration)))
                or not 0 <= min_force <= max_force or duration <= 0 or not isfinite(step_count)):
            raise ValueError("力需满足 0 ≤ 下限 ≤ 上限，时长需大于 0。")
        azimuth = self.random.uniform(0.0, tau)
        # 三维模式直接抽取 z 方向分量占比，正值向上，负值向下。
        vertical_ratio = self.random.uniform(-1.0, 1.0) if vertical else 0.0
        horizontal_ratio = sqrt(1.0 - vertical_ratio ** 2)
        magnitude = self.random.uniform(min_force, max_force)
        force = magnitude * np.array([horizontal_ratio * cos(azimuth),
                                      horizontal_ratio * sin(azimuth), vertical_ratio])
        # 以控制周期计时，末尾的实际持续时间显示在面板上。
        steps = max(1, round(step_count))
        self.pending_pushes.append((force, steps))

    def apply_push(self):
        if not self.push_steps_left and self.pending_pushes:
            self.last_push = self.pending_pushes.popleft()
            force, self.push_steps_left = self.last_push
            self.push_ended_at = self.within_recovery_since = self.recovery_s = None
            if self.compensation is not None:
                self.compensation.reset()
            print(f"t={self.data.time:.2f} s  推力 {np.linalg.norm(force):.2f} N  "
                  f"(Fx, Fy, Fz)=({force[0]:+.2f}, {force[1]:+.2f}, {force[2]:+.2f}) N  "
                  f"持续 {self.push_steps_left * self.plant.dt:.2f} s")
        if self.push_steps_left:
            force, _ = self.last_push
            wrench = self.data.xfrc_applied[self.duck_body]
            # 在质心前方施力，等效成质心处的外力加一个转动力矩。
            mujoco.mju_rotVecQuat(self.push_arm_world, self.push_arm_local, self.duck.qpos[3:7])
            wrench[:3] = force
            wrench[3:] = np.cross(self.push_arm_world, force)

    def gap(self):
        # 先按最新的位置更新几何体坐标，再取球与三个碰撞几何体中最近的一处。
        model = self.plant.model
        mujoco.mj_kinematics(model, self.data)
        return min(mujoco.mj_geomDistance(model, self.data, self.ball_geom, geom, 10.0, None)
                   for geom in self.duck_geoms)

    def drive_ball(self, direction):
        # 一个二维向量表示按键方向。斜向组合键也使用相同的速度上限。
        direction = np.asarray(direction, dtype=float)
        speed_command = Ball_Max_Speed_M_S * direction / max(1.0, np.linalg.norm(direction))
        speed_change = speed_command - self.ball_speed_reference
        speed_limit = Ball_Accel_M_S2 * self.plant.dt
        self.ball_speed_reference += speed_change / max(1.0, np.linalg.norm(speed_change) / speed_limit)
        force = Ball_Speed_Kp * (self.ball_speed_reference - self.ball.qvel[:2])
        self.data.xfrc_applied[self.ball_body, :2] = force / max(1.0, np.linalg.norm(force) / Ball_Max_Force_N)

    def ball_direction(self, yaw):
        relative_position = self.ball.qpos[:2] - self.duck.qpos[:2]
        ball_distance = np.linalg.norm(relative_position)
        direction_to_ball = relative_position / max(ball_distance, 1e-6)
        target_yaw = atan2(relative_position[1], relative_position[0]) if ball_distance > 1e-6 else yaw
        return direction_to_ball, remainder(target_yaw - yaw, tau)

    def wheel_commands(self, yaw):
        direction_to_ball, heading_error = self.ball_direction(yaw)
        yaw_rate_reference = float(np.clip(Heading_Kp * heading_error, -Follow_Max_Yaw_Rate_Rad_S,
                                           Follow_Max_Yaw_Rate_Rad_S))
        # 航向偏得越多就走得越慢。超过设定角度后，转向优先。
        alignment = np.interp(abs(heading_error), (0.0, radians(Follow_Turn_First_Deg)), (1.0, 0.0))
        gap_error = Follow_Gap_M - self.gap()
        # 先跟上球沿连线方向的速度，再按间隙误差加减速度。
        ball_speed_along_line = self.ball.qvel[:2] @ direction_to_ball
        closing_speed = max(0.0, self.duck.qvel[:2] @ direction_to_ball - ball_speed_along_line)
        braking_error = gap_error + Gap_Brake_Preview_S * closing_speed
        speed_command = alignment * np.clip(ball_speed_along_line - Gap_Kp * braking_error, 0.0, Follow_Max_Speed_M_S)
        speed_change = Follow_Accel_M_S2 * self.plant.dt
        self.forward_speed_reference += np.clip(speed_command - self.forward_speed_reference,
                                                -speed_change, speed_change)
        # 距离环生成有界的局部参考点。下面的四项仍按参考减实际送给 LQR。
        position_error = alignment * np.clip(-braking_error, -Position_Lead_Limit_M, Position_Lead_Limit_M)
        # 接近速度上限时，限制参考点超前量，给位置反馈留下的速度余量由本拍指令决定。
        speed_margin = max(0.0, Follow_Max_Speed_M_S - self.forward_speed_reference)
        position_error = min(position_error, speed_margin / self.speed_per_position_error)
        forward = np.array([cos(yaw), sin(yaw)])
        speed_error = self.forward_speed_reference - forward @ self.duck.qvel[:2]
        pitch, pitch_rate = self.plant.state()[2:]
        errors = np.array([position_error, speed_error, -pitch, -pitch_rate])
        common_torque = (self.gain @ errors).item()
        self.residual_torque = 0.0
        if self.compensation_active:
            step = round(self.data.time / self.plant.dt)
            residual = self.compensation.torque(self.gain, errors, step)
            # 速度增益为负。负的补偿扭矩等效于进一步提高速度指令，所以也计入速度余量。
            torque_margin = self.gain[0, 1] * speed_margin - self.gain[0, 0] * position_error
            self.residual_torque = max(residual, min(0.0, torque_margin))
            common_torque += self.residual_torque
        return self.yaw_pid.step(yaw_rate_reference, self.yaw_rate, common_torque,
                                 self.plant.min_torque, self.plant.max_torque)

    def measure_recovery(self, gap, yaw):
        if self.push_ended_at is None or self.recovery_s is not None:
            return
        direction, heading_error = self.ball_direction(yaw)
        # 只比较沿连线的速度。球仍在滚动时，球和鸭也可以恢复到稳定的跟随状态。
        relative_speed = (self.ball.qvel[:2] - self.duck.qvel[:2]) @ direction
        if (abs(Follow_Gap_M - gap) > Recovery_Gap_Tolerance_M
                or abs(degrees(heading_error)) > Recovery_Heading_Tolerance_Deg
                or abs(relative_speed) > Recovery_Relative_Speed_Tolerance_M_S):
            self.within_recovery_since = None
            return
        if self.within_recovery_since is None:
            self.within_recovery_since = self.data.time
        if self.data.time - self.within_recovery_since >= Recovery_Hold_S - 1e-9:
            self.recovery_s = self.within_recovery_since - self.push_ended_at
            print(f"撤力后 {self.recovery_s:.2f} s 恢复跟随")

    def step(self, direction):
        if self.fallen:
            return
        self.drive_ball(direction)
        yaw = read_yaw(self.data)
        self.apply_push()
        state = self.plant.step(self.wheel_commands(yaw))
        new_yaw = read_yaw(self.data)
        self.yaw_rate = remainder(new_yaw - yaw, tau) / self.plant.dt
        gap = self.gap()
        pitch_deg = abs(degrees(state[2]))
        self.peak_pitch_deg = max(self.peak_pitch_deg, pitch_deg)
        self.minimum_gap = min(self.minimum_gap, gap)
        # 比较车身竖直轴与世界竖直轴。斜推造成向侧面倾倒时也能判定失衡。
        _, qx, qy, _ = self.duck.qpos[3:7]
        tilt_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.fallen = degrees(acos(float(np.clip(tilt_cos, -1.0, 1.0)))) >= Fall_Pitch_Deg
        if self.push_steps_left:
            self.push_steps_left -= 1
            if not self.push_steps_left:
                self.push_ended_at = self.data.time
        if not self.push_steps_left or self.fallen:
            self.data.xfrc_applied[self.duck_body] = 0.0
        if self.fallen:
            self.data.xfrc_applied[self.ball_body] = 0.0
            self.push_steps_left = 0
            self.pending_pushes.clear()
            self.residual_torque = 0.0
            return
        self.measure_recovery(gap, new_yaw)


class SimulationView:
    """将 MuJoCo 画面放在主窗口左侧，鼠标拖动旋转视角，滚轮缩放。"""

    def __init__(self, root, sim):
        import tkinter as tk

        self.sim = sim
        self.width = min(900, max(420, root.winfo_screenwidth() - 370))
        self.height = min(600, max(400, root.winfo_screenheight() - 140))
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, highlightthickness=0, takefocus=True)
        self.canvas.grid(row=0, column=0)
        self.image = tk.PhotoImage(master=root, width=self.width, height=self.height)
        self.canvas.create_image(0, 0, image=self.image, anchor="nw")
        # 像素是 MuJoCo 画好的 RGB 颜色，PPM 文件头告诉 Tk 要按多大的矩形显示。
        self.pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.ppm_header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.25, 0.08, 0.07)
        self.camera.distance = 1.38
        self.camera.azimuth = 0.0  # 初始视角中，方向键的上方对应世界 +x。
        self.camera.elevation = -52.0
        self.drag_position = (0, 0)
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.orbit)
        self.canvas.bind("<MouseWheel>", self.zoom)

    def start_drag(self, event):
        self.canvas.focus_set()
        self.drag_position = (event.x, event.y)

    def orbit(self, event):
        x, y = self.drag_position
        self.camera.azimuth += 0.5 * (event.x - x)
        self.camera.elevation = float(np.clip(self.camera.elevation - 0.5 * (event.y - y), -85, -10))
        self.drag_position = (event.x, event.y)

    def zoom(self, event):
        self.camera.distance = float(np.clip(self.camera.distance * 0.90 ** (event.delta / 120), 0.4, 6.0))

    def draw(self, renderer):
        self.camera.lookat[:2] = (self.sim.duck.qpos[:2] + self.sim.ball.qpos[:2]) / 2
        renderer.update_scene(self.sim.data, self.camera)
        renderer.render(out=self.pixels)
        self.image.configure(data=self.ppm_header + self.pixels.tobytes(), format="PPM")


class ControlPanel:
    """主窗口右侧的跟随设置和推力设置。"""

    def __init__(self, root, sim):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.sim = sim
        self.running = True
        self.pressed = set()

        content = ttk.Frame(root, padding=(16, 12))
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure((1, 2), weight=1)
        ttk.Label(content, text="红球跟随与抗扰", font=("Microsoft YaHei UI", 14, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8),
        )
        ttk.Label(content, text="方向键滚动，松开减速 · P 施加推力\n拖动画面旋转视角，滚轮缩放").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10),
        )
        ttk.Separator(content).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        self.min_force = tk.StringVar(value=f"{Push_Min_Force_N:g}")
        self.max_force = tk.StringVar(value=f"{Push_Max_Force_N:g}")
        self.duration = tk.StringVar(value=f"{Push_Duration_S:g}")
        self.mode = tk.StringVar(value="三维随机")
        self.use_compensation = tk.BooleanVar(value=sim.compensation_enabled)
        for row, label in enumerate(("预留间隙（cm）", "推力下限（N）", "推力上限（N）",
                                     "持续时间（s）", "推力方向"), start=3):
            ttk.Label(content, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Label(content, text=f"{Follow_Gap_M * 100:g}").grid(row=3, column=1, columnspan=2, sticky="w")
        for row, value in enumerate((self.min_force, self.max_force, self.duration), start=4):
            ttk.Entry(content, textvariable=value, width=12).grid(row=row, column=1, columnspan=2, sticky="ew")
        for column, mode in enumerate(("三维随机", "水平随机"), start=1):
            ttk.Radiobutton(content, text=mode, value=mode, variable=self.mode).grid(row=7, column=column, sticky="w")
        compensation_toggle = ttk.Checkbutton(
            content, text="启用 PPO 扭矩补偿", variable=self.use_compensation, command=self.change_compensation,
        )
        compensation_toggle.grid(row=8, column=0, columnspan=3, sticky="w", pady=(8, 0))
        if sim.compensation is None:
            compensation_toggle.state(["disabled"])
        ttk.Button(content, text="施加推力（P）", command=self.request_push).grid(row=9, column=0, sticky="w", pady=10)
        ttk.Button(content, text="复位", command=self.reset).grid(row=9, column=2, sticky="e", pady=10)

        self.notice = tk.StringVar()
        self.readout = tk.StringVar()
        self.push_readout = tk.StringVar()
        for row, value in enumerate((self.notice, self.readout, self.push_readout), start=10):
            ttk.Label(content, textvariable=value, width=38, justify="left").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(0, 6),
            )
        self.refresh()

    def bind_keyboard(self):
        # 把运动键交给整个窗口。调整数字时，方向键和 P 仍然是仿真的快捷键。
        # Tk 的绑定标签是一组事件处理顺序，把仿真快捷键放在输入框的默认操作之前。
        shortcuts = "BallTrackingShortcuts"
        self.root.bind_class(shortcuts, "<KeyPress>", self.press_key)
        self.root.bind_class(shortcuts, "<KeyRelease>", self.release_key)
        self.root.bind("<FocusOut>", self.focus_changed)
        widgets = [self.root]
        while widgets:
            widget = widgets.pop()
            widget.bindtags((shortcuts, *widget.bindtags()))
            widgets.extend(widget.winfo_children())

    def focus_changed(self, event):
        # 同窗切换到输入框时保留按键。离开整个窗口时清除，避免漏掉外部的松键事件。
        self.root.after_idle(self.clear_background_keys)

    def clear_background_keys(self):
        if self.root.focus_displayof() is None:
            self.pressed.clear()

    def change_compensation(self):
        self.sim.compensation_enabled = self.use_compensation.get()
        self.sim.compensation.reset()
        self.sim.residual_torque = 0.0
        self.refresh()

    def press_key(self, event):
        key = event.keysym
        if key in Arrow_Keys:
            self.pressed.add(key)
            return "break"
        if key.lower() == "p":
            if "p" not in self.pressed:
                self.pressed.add("p")
                self.request_push()
            return "break"

    def release_key(self, event):
        self.pressed.discard(event.keysym if event.keysym in Arrow_Keys else event.keysym.lower())

    def direction(self):
        return (int("Up" in self.pressed) - int("Down" in self.pressed),
                int("Left" in self.pressed) - int("Right" in self.pressed))

    def request_push(self):
        if self.sim.fallen:
            self.notice.set("车身已失衡，请先复位。")
            return
        try:
            self.sim.queue_push(float(self.min_force.get()), float(self.max_force.get()), float(self.duration.get()),
                                vertical=self.mode.get() == "三维随机")
        except ValueError:
            self.notice.set("请输入有效数值，力满足 0 ≤ 下限 ≤ 上限，时长 > 0。")
            return
        self.notice.set("已登记一次推力。")

    def reset(self):
        self.pressed.clear()
        self.sim.reset()
        self.notice.set("已复位。")
        self.refresh()

    def close(self):
        self.running = False
        self.pressed.clear()

    def refresh(self):
        sim = self.sim
        actual_gap_cm = 100 * sim.gap()
        ball_speed = np.linalg.norm(sim.ball.qvel[:2])
        status = "车身失衡，等待复位" if sim.fallen else "跟随中"
        if sim.compensation_active:
            ppo_status = f"补偿指令 {sim.residual_torque:+.4f} N·m/轮"
        elif sim.compensation_enabled:
            ppo_status = "待命"
        else:
            ppo_status = "策略未加载" if sim.compensation is None else "已关闭"
        self.readout.set(
            f"{status}  ·  {sim.data.time:.1f} s\n"
            f"当前间隙 {actual_gap_cm:.1f} cm   最小 {100 * sim.minimum_gap:.1f} cm\n"
            f"球速 {ball_speed:.2f} m/s   最大俯仰 {sim.peak_pitch_deg:.1f}°\n"
            f"PPO {ppo_status}"
        )
        if sim.last_push is None:
            self.push_readout.set("推力记录\n尚未施力")
            return
        force, steps = sim.last_push
        dt = sim.plant.dt
        recovery = "等待恢复" if sim.recovery_s is None else f"恢复用时 {sim.recovery_s:.2f} s"
        if sim.push_steps_left:
            recovery = "施力中"
        elif sim.fallen:
            recovery = "车身失衡"
        self.push_readout.set(
            f"最近推力 {np.linalg.norm(force):.2f} N · {steps * dt:.2f} s\n"
            f"(Fx, Fy, Fz) = ({force[0]:+.2f}, {force[1]:+.2f}, {force[2]:+.2f}) N\n"
            f"剩余 {sim.push_steps_left * dt:.2f} s   排队 {len(sim.pending_pushes)} 次\n"
            f"{recovery}"
        )


def main():
    import tkinter as tk

    try:
        compensation = PpoTorqueCompensation()
    except (ImportError, FileNotFoundError) as error:
        print(f"PPO 策略加载失败，使用 LQR 和转向 PID。{error}")
        compensation = None
    sim = BallTracking(compensation=compensation)
    root = tk.Tk()
    try:
        root.title("MuJoCo · 06 红球跟随与抗扰")
        root.resizable(False, False)
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        view = SimulationView(root, sim)
        panel = ControlPanel(root, sim)
        panel.bind_keyboard()
        root.protocol("WM_DELETE_WINDOW", panel.close)
        root.update_idletasks()
        view.canvas.focus_set()

        model = sim.plant.model
        model.vis.global_.offwidth = view.width
        model.vis.global_.offheight = view.height
        with mujoco.Renderer(model, height=view.height, width=view.width) as renderer:
            deadline = next_frame = next_readout = perf_counter()
            while panel.running:
                root.update()
                if not panel.running:
                    break
                now = perf_counter()
                # 控制每 0.01 仿真秒推进一次。绘制较慢的一帧时，补做这段时间内到期的仿真步。
                deadline = max(deadline, now - 0.10)
                direction = panel.direction()
                while deadline <= now:
                    sim.step(direction)
                    deadline += sim.plant.dt
                if now >= next_frame:
                    view.draw(renderer)
                    next_frame = now + 1 / 40
                if now >= next_readout:
                    panel.refresh()
                    next_readout = now + 0.10
                sleep(max(0.0, min(deadline, next_frame) - perf_counter()))
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
