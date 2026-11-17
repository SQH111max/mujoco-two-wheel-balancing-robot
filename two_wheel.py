"""MuJoCo 模型、线性化、参考轨迹和各实验共用的仿真工具。"""
from contextlib import contextmanager
from math import asin, cos, radians, sin
from pathlib import Path
from time import perf_counter, sleep

import mujoco
import numpy as np


Prepare_Duration_S = 0.10  # 每组实验先用零扭矩让车轮与地面接触稳定。
Initial_Pitch_Deg = 3.0
Duration_S = 10.0
Fall_Pitch_Deg = 30.0
Control_Steps = 2  # 两个物理步更新一次控制，共 0.01 秒。
Difference_Step = 1e-5  # 中心差分的扰动。
Position_Tolerance_M = 0.01
Pitch_Tolerance_Deg = 1.0
Hold_Duration_S = 1.0
Target_Speed_M_S = 0.10
Target_Arc_Yaw_Rate_Rad_S = 0.50  # 正值左转，负值右转。
Drive_Start_S = 2.0
Drive_Stop_S = 5.0


def read_model_xml():
    # __file__ 是本文件的路径。读取同目录的 XML 文本，供调用方加载车身、车轮和电机。
    # 先用 Python 读取 UTF-8 文本，方便 MuJoCo 从中文路径加载模型。
    return Path(__file__).with_name("two_wheel.xml").read_text(encoding="utf-8")


def curve_output_path(filename):
    # 以项目目录为基准，让各实验的曲线图统一保存到同一个文件夹。
    folder = Path(__file__).with_name("曲线结果图")
    folder.mkdir(exist_ok=True)
    return folder / filename


def prepare_contacts(model, data):
    # 清空上次实验的数据，让两只轮子从同一个接地状态开始。
    mujoco.mj_resetData(model, data)
    # nstep 是物理步数。用预备时间除以仿真步长，算出要推进多少步。
    mujoco.mj_step(model, data, nstep=round(Prepare_Duration_S / model.opt.timestep))


@contextmanager
def playback(xml, data, enabled, *, floor_half_length=None):
    # “with playback(...)” 把窗口和逐帧显示函数交给实验。退出 with 时关闭窗口。
    if not enabled:
        yield None, None
        return

    from mujoco import viewer

    # 窗口用独立的模型副本，实际控制和数据记录用原来的仿真模型。
    display_model = mujoco.MjModel.from_xml_string(xml)
    if floor_half_length is not None:
        # 延长 Viewer 中显示的格子地面，让整条长路线都能看到地面。
        floor = display_model.geom("floor")
        floor.size[0] = max(float(floor.size[0]), floor_half_length)
    display_data = mujoco.MjData(display_model)
    with viewer.launch_passive(
        display_model, display_data, show_left_ui=False, show_right_ui=False,
    ) as window:
        with window.lock():
            # lookat 指向鸭身，distance、azimuth 和 elevation 决定镜头的位置和角度。
            window.cam.lookat[:] = (0.005, 0.0, 0.15)
            window.cam.distance = 0.57
            window.cam.azimuth = 145.0
            window.cam.elevation = -19.0

        def show_frame(delay):
            if not window.is_running():
                return
            with window.lock():
                # 把当前的位置、速度和电机输入复制给显示模型。
                mujoco.mj_copyData(display_data, display_model, data)
            # 先刷新画面，再等待指定的真实时间。仿真时间由 mj_step() 推进。
            window.sync(state_only=True)
            sleep(delay)

        yield window, show_frame


class BalancePlant:
    """用同一个 MuJoCo 模型读取前后方向的状态并驱动左右轮。"""

    def __init__(self, model):
        self.model = model
        # data 是正在运行的轨迹。equilibrium 是竖直接地后的快照，供每次试验复位。
        self.data = mujoco.MjData(model)
        self.equilibrium = mujoco.MjData(model)
        prepare_contacts(model, self.equilibrium)
        self.dt = Control_Steps * model.opt.timestep
        self.wheel_radius = float(model.geom("left_contact").size[0])
        # 两轮用同一值，因此取左右电机限幅的交集。
        self.min_torque = float(max(model.actuator_ctrlrange[:, 0]))
        self.max_torque = float(min(model.actuator_ctrlrange[:, 1]))

    def reset(self, state):
        position, velocity, pitch, pitch_rate = state
        # 从同一个接地快照出发，再写入指定的前后状态。中心差分和稳摆试验都用它。
        mujoco.mj_copyData(self.data, self.model, self.equilibrium)
        self.data.time = 0.0
        base = self.data.joint("base_joint")
        base.qpos[0] = position
        # 绕车身 y 轴倾斜。MuJoCo 用四元数 [w, x, y, z] 保存车身朝向。
        base.qpos[3:7] = (cos(pitch / 2), 0.0, sin(pitch / 2), 0.0)
        base.qvel[0] = velocity
        base.qvel[4] = pitch_rate
        # 让车轮从与车速一致的滚动速度开始。车轮铰链读数相对车身，
        # 所以从车轮绝对角速度 v/r 中减去车身俯仰角速度。
        wheel_rate = velocity / self.wheel_radius - pitch_rate
        self.data.joint("left_axle").qvel[0] = wheel_rate
        self.data.joint("right_axle").qvel[0] = wheel_rate
        # 复位后更新由位置和速度计算的派生量，再读出初始状态。
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        base = self.data.joint("base_joint")
        # 在直线试验中，世界 +x 就是车头前方。目标是车身竖直并停在起点，四个状态目标均为 0。
        # s[0] 实测前后位置（m），base.qpos[0]。以起点 x = 0 为基准，向前为正。
        # s[1] 前后速度（m/s），base.qvel[0]。向前移动为正，向后移动为负。
        # s[2] 俯仰角（rad），pitch。以竖直为零，车身顶部向前倾为正。 以车身竖直为 \(0^\circ\)。鸭头相对轮轴往鸭嘴方向倒，叫前倾，角度为正。鸭头往尾巴方向倒，叫后仰，角度为负。
        # s[3] 俯仰角速度（rad/s），base.qvel[4]。取绕车身 y 轴的角速度，向前倾转为正。
        # 下面的 w、x、y、z 是表示车身朝向的四元数分量。
        w, x, y, z = base.qpos[3:7]
        # 把四元数换成俯仰角。clip 防止浮点舍入让 asin 的输入略超出范围。
        pitch = asin(float(np.clip(2 * (w * y - x * z), -1.0, 1.0)))
        # 返回顺序与增益 K 的四列一一对应，扭矩由这四个状态共同决定。
        return np.array([base.qpos[0], base.qvel[0], pitch, base.qvel[4]])

    def step(self, torque):
        # 标量扭矩给两轮同一个值。传入两个值时按左轮、右轮的顺序分别施加。
        # 两种输入都使用从 XML 读取的扭矩范围限幅。
        self.data.ctrl[:] = np.clip(torque, self.min_torque, self.max_torque)
        # 同一扭矩下连续做两次 MuJoCo 物理步，然后给控制器一次新的状态读数。
        mujoco.mj_step(self.model, self.data, nstep=Control_Steps)
        return self.state()


def linearize(plant):
    # 用中心差分建立竖直平衡点附近的离散模型 s_next ≈ A @ s + B @ torque。
    # A 描述原有状态怎样影响下一步。B 描述两轮共同施加的单轮扭矩怎样影响下一步。
    equilibrium = np.zeros(4)
    step = Difference_Step
    a = np.empty((4, 4))
    # 分别往正、负方向微调每个状态，走一个控制周期，估计 A 的一列。
    for index in range(4):
        change = np.zeros(4)
        change[index] = step
        plant.reset(equilibrium + change)
        plus = plant.step(0.0)
        plant.reset(equilibrium - change)
        minus = plant.step(0.0)
        # 两次只改变一个状态，下一步状态的差除以初始差 2 * step，得到这一列。
        a[:, index] = (plus - minus) / (2 * step)

    # 对共同扭矩做同样的差分，得到 B 的一列。
    plant.reset(equilibrium)
    plus = plant.step(step)
    plant.reset(equilibrium)
    minus = plant.step(-step)
    b = ((plus - minus) / (2 * step)).reshape(4, 1)
    return a, b


def load_plant():
    xml = read_model_xml()
    return xml, BalancePlant(mujoco.MjModel.from_xml_string(xml))


def configure_plot():
    from matplotlib import pyplot as plt

    plt.switch_backend("Agg")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    return plt


def style_time_axes(axes, period=None):
    for axis in axes:
        if period is not None:
            axis.axvspan(*period, color="#C99B2D", alpha=0.14)
        axis.axhline(0.0, color="#64748B", ls=":", lw=1)
        axis.grid(alpha=0.22)


def make_speed_reference(duration, dt, *, stop=True):
    steps = round(duration / dt)
    start_step = round(Drive_Start_S / dt)
    stop_step = round(Drive_Stop_S / dt) if stop else steps
    if not 0 <= start_step < stop_step <= steps or (stop and stop_step == steps):
        raise ValueError("行驶时段必须在仿真时间内。")
    # 每行依次为参考位置、参考速度、参考俯仰角、参考俯仰角速度。
    reference = np.zeros((steps + 1, 4))
    speed_end = stop_step if stop else steps + 1
    reference[start_step:speed_end, 1] = Target_Speed_M_S
    reference[1:, 0] = np.cumsum(reference[:-1, 1] * dt)
    return reference, start_step, stop_step


def speed_viewer(window, end_x):
    with window.lock():
        window.cam.lookat[0] = end_x / 2.0
        window.cam.azimuth = 100.0
        window.cam.distance = 0.85 + abs(end_x) / 2.0


def make_arc_reference(duration, dt):
    steps = round(duration / dt)
    start_step = round(Drive_Start_S / dt)
    stop_step = round(Drive_Stop_S / dt)
    if not 0 <= start_step < stop_step < steps:
        raise ValueError("行驶时段必须在仿真时间内，并留出停车观察时间。")
    forward_speeds = np.zeros(steps)
    yaw_rates = np.zeros(steps)
    forward_speeds[start_step:stop_step] = Target_Speed_M_S
    yaw_rates[start_step:stop_step] = Target_Arc_Yaw_Rate_Rad_S

    # 每拍转过 yaw_rate × dt 弧度，再沿这一拍中间时刻的目标车头方向走 speed × dt 米。
    # 把这段位移拆成世界坐标的 x、y 分量，逐拍相加得到参考圆弧。停车后目标位置固定。
    target_yaw = np.r_[0.0, np.cumsum(yaw_rates * dt)]
    mid_yaw = (target_yaw[:-1] + target_yaw[1:]) / 2.0
    direction = np.column_stack((np.cos(mid_yaw), np.sin(mid_yaw)))
    position_steps = direction * (forward_speeds * dt)[:, None]
    positions = np.vstack((np.zeros(2), np.cumsum(position_steps, axis=0)))
    # 第 k 行对应第 k 拍的 [目标 x, 目标 y, 目标前进速度]。positions 额外保留末帧供绘图。
    translation = np.column_stack((positions[:-1], forward_speeds))
    return translation, yaw_rates, positions, start_step, stop_step


def run_speed_trial(
    plant, gain, reference, initial_pitch, fall_pitch, show_frame=None, *, torque_policy=None, torque_limit=None,
    before_step=None,
):
    state = plant.reset((0.0, 0.0, radians(initial_pitch), 0.0))
    times = [0.0]
    states = [state]
    torques = []
    reason = "达到实验时长"
    if show_frame is not None:
        show_frame(0.7)
    start = perf_counter()

    # 最后一行对应仿真结束时的目标。每个控制区间只读取该区间起点的目标。
    for step, target in enumerate(reference[:-1]):
        if torque_policy is None:
            # 四个状态的误差统一为参考值减实际值。
            error = target - state
            requested_torque = (gain @ error).item()
        else:
            # MPC 对照实验也走同一个仿真循环，只把每拍的扭矩计算交给 MPC。
            requested_torque = torque_policy(state, target)
        if torque_limit is not None:
            # 对照实验的共同上限可以比 XML 中的电机上限更小。
            requested_torque = float(np.clip(requested_torque, -torque_limit, torque_limit))
        if before_step is not None:
            # 外力由实验配置写入 MuJoCo，再与当前扭矩一起推进物理仿真。
            before_step(step)
        # 两只轮子收到同一个扭矩。MuJoCo 推进一个控制周期，再读回这一拍真正的状态。
        state = plant.step(requested_torque)
        times.append(float(plant.data.time))
        states.append(state)
        torques.append(float(plant.data.actuator_force[0]))
        if show_frame is not None:
            show_frame(max(0.0, start + plant.data.time - perf_counter()))
        if abs(state[2]) >= radians(fall_pitch):
            reason = "达到俯仰角边界"
            break
    if show_frame is not None:
        show_frame(0.7)
    return np.array(times), np.array(states), torques, reason


def setup_line_viewer(window, end_x):
    with window.lock():
        window.cam.lookat[:] = (end_x / 2.0, 0.0, 0.05)
        window.cam.azimuth = 90.0
        window.cam.elevation = -60.0
        window.cam.distance = 0.95 + end_x / 2.0
        # 把参考线画在 Viewer 的绘图场景里，地面上的标记没有物理碰撞。
        scene = window.user_scn
        mujoco.mjv_initGeom(
            scene.geoms[0], mujoco.mjtGeom.mjGEOM_BOX,
            np.array([end_x / 2.0 + 0.05, 0.004, 0.0005]), np.array([end_x / 2.0, 0.0, 0.001]),
            np.eye(3).ravel(), np.array([0.08, 0.50, 0.76, 1.0], dtype=np.float32),
        )
        scene.ngeom = 1
    window.set_texts((None, None, "Control", "Straight-Line Tracking"))


def setup_arc_viewer(window, positions):
    with window.lock():
        window.cam.lookat[:2] = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
        window.cam.azimuth = 135.0
        window.cam.elevation = -55.0
        window.cam.distance = 0.85 + np.max(np.linalg.norm(positions, axis=1)) / 2.0
        # 用一串蓝色短线在 Viewer 地面上画参考圆弧。它们只用于显示。
        scene = window.user_scn
        sample = np.linspace(0, len(positions) - 1, min(61, len(positions), scene.maxgeom + 1), dtype=int)
        points = np.column_stack((positions[sample], np.full(len(sample), 0.002)))
        geom_type = mujoco.mjtGeom.mjGEOM_CAPSULE
        origin = np.zeros(3)
        rotation = np.eye(3).ravel()
        color = np.array([0.08, 0.50, 0.76, 1.0], dtype=np.float32)
        for index, (first, last) in enumerate(zip(points[:-1], points[1:])):
            geom = scene.geoms[index]
            mujoco.mjv_initGeom(geom, geom_type, origin, origin, rotation, color)
            mujoco.mjv_connector(geom, geom_type, 0.003, first, last)
        scene.ngeom = len(points) - 1
