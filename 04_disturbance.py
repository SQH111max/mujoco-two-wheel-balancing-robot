# 模块 4 · 直线行驶中施加纵向与斜向外力，比较 LQR 和 MPC 的恢复过程。
import argparse
from importlib import import_module
from math import cos, radians, sin

import mujoco
import numpy as np

from controllers import SpeedMpcPolicy, solve_riccati
from two_wheel import (
    Fall_Pitch_Deg, Initial_Pitch_Deg, configure_plot, curve_output_path,
    linearize, load_plant, make_speed_reference, playback, run_speed_trial, setup_line_viewer, style_time_axes,
)

Disturbance_Duration_S = 20.0  # 本模块的仿真时长，给第二次外力留出恢复时间。
Push_Start_S = 5.0
Push_Duration_S = 1.00
Push_Force_N = -0.80  # 世界 x 方向的外力，负值阻碍小车前进。
Oblique_Push_Start_S = 10.0
Oblique_Push_Duration_S = 1.00
Oblique_Push_Force_N = 2.00
Oblique_Push_Angle_Deg = 45.0  # 从正后方朝左偏转的角度。世界 +y 对应直线左侧。
Oblique_Push_Forward_Offset_M = 0.10  # 作用点沿车头方向比质心靠前 10 cm。
Position_Tolerance_M = 0.01
Lateral_Tolerance_M = 0.01
Speed_Tolerance_M_S = 0.015
Pitch_Tolerance_Deg = 1.0
Heading_Tolerance_Deg = 1.0
Recovery_Hold_S = 0.50


def make_pulse(steps, dt, start_s, duration_s, force_n):
    start = round(start_s / dt)
    stop = start + round(duration_s / dt)
    if not 0 <= start < stop < steps:
        raise ValueError("外力时段必须在实验内，并留出恢复时间。")
    forces = np.zeros(steps)
    forces[start:stop] = force_n
    return forces, start, stop


def make_force_schedule(steps, dt):
    axial, first_start, first_stop = make_pulse(steps, dt, Push_Start_S, Push_Duration_S, Push_Force_N)
    magnitude, second_start, second_stop = make_pulse(
        steps, dt, Oblique_Push_Start_S, Oblique_Push_Duration_S, Oblique_Push_Force_N,
    )
    if first_stop >= second_start:
        raise ValueError("两次外力需要按顺序施加，并在中间留出恢复时间。")
    angle = radians(Oblique_Push_Angle_Deg)
    forces = np.column_stack((axial - magnitude * cos(angle), magnitude * sin(angle)))
    windows = ((first_start, first_stop), (second_start, second_stop))
    return forces, windows


def print_force_schedule(forces, windows, dt):
    for index, (start, stop) in enumerate(windows, start=1):
        fx, fy = forces[start]
        print(f"第 {index} 次外力｜{start * dt:.2f}–{stop * dt:.2f} s｜"
              f"Fx {fx:+.3f} N，Fy {fy:+.3f} N")
    print(f"斜推作用点沿车头方向在车身质心前 {100 * Oblique_Push_Forward_Offset_M:g} cm")


def run_trial(plant, gain, reference, forces, initial_pitch, fall_pitch, show_frame=None, *, torque_policy=None):
    if len(forces) != len(reference) - 1:
        raise ValueError("外力时序需要和每拍参考状态对齐。")
    # xfrc_applied 的前三项是世界坐标下施加在刚体质心的力 (Fx, Fy, Fz)。
    body_id = plant.model.body("chassis").id

    def before_step(step):
        plant.data.xfrc_applied[body_id, 0] = forces[step]

    try:
        return run_speed_trial(
            plant, gain, reference, initial_pitch, fall_pitch, show_frame,
            torque_policy=torque_policy, before_step=before_step,
        )
    finally:
        plant.data.xfrc_applied[body_id, 0] = 0.0


def run_planar_trial(
    plant, gain, path, reference, drive_start, forces, oblique_window, show_frame=None, *, common_torque_policy=None,
):
    if forces.shape != (len(reference) - 1, 2):
        raise ValueError("每个控制周期需要一对纵向、横向外力。")
    # 参考位置沿世界 x 轴前进，参考 y 为零。转向环把侧向偏差拉回直线。
    translation = np.column_stack((reference[:-1, 0], np.zeros(len(forces)), reference[:-1, 1]))
    chassis = plant.data.body("chassis")
    body_force = plant.data.xfrc_applied[chassis.id]
    base = plant.data.joint("base_joint")
    local_lever_arm = np.array([Oblique_Push_Forward_Offset_M, 0.0, 0.0])
    world_lever_arm = np.empty(3)
    oblique_start, oblique_stop = oblique_window

    def before_step(step):
        # xfrc_applied 在质心施力。斜推的作用点靠前，因此用力臂叉乘外力得到等效力矩。
        body_force[:2] = forces[step]
        if oblique_start <= step < oblique_stop:
            # 用当前车身四元数把局部力臂转为世界坐标，再和世界坐标下的推力求叉乘。
            mujoco.mju_rotVecQuat(world_lever_arm, local_lever_arm, base.qpos[3:7])
            body_force[3:] = np.cross(world_lever_arm, body_force[:3])
        else:
            body_force[3:] = 0.0

    try:
        return path.run_line_trial(
            plant, gain, translation, drive_start, Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame,
            initial_y_m=0.0, before_step=before_step, common_torque_policy=common_torque_policy,
        )
    finally:
        body_force[:] = 0.0


def get_recovery_time(times, outside, dt):
    # 取撤力后的最后一次越界。下一拍起要留在容差内，直至该段观察结束且至少持续半秒。
    first_inside = outside[-1] + 1 if outside.size else 0
    if first_inside < len(times) and times[-1] - times[first_inside] >= Recovery_Hold_S - dt * 1e-6:
        return times[first_inside] - times[0]
    return None


def measure(trial, reference, push_start, push_stop, dt):
    times, states, _, _ = trial
    if len(times) <= push_stop:
        return None
    errors = reference[:len(states)] - states
    recovery = errors[push_stop:]
    outside = np.flatnonzero(
        (np.abs(recovery[:, 0]) > Position_Tolerance_M)
        | (np.abs(recovery[:, 1]) > Speed_Tolerance_M_S)
        | (np.abs(recovery[:, 2]) > np.deg2rad(Pitch_Tolerance_Deg))
    )
    return {
        "peak_position_cm": 100 * np.max(np.abs(errors[push_start:, 0])),
        "peak_speed_m_s": np.max(np.abs(errors[push_start:, 1])),
        "peak_pitch_deg": np.rad2deg(np.max(np.abs(errors[push_start:, 2]))),
        "position_rmse_cm": 100 * np.sqrt(np.mean(recovery[:, 0] ** 2)),
        "recovery_s": get_recovery_time(times[push_stop:], outside, dt),
    }


def print_results(trials, reference, push_start, push_stop, dt):
    for label, _, trial in trials:
        times, _, torques, reason = trial
        print(f"{label} | {reason} | {times[-1]:.2f} s | 峰值单轮扭矩 {max(map(abs, torques)):.4f} N·m")
        metrics = measure(trial, reference, push_start, push_stop, dt)
        if metrics is None:
            continue
        recovered = metrics["recovery_s"]
        recovered_text = "未达到恢复判据" if recovered is None else f"撤力后 {recovered:.2f} s 恢复"
        print(f"  最大位置偏差 {metrics['peak_position_cm']:.3f} cm，最大速度误差 {metrics['peak_speed_m_s']:.4f} m/s，"
              f"最大绝对俯仰角 {metrics['peak_pitch_deg']:.2f}°")
        print(f"  撤力后位置均方根误差 {metrics['position_rmse_cm']:.3f} cm，{recovered_text}")


def plot_results(trials, reference, forces, push_start, push_stop, dt, filename):
    plt = configure_plot()
    fig, (force_ax, pos_ax, speed_ax, pitch_ax, torque_ax) = plt.subplots(
        5, 1, sharex=True, figsize=(9, 9), layout="constrained",
    )
    fig.suptitle("双轮小黄鸭直线行驶抗扰实验")
    style_time_axes((force_ax, pos_ax, speed_ax, pitch_ax, torque_ax), (push_start * dt, push_stop * dt))
    force_ax.stairs(forces, np.arange(len(reference)) * dt, color="#765400", lw=1.7, baseline=None)
    for index, (label, color, (times, states, torques, _)) in enumerate(trials):
        # 图中的位置和速度误差都按参考值减实测值计算。
        errors = reference[:len(states)] - states
        style = dict(color=color, ls="-" if index == 0 else "-.", lw=1.8, label=label)
        pos_ax.plot(times, errors[:, 0] * 100, **style)
        speed_ax.plot(times, errors[:, 1], **style)
        pitch_ax.plot(times, np.rad2deg(states[:, 2]), **style)
        torque_ax.stairs(torques, times, baseline=None, **style)
    force_ax.set_ylabel("车身外力（N）")
    pos_ax.set_ylabel("位置误差（cm）")
    pos_ax.legend(frameon=False)
    speed_ax.set_ylabel("前进速度误差（m/s）")
    pitch_ax.set_ylabel("俯仰角（°）")
    torque_ax.set(xlabel="时间（秒）", ylabel="单轮扭矩（N·m）", xlim=(0.0, len(forces) * dt))
    output = curve_output_path(filename)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def planar_errors(path, result, reference):
    # 有符号误差统一为参考减实际。直线的参考横向位置、航向和俯仰角均为零。
    positions = result["positions_xy"]
    reference = reference[:len(positions)]
    return {
        "x": reference[:, 0] - positions[:, 0],
        "y": -positions[:, 1],
        "speed": reference[:, 1] - result["forward_speeds"],
        "heading": np.rad2deg(path.angle_error(0.0, result["yaw_angles"])),
        "pitch": np.rad2deg(reference[:, 2] - result["pitches"]),
    }


def measure_planar(times, errors, push_start, push_stop, observation_end, dt):
    # 第一次恢复只统计到第二次施力前的状态。第二次则观察至仿真结束。
    end = min(observation_end, len(times) - 1)
    if end < push_stop:
        return None
    after_release = slice(push_stop, end + 1)
    affected = slice(push_start, end + 1)
    tolerances = (
        ("x", Position_Tolerance_M), ("y", Lateral_Tolerance_M), ("speed", Speed_Tolerance_M_S),
        ("heading", Heading_Tolerance_Deg), ("pitch", Pitch_Tolerance_Deg),
    )
    outside = np.flatnonzero(np.logical_or.reduce([
        np.abs(errors[key][after_release]) > tolerance for key, tolerance in tolerances
    ]))
    peaks = {key: np.max(np.abs(series[affected])) for key, series in errors.items()}
    return peaks, get_recovery_time(times[after_release], outside, dt)


def print_planar_results(trials, windows, total_steps, dt):
    for label, _, result, reason, errors in trials:
        times = result["times"]
        peak_torque = np.max(np.abs(result["wheel_torques"]))
        print(f"{label} + 转向 PID | {reason} | {times[-1]:.2f} s | 峰值单轮扭矩 {peak_torque:.4f} N·m")
        observation_ends = (windows[1][0], total_steps)
        for index, ((start, stop), end) in enumerate(zip(windows, observation_ends), start=1):
            measurement = measure_planar(times, errors, start, stop, end, dt)
            if measurement is None:
                print(f"  第 {index} 次外力｜仿真在撤力前结束")
                continue
            peaks, recover = measurement
            text = "未达到恢复判据" if recover is None else f"撤力后 {recover:.2f} s 恢复"
            print(f"  第 {index} 次外力｜{text}｜最大纵向偏差 {100 * peaks['x']:.3f} cm，"
                  f"横向偏差 {100 * peaks['y']:.3f} cm")
            print(f"    最大速度误差 {peaks['speed']:.4f} m/s，航向误差 {peaks['heading']:.2f}°，"
                  f"最大绝对俯仰角 {peaks['pitch']:.2f}°")


def plot_planar_results(trials, forces, windows, dt, *, filename="04_disturbance_lqr_vs_mpc.png"):
    plt = configure_plot()
    fig, axes = plt.subplot_mosaic(
        [["force", "force"], ["x", "y"], ["speed", "heading"], ["pitch", "torque"]],
        sharex=True, figsize=(11, 9), layout="constrained",
    )
    fig.suptitle("双轮小黄鸭抗扰对照 · 纵向推力与斜向推力")
    style_time_axes(axes.values())
    for index, (start, stop) in enumerate(windows):
        for axis in axes.values():
            axis.axvspan(start * dt, stop * dt, color=("#C99B2D", "#5890BB")[index], alpha=0.20)
    edges = np.arange(len(forces) + 1) * dt
    axes["force"].stairs(forces[:, 0], edges, baseline=None, color="#765400", lw=1.8, label="纵向 Fx")
    axes["force"].stairs(forces[:, 1], edges, baseline=None, color="#196CB1", lw=1.8, ls="--", label="横向 Fy")
    for index, (label, color, result, _, errors) in enumerate(trials):
        times = result["times"]
        line = dict(color=color, ls="-" if index == 0 else "-.", lw=1.8, label=label)
        for key in ("x", "y", "speed", "heading"):
            scale = 100.0 if key in ("x", "y") else 1.0
            axes[key].plot(times, scale * errors[key], **line)
        axes["pitch"].plot(times, np.rad2deg(result["pitches"]), **line)
        max_torques = np.max(np.abs(result["wheel_torques"]), axis=1)
        axes["torque"].stairs(max_torques, times, baseline=None, **line)
    for key, label in (
        ("force", "车身外力（N）"), ("x", "纵向位置误差（cm）"), ("y", "横向位置误差（cm）"),
        ("speed", "前进速度误差（m/s）"), ("heading", "航向误差（°）"), ("pitch", "俯仰角（°）"),
        ("torque", "两轮中较大的绝对扭矩（N·m）"),
    ):
        axes[key].set_ylabel(label)
    for key in ("force", "x"):
        axes[key].legend(frameon=False)
    for key in ("pitch", "torque"):
        axes[key].set(xlabel="时间（秒）", xlim=(0.0, edges[-1]))
    output = curve_output_path(filename)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def main():
    parser = argparse.ArgumentParser(description="在 20 秒直行实验中施加纵向与斜向外力，对照 LQR 和 MPC。")
    parser.add_argument("--viewer", action="store_true", help="在 MuJoCo Viewer 回放。")
    args = parser.parse_args()
    xml, plant = load_plant()
    a, b = linearize(plant)
    terminal_weight, gain = solve_riccati(a, b)
    path = import_module("03_path_tracking")
    torque_limit = min(plant.max_torque, -plant.min_torque)
    mpc_policy = SpeedMpcPolicy(plant, a, b, gain, terminal_weight, torque_limit)
    reference, drive_start, _ = make_speed_reference(Disturbance_Duration_S, plant.dt, stop=False)
    total_steps = len(reference) - 1
    forces, windows = make_force_schedule(total_steps, plant.dt)

    def common_mpc_torque(state, target_speed):
        # 以当拍的参考点为局部纵向坐标原点，预测共同扭矩。转向与 LQR 组共用同一个 PID。
        return mpc_policy(state, (0.0, target_speed, 0.0, 0.0))

    trials = []
    print_force_schedule(forces, windows, plant.dt)
    with playback(xml, plant.data, args.viewer) as (window, show_frame):
        if window is not None and window.is_running():
            setup_line_viewer(window, reference[-1, 0])
        for label, color, policy in (("LQR", "#007872", None), ("MPC", "#73459E", common_mpc_torque)):
            if window is not None and window.is_running():
                window.set_texts((None, None, "Control", f"{label} + Yaw PID | Two pushes"))
            result, reason = run_planar_trial(
                plant, gain, path, reference, drive_start, forces, windows[1], show_frame, common_torque_policy=policy,
            )
            trials.append((label, color, result, reason, planar_errors(path, result, reference)))
    print_planar_results(trials, windows, total_steps, plant.dt)
    plot_planar_results(trials, forces, windows, plant.dt)


if __name__ == "__main__":
    main()
