# 模块 2 · 转向、直线启停、边走边转和直线 LQR/MPC 对照。
import argparse
from math import radians

import numpy as np

from controllers import (
    Derivative_Filter_S, Prediction_Horizon, SpeedMpcPolicy, Yaw_Kd, Yaw_Ki,
    run_yaw_trial, solve_riccati,
)
from two_wheel import (
    Duration_S, Fall_Pitch_Deg, Hold_Duration_S, Initial_Pitch_Deg,
    Pitch_Tolerance_Deg, Position_Tolerance_M, Target_Arc_Yaw_Rate_Rad_S, Target_Speed_M_S,
    configure_plot, curve_output_path, linearize, load_plant, make_arc_reference,
    make_speed_reference, playback, run_speed_trial, speed_viewer, style_time_axes,
)

Turn_Start_S = 2.0
Turn_Stop_S = 4.0
Target_Yaw_Rate_Rad_S = 0.50
Stop_Rate_Tolerance_Rad_S = 0.02
Speed_Tolerance_M_S = 0.01
Tight_Torque_Limit_Nm = 0.015


def make_yaw_reference(duration, dt):
    # 用控制拍数定义转向时段。默认第 2 到第 4 秒连续给 200 拍指令。
    steps = round(duration / dt)
    start_step = round(Turn_Start_S / dt)
    stop_step = round(Turn_Stop_S / dt)
    if not 0 <= start_step < stop_step < steps:
        raise ValueError("转向时段必须在仿真时间内，并留出停转的观察时间。")
    reference_rates = np.zeros(steps)
    reference_rates[start_step:stop_step] = Target_Yaw_Rate_Rad_S
    return reference_rates, start_step * dt, stop_step * dt


def print_yaw_results(result, reason, turn_start, turn_stop, hold_duration, label="转向 P 环"):
    times = result["times"]
    yaw_rates = result["yaw_rates"]
    pitch_deg = np.rad2deg(result["pitches"])
    print(f"{label} | {reason} | 仿真 {times[-1]:.2f} s")

    tail = (times > max(turn_start, turn_stop - 0.5) + 1e-9) & (times <= turn_stop + 1e-9)
    if times[-1] >= turn_stop - 1e-9 and np.any(tail):
        # 用撤指令前最后 0.5 秒的平均值看持续偏差，避开刚开始转向的过渡过程。
        mean_rate = float(np.mean(yaw_rates[tail]))
        print(f"  目标偏航角速度 {Target_Yaw_Rate_Rad_S:+.3f} rad/s，撤指令前平均 {mean_rate:+.4f} rad/s")
        print(f"  平均速度误差 {Target_Yaw_Rate_Rad_S - mean_rate:+.4f} rad/s")

        # 撤指令后找最后一次偏航角速度越出容差带的采样点。
        after_release = np.flatnonzero(times >= turn_stop - 1e-9)
        after_release = after_release[np.abs(yaw_rates[after_release]) > Stop_Rate_Tolerance_Rad_S]
        start = after_release[-1] + 1 if after_release.size else int(np.searchsorted(times, turn_stop - 1e-9))
        if start < len(times) and times[-1] - times[start] >= hold_duration - 1e-9:
            print(f"  撤指令后 {max(0.0, times[start] - turn_stop):.2f} s 进入停转容差带")
        else:
            print("  撤指令后未达到停转判据")

    during_turn = times >= turn_start - 1e-9
    if np.any(during_turn):
        print(f"  发出转向指令后的最大绝对俯仰角 {max(abs(pitch_deg[during_turn])):.3f}°")
    print(f"  最终偏航角 {np.rad2deg(result['yaw_angles'][-1]):+.2f}°")
    print(f"  峰值单轮实际扭矩 {np.max(np.abs(result['wheel_torques'])):.4f} N·m")


def print_turning_peaks(trials, turn_start, turn_stop):
    for label, _, result in trials:
        times = result["times"]
        # 测量点对应刚结束的控制拍，因此包含撤令时读到的最后一个转向速度。
        turning = (times > turn_start + 1e-9) & (times <= turn_stop + 1e-9)
        if not np.any(turning):
            continue
        first_step = int(np.searchsorted(times, turn_start - 1e-9))
        target = result["reference_rates"][first_step]
        # 统一在指令方向上计算峰值，左转和右转都可用。
        peak = float(np.max(np.sign(target) * result["yaw_rates"][turning]))
        overshoot = max(0.0, peak - abs(target))
        print(f"{label} | 转向时段同向速度峰值 {peak:.4f} rad/s，超出目标 {overshoot:.4f} rad/s")


def plot_yaw_comparison(trials, turn_start, turn_stop):
    plt = configure_plot()
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8), layout="constrained")
    (rate_ax, yaw_ax), (pitch_ax, wheel_ax), (i_ax, d_ax) = axes
    fig.suptitle("双轮小黄鸭偏航速度跟踪 · P、PI 与 PID")
    style_time_axes(axes.flat, (turn_start, turn_stop))
    reference = max((result for _, _, result in trials), key=lambda item: len(item["times"]))
    rate_ax.stairs(reference["reference_rates"], reference["times"], baseline=None,
                   label="目标", color="#475569", lw=1.5, ls="--")
    for label, color, result in trials:
        times = result["times"]
        rate_ax.plot(times, result["yaw_rates"], color=color, label=label, lw=1.8)
        yaw_ax.plot(times, np.rad2deg(result["yaw_angles"]), color=color, lw=1.8)
        pitch_ax.plot(times, np.rad2deg(result["pitches"]), color=color, lw=1.8)
        wheel_ax.stairs(result["wheel_torques"][:, 0], times, baseline=None, color=color, lw=1.6)
        wheel_ax.stairs(result["wheel_torques"][:, 1], times, baseline=None, color=color, lw=1.6, ls="--")
        i_ax.stairs(result["integral_torques"], times, baseline=None, color=color, lw=1.8)
        d_ax.stairs(result["derivative_torques"], times, baseline=None, color=color, lw=1.8)
    rate_ax.legend(loc="upper right", ncol=2, frameon=False)
    for axis, label in zip(axes.flat, (
        "偏航角速度（rad/s）", "偏航角（°）", "俯仰角（°）", "单轮扭矩（N·m）",
        "I 项扭矩（N·m）", "D 项扭矩（N·m）",
    )):
        axis.set(xlabel="时间（秒）", ylabel=label, xlim=(0.0, reference["times"][-1]))
    wheel_ax.set_title("实线为左轮 · 虚线为右轮")
    output = curve_output_path("02_yaw_comparison.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def print_speed_results(trial, reference, start_step, stop_step, dt):
    times, states, torques, reason = trial
    start_time = start_step * dt
    stop_time = stop_step * dt
    target_position = reference[stop_step, 0]
    print(f"直线 LQR | {reason} | 仿真 {times[-1]:.2f} s")
    print(f"  {start_time:.2f}–{stop_time:.2f} s 的目标速度 {Target_Speed_M_S:+.2f} m/s")

    if stop_step < len(times):
        # 最后一帧的速度是撤指令前最后一拍跑出来的，所以把它计入行驶末段。
        tail = (times > max(start_time, stop_time - 0.5) + 1e-9) & (times <= stop_time + 1e-9)
        mean_speed = float(np.mean(states[tail, 1]))
        print(f"  行驶末 0.5 秒的平均车速 {mean_speed:+.4f} m/s，误差 {Target_Speed_M_S - mean_speed:+.4f} m/s")

        drive_speeds = states[start_step + 1:stop_step + 1, 1]
        speed_error = Target_Speed_M_S - drive_speeds
        print(f"  行驶时段速度均方根误差 {np.sqrt(np.mean(speed_error ** 2)):.4f} m/s")
        outside_speed = np.flatnonzero(np.abs(speed_error) > Speed_Tolerance_M_S)
        first_in_band = outside_speed[-1] + 1 if outside_speed.size else 0
        if first_in_band < len(drive_speeds):
            tracking_time = times[start_step + 1 + first_in_band] - start_time
            print(f"  发指令后 {tracking_time:.2f} s 进入速度容差带，并维持到停车指令")
        else:
            print("  速度尚未持续进入容差带")

        # 停车时要求同时到达新目标位置、速度接近零、车身接近竖直。
        # 最后一次越出任一容差带之后，取下一个采样点作为停车时间。
        stopped_states = states[stop_step:]
        outside = np.flatnonzero(
            (np.abs(target_position - stopped_states[:, 0]) > Position_Tolerance_M)
            | (np.abs(stopped_states[:, 1]) > Speed_Tolerance_M_S)
            | (np.abs(stopped_states[:, 2]) > radians(Pitch_Tolerance_Deg))
        )
        first_stopped = stop_step + outside[-1] + 1 if outside.size else stop_step
        if first_stopped < len(times) and times[-1] - times[first_stopped] >= Hold_Duration_S - 1e-9:
            print(f"  撤指令后 {max(0.0, times[first_stopped] - stop_time):.2f} s 达到停车判据")
        else:
            print("  撤指令后未达到停车判据")

        # 把位置投影到这组指令的行驶方向，正向或反向行驶都能按同一方式计算超调。
        overshoot = max(0.0, float(np.max(np.sign(Target_Speed_M_S) * (stopped_states[:, 0] - target_position))))
        print(f"  停车过程最大位置超调 {overshoot * 100:.2f} cm")
    print(f"  目标位置 {target_position * 100:+.2f} cm，最终位置 {states[-1, 0] * 100:+.2f} cm")
    print(f"  最终车速 {states[-1, 1]:+.4f} m/s，峰值单轮扭矩 {max(map(abs, torques)):.4f} N·m")
    if start_step < len(times):
        peak_pitch = np.rad2deg(np.max(np.abs(states[start_step:, 2])))
        print(f"  发出行驶指令后的最大绝对俯仰角 {peak_pitch:.3f}°")


def print_comparison_results(label, trial, reference, start_step, stop_step, torque_limit):
    times, states, torques, reason = trial
    at_limit = np.count_nonzero(np.isclose(np.abs(torques), torque_limit, atol=1e-7, rtol=0.0))
    print(f"  {label} | {reason} | {times[-1]:.2f} s | 实际触及扭矩边界 {at_limit} 拍")
    if stop_step >= len(states):
        return
    # 第 k 拍的车速测量值记在该拍结束时的第 k+1 行。
    speed_errors = reference[start_step:stop_step, 1] - states[start_step + 1:stop_step + 1, 1]
    speed_rmse = np.sqrt(np.mean(speed_errors ** 2))
    target_position = reference[stop_step, 0]
    direction = np.sign(reference[start_step, 1])
    overshoot = 100 * np.max(direction * (states[stop_step:, 0] - target_position), initial=0.0)
    position_error = 100 * abs(target_position - states[-1, 0])
    peak_pitch = np.rad2deg(np.max(np.abs(states[start_step:, 2])))
    print(f"    行驶段速度均方根误差 {speed_rmse:.5f} m/s，停车最大超程 {overshoot:.3f} cm，"
          f"末端位置误差 {position_error:.3f} cm")
    print(f"    发出行驶指令后最大绝对俯仰角 {peak_pitch:.2f}°，"
          f"峰值单轮扭矩 {max(map(abs, torques)):.4f} N·m")


def plot_speed_cases(cases, reference, start_step, stop_step, dt):
    plt = configure_plot()
    fig, axes = plt.subplots(
        4, len(cases), sharex=True, squeeze=False, figsize=(5.5 * len(cases) + 1.5, 8.5),
        layout="constrained",
    )
    compare = len(cases) > 1
    fig.suptitle("双轮小黄鸭直线速度跟踪 · " + ("LQR 与 50 步 MPC" if compare else "LQR"))
    target_time = np.arange(len(reference)) * dt
    target_style = dict(color="#64748B", ls="--", lw=1.5)
    period = (start_step * dt, stop_step * dt)
    for column, (case_label, limit, trials) in enumerate(cases):
        speed_ax, position_ax, pitch_ax, torque_ax = axes[:, column]
        style_time_axes(axes[:, column], period)
        if compare:
            speed_ax.set_title(f"{case_label} · 单轮 ±{limit:.3f} N·m")
        speed_ax.stairs(reference[:-1, 1], target_time, baseline=None, label="目标", **target_style)
        position_ax.plot(target_time, reference[:, 0] * 100, **target_style)
        for label, color, style, (times, states, torques, _) in trials:
            line = dict(color=color, ls=style, lw=1.7)
            speed_ax.plot(times, states[:, 1], label=label, **line)
            position_ax.plot(times, states[:, 0] * 100, **line)
            pitch_ax.plot(times, np.rad2deg(states[:, 2]), **line)
            torque_ax.stairs(torques, times, baseline=None, **line)
        if column == 1:
            for boundary in (-limit, limit):
                torque_ax.axhline(boundary, color="#64748B", ls=":", lw=1)
        speed_ax.legend(loc="upper right", ncol=3, frameon=False)
        torque_ax.set(xlabel="时间（秒）", xlim=(0.0, target_time[-1]))
    for axis, label in zip(axes[:, 0], ("前进速度（m/s）", "前后位置（cm）", "俯仰角（°）", "单轮扭矩（N·m）")):
        axis.set_ylabel(label)
    output = curve_output_path("02_speed_lqr_vs_mpc.png" if compare else "02_speed_tracking.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def print_drive_results(result, reason, positions, start_step, stop_step, dt):
    times = result["times"]
    print(f"前进 LQR + 转向 PID | {reason} | 仿真 {times[-1]:.2f} s")
    if stop_step < len(times):
        # 撤指令前最后一拍的实测值记在 stop_step，统计最后 0.5 秒的平均值。
        tail = slice(max(start_step, stop_step - round(0.5 / dt)) + 1, stop_step + 1)
        print(f"  行驶末 0.5 s 平均前进速度 {np.mean(result['forward_speeds'][tail]):+.4f} m/s"
              f"（目标 {Target_Speed_M_S:+.2f} m/s）")
        print(f"  行驶末 0.5 s 平均偏航角速度 {np.mean(result['yaw_rates'][tail]):+.4f} rad/s"
              f"（目标 {Target_Arc_Yaw_Rate_Rad_S:+.2f} rad/s）")
    if start_step < len(times):
        peak_pitch = np.rad2deg(np.max(np.abs(result["pitches"][start_step:])))
        print(f"  发出行驶指令后最大绝对俯仰角 {peak_pitch:.2f}°")
    target_cm = 100 * positions[stop_step]
    actual_cm = 100 * result["positions_xy"][-1]
    print(f"  参考终点 ({target_cm[0]:.2f}, {target_cm[1]:.2f}) cm，"
          f"最终车位 ({actual_cm[0]:.2f}, {actual_cm[1]:.2f}) cm，"
          f"相距 {np.linalg.norm(actual_cm - target_cm):.2f} cm")
    print(f"  结束时前进速度 {result['forward_speeds'][-1]:+.4f} m/s，"
          f"偏航角速度 {result['yaw_rates'][-1]:+.4f} rad/s，"
          f"峰值单轮扭矩 {np.max(np.abs(result['wheel_torques'])):.4f} N·m")


def plot_drive_results(result, translation, positions, motion_start, motion_stop):
    plt = configure_plot()
    times = result["times"]
    target_cm = positions[:len(times)] * 100
    actual_cm = result["positions_xy"] * 100
    fig, ((path_ax, speed_ax), (yaw_ax, pitch_ax)) = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    fig.suptitle("双轮小黄鸭边走边转 · 前进 LQR + 转向 PID")
    style_time_axes((speed_ax, yaw_ax, pitch_ax), (motion_start, motion_stop))
    for axis in (speed_ax, yaw_ax, pitch_ax):
        axis.set(xlabel="时间（秒）", xlim=(0.0, times[-1]))
    path_ax.grid(alpha=0.22)

    # XY 图用相同的横、纵比例，画出的弯道形状才和实际尺度一致。
    path_ax.plot(*target_cm.T, color="#7A5700", ls="--", lw=1.7, label="参考路线")
    path_ax.plot(*actual_cm.T, color="#007872", lw=2, label="实际路线")
    path_ax.scatter(*target_cm[-1], facecolors="none", edgecolors="#7A5700", zorder=3)
    path_ax.scatter(*actual_cm[-1], color="#007872", s=21, zorder=3)
    path_ax.set(xlabel="世界 x（cm）", ylabel="世界 y（cm）", title="俯视轨迹")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.legend(loc="upper left", frameon=False)

    speed_ax.stairs(translation[:len(times) - 1, 2], times, baseline=None,
                    color="#7A5700", ls="--", lw=1.7, label="目标")
    speed_ax.plot(times, result["forward_speeds"], color="#007872", lw=2, label="实测")
    speed_ax.set_ylabel("前进速度（m/s）")
    speed_ax.legend(loc="upper right", frameon=False)
    yaw_ax.stairs(result["reference_rates"], times, baseline=None,
                  color="#7A5700", ls="--", lw=1.7, label="目标")
    yaw_ax.plot(times, result["yaw_rates"], color="#007872", lw=2, label="实测")
    yaw_ax.set_ylabel("偏航角速度（rad/s）")
    yaw_ax.legend(loc="upper right", frameon=False)
    pitch_ax.plot(times, np.rad2deg(result["pitches"]), color="#563E91", lw=2)
    pitch_ax.set_ylabel("俯仰角（°）")

    output = curve_output_path("02_drive_and_turn.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def yaw_experiment(plant, gain, window, show_frame):
    rates, turn_start, turn_stop = make_yaw_reference(Duration_S, plant.dt)
    trials = []
    controllers = (
        ("P", "#B65E2A", 0.0, 0.0), ("PI", "#007872", Yaw_Ki, 0.0),
        ("PID", "#73459E", Yaw_Ki, Yaw_Kd),
    )
    for label, color, ki, kd in controllers:
        if window is not None and window.is_running():
            window.set_texts((None, None, "Control", f"LQR + Yaw {label}"))
        result, reason = run_yaw_trial(
            plant, gain, rates, Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame,
            yaw_ki=ki, yaw_kd=kd, derivative_filter_s=Derivative_Filter_S,
        )
        print_yaw_results(result, reason, turn_start, turn_stop, Hold_Duration_S, label=f"转向 {label} 环")
        if ki != 0.0 and result["times"][-1] >= turn_stop - 1e-9:
            stop_step = int(np.searchsorted(result["times"], turn_stop - 1e-9))
            print(f"  撤指令前 I 项扭矩 {result['integral_torques'][stop_step - 1]:+.5f} N·m")
        trials.append((label, color, result))
    print_turning_peaks(trials, turn_start, turn_stop)
    plot_yaw_comparison(trials, turn_start, turn_stop)


def speed_experiment(plant, a, b, terminal_weight, gain, window, show_frame, compare):
    reference, start_step, stop_step = make_speed_reference(Duration_S, plant.dt)
    if window is not None and window.is_running():
        speed_viewer(window, reference[stop_step, 0])
    limit = min(plant.max_torque, -plant.min_torque)
    options = (("常规限幅", limit),)
    if compare:
        if not 0.0 < Tight_Torque_Limit_Nm < limit:
            raise ValueError("紧限幅必须大于零，并小于模型的单轮扭矩上限。")
        options += (("紧限幅", Tight_Torque_Limit_Nm),)
        print(f"预测时域 {Prediction_Horizon} 步，{Prediction_Horizon * plant.dt:.2f} s")
    cases = []
    for case_label, torque_limit in options:
        if window is not None and window.is_running():
            window.set_texts((None, None, "Control", f"LQR | {torque_limit:g} Nm"))
        lqr_trial = run_speed_trial(
            plant, gain, reference, Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame, torque_limit=torque_limit,
        )
        trials = [("LQR", "#007872", "-", lqr_trial)]
        if compare:
            policy = SpeedMpcPolicy(plant, a, b, gain, terminal_weight, torque_limit)
            if window is not None and window.is_running():
                window.set_texts((None, None, "Control", f"MPC | {torque_limit:g} Nm"))
            mpc_trial = run_speed_trial(
                plant, gain, reference, Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame,
                torque_policy=policy, torque_limit=torque_limit,
            )
            trials.append(("MPC", "#73459E", "-.", mpc_trial))
            print(f"{case_label} | 单轮扭矩范围 ±{torque_limit:.3f} N·m")
            for label, _, _, trial in trials:
                print_comparison_results(label, trial, reference, start_step, stop_step, torque_limit)
            print(f"    MPC 有 {policy.constrained_solves}/{len(mpc_trial[2])} 拍的预测序列需要约束优化。")
        else:
            print_speed_results(lqr_trial, reference, start_step, stop_step, plant.dt)
        cases.append((case_label, torque_limit, trials))
    plot_speed_cases(cases, reference, start_step, stop_step, plant.dt)


def drive_turn_experiment(plant, gain, window, show_frame):
    translation, yaw_rates, positions, start_step, stop_step = make_arc_reference(Duration_S, plant.dt)
    if window is not None and window.is_running():
        with window.lock():
            window.cam.lookat[:2] = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
            window.cam.azimuth = 135.0
            window.cam.elevation = -50.0
            window.cam.distance = 0.85 + np.max(np.linalg.norm(positions, axis=1)) / 2.0
        window.set_texts((None, None, "Control", "LQR Drive + PID Turn"))
    result, reason = run_yaw_trial(
        plant, gain, yaw_rates, Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame,
        yaw_ki=Yaw_Ki, yaw_kd=Yaw_Kd, derivative_filter_s=Derivative_Filter_S,
        translation_reference=translation,
    )
    print_drive_results(result, reason, positions, start_step, stop_step, plant.dt)
    plot_drive_results(result, translation, positions, start_step * plant.dt, stop_step * plant.dt)


def main():
    parser = argparse.ArgumentParser(description="运行转向、直线速度或边走边转实验。")
    parser.add_argument("experiment", nargs="?", default="yaw",
                        choices=("yaw", "speed", "speed-compare", "drive-turn"))
    parser.add_argument("--viewer", action="store_true", help="在 MuJoCo Viewer 回放。")
    args = parser.parse_args()
    xml, plant = load_plant()
    a, b = linearize(plant)
    terminal_weight, gain = solve_riccati(a, b)
    with playback(xml, plant.data, args.viewer) as (window, show_frame):
        if args.experiment == "yaw":
            yaw_experiment(plant, gain, window, show_frame)
        elif args.experiment == "drive-turn":
            drive_turn_experiment(plant, gain, window, show_frame)
        else:
            speed_experiment(plant, a, b, terminal_weight, gain, window, show_frame, args.experiment == "speed-compare")


if __name__ == "__main__":
    main()
