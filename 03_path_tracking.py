# 模块 3 · 直线、圆弧纠偏和圆弧 LQR/MPC 对照。
import argparse
from itertools import chain, repeat
from math import ceil, cos, remainder, sin, tau

import numpy as np

from controllers import (
    Derivative_Filter_S, Prediction_Horizon, SpeedMpcPolicy, Yaw_Kd, Yaw_Ki,
    run_yaw_trial, solve_riccati,
)
from two_wheel import (
    Duration_S, Fall_Pitch_Deg, Initial_Pitch_Deg, Target_Arc_Yaw_Rate_Rad_S, Target_Speed_M_S,
    configure_plot, curve_output_path, linearize, load_plant, make_arc_reference, make_speed_reference,
    playback, setup_arc_viewer, setup_line_viewer, style_time_axes,
)

Line_Initial_Lateral_Offset_M = 0.03  # 从直线左侧 3 cm 起步。
Line_Lateral_Kp = 10.0
Line_Heading_Kp = 2.0
Line_Max_Yaw_Rate_Rad_S = 0.50
Arc_Initial_Lateral_Offset_M = 0.0
Arc_Lateral_Kp = 20.0
Arc_Heading_Kp = 3.0
Arc_Max_Yaw_Rate_Rad_S = 0.80
S_Curve_Radii_M = (0.40, 0.50, 0.80)  # 三段 S 形的圆弧半径，单位 m。
S_Curve_Arc_Deg = 60.0  # 每个小圆弧的转角，每段 S 形由两个反向圆弧组成。
S_Curve_Start_S = 0.80  # 从第一段起点扶正后开始行驶。
S_Curve_Cruise_Speed_M_S = 0.38  # 延用原来约 0.38 m/s 的参考速度，行驶时长随弧长而变。
S_Curve_Ramp_S = 1.00
S_Curve_Max_Yaw_Rate_Rad_S = 1.30
S_Curve_Position_Tolerance_M = 0.02
S_Curve_Stop_Speed_M_S = 0.01
S_Curve_Stop_Hold_S = 0.50


def correction_rate(actual_y, actual_yaw):
    # 本实验的参考线是 y = 0，路径朝向是 0 rad。两个误差都按目标减实际来算。
    lateral_error = 0.0 - actual_y  # 在直线左侧时为负，横向项给出向右转的指令。
    heading_error = remainder(0.0 - actual_yaw, tau)  # 取最短的航向角差，范围约为 ±π。
    # 车头向右偏时，航向误差为正，产生向左转回的修正。
    rate = Line_Lateral_Kp * lateral_error + Line_Heading_Kp * heading_error
    return float(np.clip(rate, -Line_Max_Yaw_Rate_Rad_S, Line_Max_Yaw_Rate_Rad_S))


def run_line_trial(
    plant, gain, translation, start_step, initial_pitch, fall_pitch, show_frame=None, *,
    initial_y_m=Line_Initial_Lateral_Offset_M, before_step=None, common_torque_policy=None,
):
    def path_feedback(step, position_xy, yaw_angle):
        # 等开始前进再打开路径外环。位置在世界坐标系里，position_xy[1] 是 y 坐标。
        if step < start_step:
            return 0.0
        return correction_rate(position_xy[1], yaw_angle)

    # 直线的预设转速为零，行驶中的目标转速来自每拍更新的 path_feedback。
    return run_yaw_trial(
        plant, gain, np.zeros(len(translation)), initial_pitch, fall_pitch, show_frame,
        yaw_ki=Yaw_Ki, yaw_kd=Yaw_Kd, derivative_filter_s=Derivative_Filter_S,
        translation_reference=translation, initial_y_m=initial_y_m, yaw_rate_feedback=path_feedback,
        before_step=before_step, common_torque_policy=common_torque_policy,
    )


def print_line_results(result, reason, start_step):
    times = result["times"]
    y_cm = result["positions_xy"][:, 1] * 100.0
    yaw_deg = np.rad2deg(result["yaw_angles"])
    rates = result["reference_rates"]
    print(f"直线纠偏 | {reason} | 仿真 {times[-1]:.2f} s")
    print(f"  初始 y {y_cm[0]:+.2f} cm，结束 y {y_cm[-1]:+.4f} cm，结束偏航角 {yaw_deg[-1]:+.3f}°")
    if start_step < len(rates):
        print(f"  {times[start_step]:.2f} s 开始纠偏，首拍目标偏航角速度 {rates[start_step]:+.3f} rad/s")
        # 找到首次与启动转向反号、且能把当前偏航角拉回零的指令。
        drive_rates = rates[start_step:]
        drive_headings = yaw_deg[start_step:len(rates)]
        return_steps = np.flatnonzero((rates[start_step] * drive_rates < 0.0) & (drive_headings * drive_rates < 0.0))
        if return_steps.size:
            step = start_step + return_steps[0]
            print(f"  {times[step]:.2f} s 首次发出回正指令，此时 y {y_cm[step]:+.2f} cm，"
                  f"偏航角 {yaw_deg[step]:+.2f}°")
        peak_pitch = np.rad2deg(np.max(np.abs(result["pitches"][start_step:])))
        print(f"  行驶中的最大绝对俯仰角 {peak_pitch:.2f}°，"
              f"峰值单轮扭矩 {np.max(np.abs(result['wheel_torques'])):.4f} N·m")


def circle_geometry(position_xy, signed_radius):
    # 从原点沿 +x 出发时，左转的圆心在 (0, R)，右转的圆心在 (0, -R)。
    direction = np.sign(signed_radius)
    x = position_xy[..., 0]
    center_minus_y = signed_radius - position_xy[..., 1]
    distance_to_center = np.hypot(x, center_minus_y)
    # 取圆上最近点，沿切线左侧为正，按「目标 − 实际」计算有符号横向误差。
    lateral_error = direction * (distance_to_center - abs(signed_radius))
    # 参考航向是最近点的切线方向。小黄鸭向前走时，切线方向也随之改变。
    tangent_heading = np.arctan2(direction * x, direction * center_minus_y)
    return lateral_error, tangent_heading


def angle_error(target, actual):
    # 「目标 − 实际」后取最短角差，避免跨过 ±180° 时误认为差了一整圈。
    difference = target - actual
    return np.arctan2(np.sin(difference), np.cos(difference))


def run_arc_trial(
    plant, gain, reference, signed_radius, initial_pitch, fall_pitch, show_frame=None, *, common_torque_policy=None,
):
    translation, nominal_rates, _, start_step, stop_step = reference
    end_heading = float(np.sum(nominal_rates) * plant.dt)
    base = plant.data.joint("base_joint")

    def path_feedback(step, position_xy, actual_yaw):
        if step < start_step:
            return 0.0
        if step < stop_step:
            lateral_error, tangent_heading = circle_geometry(position_xy, signed_radius)
            heading_error = angle_error(tangent_heading, actual_yaw)
            # 世界坐标的速度投影到车头方向，得到实测前进速度。走得慢时也相应转得慢。
            forward_speed = cos(actual_yaw) * base.qvel[0] + sin(actual_yaw) * base.qvel[1]
            feedforward = forward_speed / signed_radius
            target_rate = feedforward + Arc_Lateral_Kp * lateral_error + Arc_Heading_Kp * heading_error
        else:
            # 进入停车阶段后，目标位置固定在圆弧终点，转向外环让车头对准终点的切线。
            target_rate = Arc_Heading_Kp * angle_error(end_heading, actual_yaw)
        return float(np.clip(target_rate, -Arc_Max_Yaw_Rate_Rad_S, Arc_Max_Yaw_Rate_Rad_S))

    # 预设转速置零，由 path_feedback 提供完整目标。PID 跟踪转速，共同扭矩控制平衡与前进。
    return run_yaw_trial(
        plant, gain, np.zeros_like(nominal_rates), initial_pitch, fall_pitch, show_frame,
        yaw_ki=Yaw_Ki, yaw_kd=Yaw_Kd, derivative_filter_s=Derivative_Filter_S,
        translation_reference=translation, initial_y_m=Arc_Initial_Lateral_Offset_M, yaw_rate_feedback=path_feedback,
        common_torque_policy=common_torque_policy,
    )


def print_arc_results(result, reason, reference, signed_radius):
    _, _, positions, start_step, stop_step = reference
    times = result["times"]
    xy = result["positions_xy"]
    lateral_errors, _ = circle_geometry(xy, signed_radius)
    print(f"圆弧路径跟踪 | {reason} | 仿真 {times[-1]:.2f} s | 半径 {abs(signed_radius):.2f} m")
    drive_errors = lateral_errors[start_step:stop_step + 1]
    if drive_errors.size:
        rms_cm = 100 * np.sqrt(np.mean(drive_errors ** 2))
        peak_cm = 100 * np.max(np.abs(drive_errors))
        peak_pitch = np.rad2deg(np.max(np.abs(result["pitches"][start_step:])))
        print(f"  圆弧行驶段横向均方根误差 {rms_cm:.3f} cm，峰值横向偏差 {peak_cm:.3f} cm")
        print(f"  发出行驶指令后最大绝对俯仰角 {peak_pitch:.2f}°，"
              f"峰值单轮扭矩 {np.max(np.abs(result['wheel_torques'])):.4f} N·m")
    print(f"  起点横向误差 {100 * lateral_errors[0]:+.3f} cm，"
          f"结束横向误差 {100 * lateral_errors[-1]:+.3f} cm")
    print(f"  结束车位距参考终点 {100 * np.linalg.norm(xy[-1] - positions[stop_step]):.3f} cm，"
          f"结束车头朝向 {np.rad2deg(result['yaw_angles'][-1]):+.2f}°")


def plot_tracking(trials, positions, speeds, start_step, dt, *, arc_radius=None, nominal_rates=None, stop_step=None):
    plt = configure_plot()
    fig, axes = plt.subplot_mosaic(
        [["path", "lateral"], ["heading", "rate"], ["speed", "pitch"], ["torque", "torque"]],
        figsize=(10.5, 10.5), layout="constrained",
    )
    arc = arc_radius is not None
    comparison = len(trials) > 1
    name = "圆弧路径跟踪" if arc else "直线纠偏"
    fig.suptitle(f"双轮小黄鸭{name}" + (" · LQR 与 MPC" if comparison else ""))
    period = (start_step * dt, stop_step * dt) if arc else None
    time_axes = [axis for key, axis in axes.items() if key != "path"]
    style_time_axes(time_axes, period)
    for axis in time_axes:
        if not arc:
            axis.axvline(start_step * dt, color="#64748B", ls=":", lw=1)
        axis.set(xlabel="时间（秒）", xlim=(0.0, len(speeds) * dt))

    target_style = dict(color="#765400", ls="--", lw=1.5)
    axes["path"].plot(*(positions * 100).T, label="参考路线", **target_style)
    axes["path"].grid(alpha=0.22)
    if arc:
        axes["path"].set_aspect("equal", adjustable="datalim")
    edges = np.arange(len(speeds) + 1) * dt
    axes["speed"].stairs(speeds, edges, baseline=None, label="目标", **target_style)
    for label, color, style, result in trials:
        times = result["times"]
        xy = result["positions_xy"]
        yaw_angles = result["yaw_angles"]
        if arc:
            lateral_errors, headings = circle_geometry(xy, arc_radius)
            headings[:start_step] = 0.0
            headings[stop_step:] = np.sum(nominal_rates) * dt
        else:
            lateral_errors, headings = -xy[:, 1], 0.0
        line = dict(color=color, ls=style, lw=1.8)
        axes["path"].plot(*(xy * 100).T, label=label, **line)
        axes["lateral"].plot(times, 100 * lateral_errors, **line)
        axes["heading"].plot(times, np.rad2deg(angle_error(headings, yaw_angles)), **line)
        axes["rate"].stairs(
            result["reference_rates"], times, baseline=None, color=color, ls=":", lw=1.25,
            label=f"{label} 目标",
        )
        axes["rate"].plot(times, result["yaw_rates"], label=f"{label} 实测", **line)
        axes["speed"].plot(times, result["forward_speeds"], label=label, **line)
        axes["pitch"].plot(times, np.rad2deg(result["pitches"]), **line)
        axes["torque"].stairs(result["wheel_torques"].mean(axis=1), times, baseline=None, **line)
    axes["path"].set(xlabel="世界 x（cm）", ylabel="世界 y（cm）")
    for key, label in (
        ("lateral", "横向误差（cm）"), ("heading", "航向误差（°）"), ("rate", "偏航角速度（rad/s）"),
        ("speed", "前进速度（m/s）"), ("pitch", "俯仰角（°）"), ("torque", "共同扭矩（N·m）"),
    ):
        axes[key].set_ylabel(label)
    for key in ("path", "rate", "speed"):
        axes[key].legend(frameon=False)
    filename = "03_arc_lqr_vs_mpc.png" if comparison else ("03_arc_tracking.png" if arc else "03_line_tracking.png")
    output = curve_output_path(filename)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def line_experiment(plant, gain, window, show_frame):
    reference, start_step, _ = make_speed_reference(Duration_S, plant.dt, stop=False)
    xy = np.column_stack((reference[:, 0], np.zeros(len(reference))))
    translation = np.column_stack((xy[:-1], reference[:-1, 1]))
    if window is not None and window.is_running():
        setup_line_viewer(window, reference[-1, 0])
    result, reason = run_line_trial(plant, gain, translation, start_step, Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame)
    print_line_results(result, reason, start_step)
    plot_tracking([("实测", "#007872", "-", result)], xy, translation[:, 2], start_step, plant.dt)


def arc_experiment(plant, a, b, terminal_weight, gain, window, show_frame, compare):
    if Target_Speed_M_S <= 0.0 or Target_Arc_Yaw_Rate_Rad_S == 0.0:
        raise ValueError("圆弧实验需要正的目标前进速度和非零目标偏航角速度。")
    signed_radius = Target_Speed_M_S / Target_Arc_Yaw_Rate_Rad_S
    reference = make_arc_reference(Duration_S, plant.dt)
    translation, nominal_rates, positions, start_step, stop_step = reference
    if window is not None and window.is_running():
        setup_arc_viewer(window, positions[start_step:stop_step + 1])
    trials = []
    options = (("LQR", "#007872", "-", None),)
    if compare:
        limit = min(plant.max_torque, -plant.min_torque)
        mpc = SpeedMpcPolicy(plant, a, b, gain, terminal_weight, limit)

        def common_torque(state, target_speed):
            # 参考点在局部纵向坐标系的原点，让 MPC 从这个参考继续外推位置。
            return mpc(state, (0.0, target_speed, 0.0, 0.0))

        options += (("MPC", "#73459E", "-.", common_torque),)
        print(f"MPC 预测时域 {Prediction_Horizon} 步，共 {Prediction_Horizon * plant.dt:.2f} s｜单轮限幅 ±{limit:.3f} N·m")
    for label, color, style, policy in options:
        if window is not None and window.is_running():
            window.set_texts((None, None, "Control", f"{label} + Yaw PID"))
        result, reason = run_arc_trial(
            plant, gain, reference, signed_radius, Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame,
            common_torque_policy=policy,
        )
        trials.append((label, color, style, result))
        if compare:
            print(f"{label} + 转向 PID")
        print_arc_results(result, reason, reference, signed_radius)
    if compare:
        lqr_xy = trials[0][3]["positions_xy"]
        mpc_xy = trials[1][3]["positions_xy"]
        samples = min(len(lqr_xy), len(mpc_xy))
        separation_mm = 1000.0 * np.max(np.linalg.norm(lqr_xy[:samples] - mpc_xy[:samples], axis=1))
        print(f"同一时刻两组车位最大相距 {separation_mm:.3f} mm")
        print(f"MPC 有 {mpc.constrained_solves}/{len(mpc_xy) - 1} 拍的预测序列需要约束优化。")
    plot_tracking(
        trials, positions, translation[:, 2], start_step, plant.dt,
        arc_radius=signed_radius, nominal_rates=nominal_rates, stop_step=stop_step,
    )


class SCurveRoute:
    """三个相切的 S 形，每个 S 形由等半径的左、右圆弧组成。"""

    def __init__(self, radii, sweep_rad):
        radii = np.asarray(radii, dtype=float)
        if radii.shape != (3,) or not np.all(np.isfinite(radii) & (radii > 0.0)):
            raise ValueError("S 形路线需要三个正半径。")
        if not 0.0 < sweep_rad < np.pi / 2.0:
            raise ValueError("每个圆弧的转角必须在 0° 与 90° 之间。")
        # 相邻 S 形镜像排列。每段 S 的出发和结束航向均为 0。
        directions = np.array([1, -1, -1, 1, 1, -1])
        self.curvatures = directions / np.repeat(radii, 2)
        self.lengths = np.repeat(radii, 2) * sweep_rad
        self.boundaries = np.r_[0.0, np.cumsum(self.lengths)]
        self.start_headings = np.r_[0.0, np.cumsum(self.curvatures * self.lengths)[:-1]]
        end_headings = self.start_headings + self.curvatures * self.lengths
        # 对圆弧切线方向积分，得到每小段的位移。上一段的终点就是下一段的起点。
        shifts = np.column_stack((np.sin(end_headings) - np.sin(self.start_headings),
                                  np.cos(self.start_headings) - np.cos(end_headings))) / self.curvatures[:, None]
        self.ends = np.cumsum(shifts, axis=0)
        self.starts = np.vstack((np.zeros(2), self.ends[:-1]))
        self.start_tangents = np.column_stack((np.cos(self.start_headings), np.sin(self.start_headings)))
        self.start_normals = np.column_stack((-self.start_tangents[:, 1], self.start_tangents[:, 0]))

    def segment_point(self, index, distance):
        # 已知某小段的起点，按弧长计算该段上的坐标和切线朝向。
        start_heading = self.start_headings[index]
        curvature = self.curvatures[index]
        heading = start_heading + curvature * distance
        shift = np.stack((np.sin(heading) - np.sin(start_heading),
                          np.cos(start_heading) - np.cos(heading)), axis=-1) / np.asarray(curvature)[..., None]
        return self.starts[index] + shift, heading

    def point(self, progress):
        progress = np.clip(progress, 0.0, self.boundaries[-1])
        index = np.clip(np.searchsorted(self.boundaries, progress, side="right") - 1, 0, len(self.lengths) - 1)
        return self.segment_point(index, progress - self.boundaries[index])

    def project(self, position):
        # 把实测位置转换到每段圆弧的起点坐标系，求最近点并限制在该段范围内。
        offset = position - self.starts
        x = np.sum(offset * self.start_tangents, axis=1)
        y = np.sum(offset * self.start_normals, axis=1)
        turn = np.sign(self.curvatures)
        sweep = np.arctan2(turn * x, turn * (1.0 / self.curvatures - y))
        arc_distances = sweep / self.curvatures
        # 若圆上的投影落在指定圆弧外，取该弧较近的端点。
        start_distance_sq = np.sum(offset ** 2, axis=1)
        end_distance_sq = np.sum((position - self.ends) ** 2, axis=1)
        nearest_end = np.where(start_distance_sq <= end_distance_sq, 0.0, self.lengths)
        on_segment = (arc_distances >= 0.0) & (arc_distances <= self.lengths)
        distances = np.where(on_segment, arc_distances, nearest_end)
        points, headings = self.segment_point(np.arange(len(self.lengths)), distances)
        index = np.argmin(np.sum((points - position) ** 2, axis=1))
        heading = headings[index]
        normal = np.array([-np.sin(heading), np.cos(heading)])
        # 横向误差遵循参考减实际，再取沿切线左侧的分量。
        error = normal @ (points[index] - position)
        return error, heading, self.curvatures[index]


def make_s_reference(route, dt):
    if S_Curve_Start_S < 0.0 or S_Curve_Ramp_S <= 0.0 or S_Curve_Cruise_Speed_M_S <= 0.0:
        raise ValueError("S 形起步时刻应非负，给定速度和加减速时长应为正。")
    length = route.boundaries[-1]
    first = round(S_Curve_Start_S / dt)
    # 两段半余弦过渡各等效半段匀速时间。行驶时长由路线长度、速度和过渡时间算出。
    travel_s = max(length / S_Curve_Cruise_Speed_M_S + S_Curve_Ramp_S, 2 * S_Curve_Ramp_S)
    count = ceil(travel_s / dt)
    last = first + count
    midpoint_time = (np.arange(count) + 0.5) * dt
    near_edge = np.minimum(midpoint_time, count * dt - midpoint_time)
    # 半余弦用来平滑起步和减速，缩放后速度积分等于路线总弧长。
    profile = 0.5 * (1.0 - np.cos(np.pi * np.minimum(near_edge / S_Curve_Ramp_S, 1.0)))
    speeds = np.r_[np.zeros(first), profile * (length / (dt * np.sum(profile)))]
    progress = np.r_[0.0, np.cumsum(speeds * dt)]
    progress[-1] = length
    positions, _ = route.point(progress)
    translation = np.column_stack((positions[:-1], speeds))
    return translation, positions, first, last


def run_s_curve_trial(plant, gain, route, translation, start_step, stop_step, show_frame=None):
    base = plant.data.joint("base_joint")
    end_position, end_heading = route.point(route.boundaries[-1])
    within_goal_since = None

    def path_feedback(step, position_xy, actual_yaw):
        if step < start_step:
            return 0.0
        if step < stop_step:
            lateral_error, tangent, curvature = route.project(position_xy)
            forward_speed = cos(actual_yaw) * base.qvel[0] + sin(actual_yaw) * base.qvel[1]
            # 与单圆弧一样，基础转速等于实测前进速度乘曲率，再加路径纠偏。
            target_rate = (forward_speed * curvature + Arc_Lateral_Kp * lateral_error
                           + Arc_Heading_Kp * angle_error(tangent, actual_yaw))
        else:
            target_rate = Arc_Heading_Kp * angle_error(end_heading, actual_yaw)
        return float(np.clip(target_rate, -S_Curve_Max_Yaw_Rate_Rad_S, S_Curve_Max_Yaw_Rate_Rad_S))

    def stop_at_goal(record):
        nonlocal within_goal_since
        times = record["times"]
        if len(times) - 1 < stop_step:
            return False
        previous, current = record["positions_xy"][-2:]
        speed = np.linalg.norm(current - previous) / (times[-1] - times[-2])
        if np.linalg.norm(end_position - current) > S_Curve_Position_Tolerance_M or speed > S_Curve_Stop_Speed_M_S:
            within_goal_since = None
            return False
        if within_goal_since is None:
            within_goal_since = times[-1]
        return times[-1] - within_goal_since >= S_Curve_Stop_Hold_S - 1e-9

    # 行驶参考结束后持续给终点位置和零速度，直到实测状态达到停车判据。
    targets = chain(translation, repeat((*end_position, 0.0)))
    return run_yaw_trial(
        plant, gain, repeat(0.0), Initial_Pitch_Deg, Fall_Pitch_Deg, show_frame,
        yaw_ki=Yaw_Ki, yaw_kd=Yaw_Kd, derivative_filter_s=Derivative_Filter_S,
        translation_reference=targets, initial_y_m=0.0, yaw_rate_feedback=path_feedback, stop_at_goal=stop_at_goal,
    )


def s_curve_metrics(route, result, reason, start_step):
    times = result["times"]
    positions = result["positions_xy"]
    checkpoints = route.ends[1::2]  # 每两个圆弧结束一次，即三段 S 形各自的终点。
    distances = np.linalg.norm(positions[:, None, :] - checkpoints, axis=2)
    passed = []
    next_sample = start_step
    for stage in range(len(checkpoints)):
        hits = np.flatnonzero(distances[next_sample:, stage] <= S_Curve_Position_Tolerance_M)
        passed.append(int(next_sample + hits[0]) if hits.size else None)
        next_sample = passed[-1] + 1 if hits.size else len(times)

    # 用与停车判据一致的方法估算实测平面速度。
    planar_speed = np.r_[0.0, np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)]
    goal_distances = distances[:, -1]
    lateral = np.array([route.project(point)[0] for point in positions])
    reached = reason == "达到终点停车判据" and all(step is not None for step in passed)
    return passed, goal_distances, planar_speed, lateral, reached


def plot_s_curve(route, result, translation, start_step, stop_step, dt, goal_distances, lateral):
    plt = configure_plot()
    fig, axes = plt.subplot_mosaic(
        [["path", "path"], ["lateral", "goal"], ["speed", "pitch"]],
        figsize=(10, 7.8), layout="constrained",
    )
    fig.suptitle("双轮小黄鸭 · 三段 S 形路径跟踪")
    for stage, (radius, color) in enumerate(zip(S_Curve_Radii_M, ("#196CB1", "#B35D20", "#7543A3"))):
        section = np.linspace(route.boundaries[2 * stage], route.boundaries[2 * stage + 2], 100)
        points, _ = route.point(section)
        axes["path"].plot(*(100 * points).T, ls="--", lw=1.8, color=color,
                          label=f"第 {stage + 1} 段参考（R = {radius * 100:g} cm）")
    xy = result["positions_xy"]
    axes["path"].plot(*(100 * xy).T, color="#007872", lw=1.8, label="实测轨迹")
    axes["path"].scatter(*(100 * route.ends[1::2]).T, color="#344257", s=22, zorder=3)
    axes["path"].scatter(*(100 * xy[-1]), color="#007872", marker="x", s=42, zorder=4, label="停车位置")
    axes["path"].set(xlabel="世界 x（cm）", ylabel="世界 y（cm）")
    axes["path"].set_aspect("equal", adjustable="datalim")
    axes["path"].grid(alpha=0.22)
    axes["path"].legend(ncol=3, frameon=False, loc="upper left")

    times = result["times"]
    style_time_axes((axes["lateral"], axes["goal"], axes["speed"], axes["pitch"]),
                    (start_step * dt, stop_step * dt))
    axes["lateral"].plot(times, 100 * lateral, color="#007872", lw=1.8)
    axes["lateral"].set_ylabel("横向误差（cm）")
    axes["goal"].plot(times, 100 * goal_distances, color="#007872", lw=1.8)
    axes["goal"].axhline(100 * S_Curve_Position_Tolerance_M, color="#B35D20", ls="--", label="终点容差")
    axes["goal"].set_ylabel("距最后终点（cm）")
    axes["goal"].legend(frameon=False)
    target_speeds = np.zeros(len(times) - 1)
    commanded = min(len(target_speeds), len(translation))
    target_speeds[:commanded] = translation[:commanded, 2]
    axes["speed"].stairs(target_speeds, times,
                         baseline=None, color="#765400", ls="--", lw=1.5, label="目标")
    axes["speed"].plot(times, result["forward_speeds"], color="#007872", lw=1.8, label="实测")
    axes["speed"].set_ylabel("前进速度（m/s）")
    axes["speed"].legend(frameon=False)
    axes["pitch"].plot(times, np.rad2deg(result["pitches"]), color="#7543A3", lw=1.8)
    axes["pitch"].set_ylabel("俯仰角（°）")
    for key in ("lateral", "speed", "pitch"):
        axes[key].set(xlabel="时间（秒）", xlim=(0.0, times[-1]))
    # 放大减速和停车阶段，便于读出越过终点和最后回稳的过程。
    finish_view_start = min(stop_step * dt - 1.0, times[-1] - 0.5)
    axes["goal"].set(xlabel="时间（秒）", xlim=(max(0.0, finish_view_start), times[-1]))
    visible = times >= max(0.0, finish_view_start)
    axes["goal"].set_ylim(0.0, 1.13 * max(S_Curve_Position_Tolerance_M * 100, 100 * max(goal_distances[visible])))

    output = curve_output_path("03_s_curve_tracking.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"结果图：{output}")


def s_curve_experiment(plant, gain, xml, show_viewer):
    route = SCurveRoute(S_Curve_Radii_M, np.deg2rad(S_Curve_Arc_Deg))
    translation, positions, start_step, stop_step = make_s_reference(route, plant.dt)
    floor_half_length = float(np.max(np.abs(positions[:, 0]))) + 0.5
    with playback(xml, plant.data, show_viewer, floor_half_length=floor_half_length) as (window, show_frame):
        if window is not None and window.is_running():
            setup_arc_viewer(window, positions[start_step:])
            window.set_texts((None, None, "Control", "Three connected S-curves"))
        result, reason = run_s_curve_trial(plant, gain, route, translation, start_step, stop_step, show_frame)
    passed, distances, speed, lateral, reached = s_curve_metrics(route, result, reason, start_step)
    times = result["times"]
    print(f"三段 S 形 | {reason} | 仿真 {times[-1]:.2f} s | 总弧长 {route.boundaries[-1]:.3f} m")
    for stage, (radius, sample) in enumerate(zip(S_Curve_Radii_M, passed), start=1):
        info = "未按顺序到达" if sample is None else f"{times[sample]:.2f} s 进入检查点容差"
        print(f"  第 {stage} 段（R = {100 * radius:g} cm）{info}")
    if passed[-1] is not None:
        print(f"  从发出行驶指令到首次到达最终终点附近 {times[passed[-1]] - times[start_step]:.2f} s")
    if reached:
        print(f"  从发出行驶指令到完成停车 {times[-1] - times[start_step]:.2f} s")
    print(f"  距最终终点 {100 * distances[-1]:.2f} cm，结束平面速度 {speed[-1]:.4f} m/s，"
          f"最大绝对俯仰角 {np.max(np.rad2deg(np.abs(result['pitches']))):.2f}°")
    print(f"  最大横向偏差 {100 * np.max(np.abs(lateral)):.2f} cm，"
          f"峰值单轮扭矩 {np.max(np.abs(result['wheel_torques'])):.3f} N·m")
    end_point, end_heading = route.point(route.boundaries[-1])
    end_tangent = np.array([cos(end_heading), sin(end_heading)])
    overshoot = np.max((result["positions_xy"] - end_point) @ end_tangent, initial=0.0)
    print(f"  沿终点切线的最大超程 {100 * overshoot:.2f} cm")
    print(f"  {'通过' if reached else '未通过'}｜顺序通过三段，并在终点 {100 * S_Curve_Position_Tolerance_M:g} cm 内"
          f"以不超过 {S_Curve_Stop_Speed_M_S:.2f} m/s 的速度保持 {S_Curve_Stop_Hold_S:.2f} s")
    plot_s_curve(route, result, translation, start_step, stop_step, plant.dt, distances, lateral)


def main():
    parser = argparse.ArgumentParser(description="运行直线纠偏、圆弧跟踪、圆弧对照或三段 S 形。")
    parser.add_argument("experiment", nargs="?", default="line", choices=("line", "arc", "arc-compare", "s-curve"))
    parser.add_argument("--viewer", action="store_true", help="在 MuJoCo Viewer 回放参考路线。")
    args = parser.parse_args()
    xml, plant = load_plant()
    a, b = linearize(plant)
    terminal_weight, gain = solve_riccati(a, b)
    if args.experiment == "s-curve":
        s_curve_experiment(plant, gain, xml, args.viewer)
        return
    with playback(xml, plant.data, args.viewer) as (window, show_frame):
        if args.experiment == "line":
            line_experiment(plant, gain, window, show_frame)
        else:
            arc_experiment(plant, a, b, terminal_weight, gain, window, show_frame, args.experiment == "arc-compare")


if __name__ == "__main__":
    main()
