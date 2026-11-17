"""小黄鸭共用的 LQR、转向 PID、扭矩分配和 MPC。"""
from itertools import repeat
from math import atan2, cos, radians, remainder, sin, tau
from time import perf_counter

import mujoco
import numpy as np

State_Weights = (20.0, 20.0, 80.0, 1.0)  # 位置、速度、俯仰角和俯仰角速度。
Torque_Weight = 3.0
Yaw_Kp = 0.20  # (N·m)/(rad/s)，根据本拍偏航角速度误差给出单侧转向修正。
Yaw_Ki = 0.80  # (N·m)/rad，乘以累计的偏航角速度误差。
Yaw_Kd = 0.008  # (N·m)/(rad/s²)，给滤波后的实测角加速度施加反向修正。
# 对相邻两拍实测角速度差分，再做一阶低通，减小 D 项的抖动。
# 新值权重 alpha = dt / (Derivative_Filter_S + dt)。滤波结果 = (1 - alpha) × 旧值 + alpha × 新值。
# dt=0.01 s 时新值占 25%。时间常数越大，输出更平滑、响应也越慢。
Derivative_Filter_S = 0.03  # s，一阶低通时间常数。
Prediction_Horizon = 50  # 预测 0.50 秒。
Qp_Tolerance = 1e-8
Qp_Max_Iterations = 500
Direction_Tolerance = 1e-10


def solve_riccati(a, b):
    # LQR 用 s.T @ Q @ s 给偏离平衡点的状态计代价，用 R 乘扭矩平方给控制用力计代价。
    q = np.diag(State_Weights)
    r = np.array([[Torque_Weight]])
    # P 表示未来最优代价的权重。先从剩余 0 步的情况开始，未来代价为零。
    p = np.zeros_like(q)
    for _ in range(8000):
        # 根据这一轮的 P 解方程 (R + B.T @ P @ B) @ K = B.T @ P @ A。
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        # 把最优反馈代回本步与后续的代价，得到往前多考虑一步的 P。
        next_p = q + a.T @ p @ a - a.T @ p @ b @ gain
        # 理论上 P 是对称矩阵，取平均纠正浮点计算产生的微小非对称。
        next_p = (next_p + next_p.T) / 2
        if np.max(np.abs(next_p - p)) < 1e-9:
            # 用收敛后的 P 最后求一次 K。仿真循环每步复用同一组增益。
            gain = np.linalg.solve(r + b.T @ next_p @ b, b.T @ next_p @ a)
            return next_p, gain
        p = next_p
    raise RuntimeError("黎卡提递推未收敛，请检查模型和权重。")


def solve_lqr(a, b):
    _, gain = solve_riccati(a, b)
    return gain


def read_yaw(data):
    # 四元数换成车头在水平面上的偏航角，正角从俯视方向看是左转。
    w, x, y, z = data.joint("base_joint").qpos[3:7]
    return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def mix_torques(common_torque, turn_torque, min_torque, max_torque):
    # 先保留 LQR 的共同扭矩，再算两轮一减一加最多能用多少扭矩。
    # 例如共同扭矩 +0.48 N·m、上限 +0.50 N·m 时，转向修正最多只能用 0.02 N·m。
    common = float(np.clip(common_torque, min_torque, max_torque))
    margin = min(common - min_torque, max_torque - common)
    turn = float(np.clip(turn_torque, -margin, margin))
    return common - turn, common + turn


class YawPid:
    """保存转向 PID 的上一拍读数，每次给出下一拍的左右轮扭矩。"""

    def __init__(self, dt, ki, kd, filter_s):
        if filter_s < 0.0:
            raise ValueError("微分滤波时间常数必须大于或等于零。")
        self.dt = dt
        self.ki = ki
        self.kd = kd
        self.filter_weight = dt / (filter_s + dt)
        self.integral_torque = 0.0
        self.derivative_torque = 0.0
        self.previous_rate = 0.0
        self.filtered_acceleration = 0.0

    def step(self, reference_rate, actual_rate, common_torque, min_torque, max_torque):
        # 正的偏航速度误差增加右轮扭矩，同时减小左轮扭矩。
        error = reference_rate - actual_rate
        proportional_torque = Yaw_Kp * error
        candidate_integral = self.integral_torque + self.ki * error * self.dt
        # D 项对相邻两拍的实测转速做差分，再用一阶低通减小抖动。
        acceleration = (actual_rate - self.previous_rate) / self.dt
        self.filtered_acceleration += self.filter_weight * (acceleration - self.filtered_acceleration)
        self.derivative_torque = -self.kd * self.filtered_acceleration
        self.previous_rate = actual_rate

        proportional_derivative_torque = proportional_torque + self.derivative_torque
        turn_torque = proportional_derivative_torque + candidate_integral
        left, right = mix_torques(common_torque, turn_torque, min_torque, max_torque)
        # 限幅时，若新积分会让扭矩进一步超限，就沿用上一拍的积分。
        clipped_amount = turn_torque - (right - left) / 2
        if abs(clipped_amount) > 1e-12 and clipped_amount * error > 0:
            return mix_torques(
                common_torque, proportional_derivative_torque + self.integral_torque, min_torque, max_torque,
            )
        self.integral_torque = candidate_integral
        return left, right


def run_yaw_trial(
    plant, gain, reference_rates, initial_pitch, fall_pitch, show_frame=None, *,
    yaw_ki=0.0, yaw_kd=0.0, derivative_filter_s=0.0, translation_reference=None,
    initial_y_m=None, yaw_rate_feedback=None, common_torque_policy=None, stop_at_goal=None, before_step=None,
):
    yaw_pid = YawPid(plant.dt, yaw_ki, yaw_kd, derivative_filter_s)
    state = plant.reset((0.0, 0.0, radians(initial_pitch), 0.0))
    base = plant.data.joint("base_joint")
    if initial_y_m is not None:
        # 直线纠偏实验在复位后横向移动起点。位置改变后重新计算接触和其他派生量。
        base.qpos[1] = initial_y_m
        mujoco.mj_forward(plant.model, plant.data)
    yaw = read_yaw(plant.data)
    yaw_rate = 0.0  # 实验从静止开始，读到第一对角度采样后再测转速。
    record = {
        "times": [0.0], "yaw_rates": [yaw_rate], "yaw_angles": [yaw], "pitches": [state[2]],
        "positions_xy": [base.qpos[:2].copy()], "forward_speeds": [float(base.qvel[0])],
        "reference_rates": [], "wheel_torques": [], "integral_torques": [], "derivative_torques": [],
    }
    reason = "达到实验时长"
    if show_frame is not None:
        show_frame(0.7)
    start = perf_counter()

    # 原地转向时，每拍的目标位置和前进速度都是零。边走边转时，两路目标逐拍对齐。
    if translation_reference is None:
        commands = zip(reference_rates, repeat((0.0, 0.0, 0.0)))
    else:
        commands = zip(reference_rates, translation_reference, strict=True)
    for step, (reference_rate, (target_x, target_y, target_speed)) in enumerate(commands):
        if yaw_rate_feedback is not None:
            # 路径外环用本拍车位和航向计算纠偏转速，与预设转速相加后交给转向 PID。
            reference_rate += yaw_rate_feedback(step, base.qpos[:2], yaw)
        # yaw 是当前车头相对世界 +x 的偏航角（rad）。cos 和 sin 给出当前车头的单位方向。
        forward = np.array([cos(yaw), sin(yaw)])
        # 先求参考位置减实际位置，再投影到当前车头方向，得到纵向位置误差（m）。
        position_error = forward @ ((target_x, target_y) - base.qpos[:2])
        forward_speed = forward @ base.qvel[:2]  # 实测前进速度（m/s）。
        # state[2] 和 state[3] 来自 BalancePlant.state()，分别是俯仰角（rad）和俯仰角速度（rad/s）。
        if common_torque_policy is None:
            # 四项误差都是参考减实际，参考俯仰角和俯仰角速度均为零。
            tracking_error = np.array([position_error, target_speed - forward_speed, -state[2], -state[3]])
            balance_torque = (gain @ tracking_error).item()
        else:
            # MPC 将本拍参考点作为局部坐标原点，输入在该坐标系下的实测状态。
            state[0] = -position_error  # 以参考点为原点的实测纵向坐标为位置误差的相反数。
            state[1] = forward_speed
            balance_torque = common_torque_policy(state, target_speed)

        left, right = yaw_pid.step(reference_rate, yaw_rate, balance_torque, plant.min_torque, plant.max_torque)
        if before_step is not None:
            # 在推进本拍物理仿真前，更新外力等实验输入。
            before_step(step)
        state = plant.step((left, right))

        # 用相邻两帧的角度差除以采样周期。remainder 让过 ±180° 时仍取到正确的转角。
        new_yaw = read_yaw(plant.data)
        yaw_increment = remainder(new_yaw - yaw, tau)
        yaw_rate = yaw_increment / plant.dt
        yaw = new_yaw
        record["times"].append(float(plant.data.time))
        record["yaw_rates"].append(yaw_rate)
        record["yaw_angles"].append(record["yaw_angles"][-1] + yaw_increment)
        record["pitches"].append(state[2])
        record["positions_xy"].append(base.qpos[:2].copy())
        record["forward_speeds"].append(float(cos(yaw) * base.qvel[0] + sin(yaw) * base.qvel[1]))
        record["reference_rates"].append(reference_rate)
        record["wheel_torques"].append(plant.data.actuator_force.copy())
        record["integral_torques"].append(yaw_pid.integral_torque)
        record["derivative_torques"].append(yaw_pid.derivative_torque)

        if show_frame is not None:
            show_frame(max(0.0, start + plant.data.time - perf_counter()))
        if abs(state[2]) >= radians(fall_pitch):
            reason = "达到俯仰角边界"
            break
        if stop_at_goal is not None and stop_at_goal(record):
            reason = "达到终点停车判据"
            break
    if show_frame is not None:
        show_frame(0.7)
    return {name: np.asarray(values) for name, values in record.items()}, reason


# MPC 沿用倒立摆项目的预测矩阵构造和对偶活跃集方法。
def prediction_matrices(a, b, horizon):
    # 将未来的 x₁ 到 x_N 堆叠为一列，使 X = Sx @ x₀ + Sv @ V。
    state_count = a.shape[0]
    sx = np.empty((horizon * state_count, state_count))
    sv = np.empty((horizon * state_count, horizon))
    state_block = np.eye(state_count)
    input_block = np.zeros((state_count, horizon))
    for step in range(horizon):
        state_block = a @ state_block
        input_block = a @ input_block
        input_block[:, step] = b[:, 0]
        rows = slice(step * state_count, (step + 1) * state_count)
        sx[rows] = state_block
        sv[rows] = input_block
    return sx, sv


class BoxConstrainedQp:
    """求解 0.5 Vᵀ H V + fᵀ V，约束为 lower ≤ C V ≤ upper。"""

    def __init__(self, hessian, constraints):
        self.hessian = hessian
        self.inequalities = np.vstack((constraints, -constraints))
        self.factor = np.linalg.cholesky(hessian)
        # 从无约束解 V₀ 出发。用 z = Lᵀ(V − V₀) 把目标变为 0.5 ||z||²。
        rows = np.linalg.solve(self.factor, self.inequalities.T).T
        self.row_norms = np.linalg.norm(rows, axis=1)
        self.rows = rows / self.row_norms[:, None]

    def solve(self, linear_cost, lower, upper):
        unconstrained = np.linalg.solve(self.hessian, -linear_cost)
        bounds = np.concatenate((upper, -lower))
        remaining = bounds - self.inequalities @ unconstrained
        if np.all(remaining >= 0.0):
            return unconstrained, 0

        # 暂时把触及边界的约束当成等式，沿着不破坏这些等式的方向消除新的超限。
        limits = remaining / self.row_norms
        adjustment = np.zeros_like(unconstrained)
        active = []
        multipliers = np.empty(0)
        incoming = None
        for iteration in range(Qp_Max_Iterations + 1):
            if incoming is None:
                violation = self.rows @ adjustment - limits
                incoming = int(np.argmax(violation))
                if violation[incoming] <= Qp_Tolerance:
                    solution = unconstrained + np.linalg.solve(self.factor.T, adjustment)
                    gradient = self.hessian @ solution + linear_cost
                    gradient += self.inequalities[active].T @ (multipliers / self.row_norms[active])
                    complementarity = violation[active] * multipliers
                    if (
                        np.max(self.inequalities @ solution - bounds) <= Qp_Tolerance
                        and np.max(np.abs(gradient)) <= Qp_Tolerance
                        and np.max(np.abs(complementarity), initial=0.0) <= Qp_Tolerance
                    ):
                        return solution, iteration
                    raise RuntimeError("QP 的约束或最优性检查未达到数值容差。")
                incoming_multiplier = 0.0

            if iteration == Qp_Max_Iterations:
                break

            row = self.rows[incoming]
            direction = -row.copy()
            dual_direction = np.empty(0)
            dual_step = np.inf
            if active:
                basis, triangular = np.linalg.qr(self.rows[active].T)
                projection = basis.T @ row
                direction += basis @ projection
                dual_direction = -np.linalg.solve(triangular, projection)
                # 某条旧约束的乘子先降到零时，移除该约束，再继续处理新约束。
                steps = np.full(len(active), np.inf)
                np.divide(multipliers, -dual_direction, out=steps, where=dual_direction < -Direction_Tolerance)
                blocking = int(np.argmin(steps))
                dual_step = steps[blocking]

            primal_step = np.inf
            slope = direction @ direction
            if slope > Direction_Tolerance**2:
                primal_step = max(0.0, row @ adjustment - limits[incoming]) / slope
            step = min(primal_step, dual_step)
            if not np.isfinite(step):
                raise RuntimeError("QP 未找到可以继续消除超限的搜索方向。")

            adjustment += step * direction
            multipliers = np.maximum(0.0, multipliers + step * dual_direction)
            incoming_multiplier += step
            if primal_step <= dual_step:
                active.append(incoming)
                multipliers = np.append(multipliers, incoming_multiplier)
                incoming = None
            else:
                del active[blocking]
                multipliers = np.delete(multipliers, blocking)

        raise RuntimeError("QP 在迭代上限内未收敛。")


class TorqueMpc:
    """预测车身的四个纵向状态，求单轮共同扭矩序列。"""

    def __init__(self, a, b, gain, q, torque_weight, terminal_weight, horizon, torque_limit):
        state_count = a.shape[0]
        # 直接展开 A 的幂会放大倒立摆的不稳定模态。令 u = -Kx + v，
        # 用闭环矩阵 A - BK 做预测，优化 v。代价和限幅仍使用还原后的实际扭矩 u。
        self.sx, sv = prediction_matrices(a - b @ gain, b, horizon)
        before_state = np.vstack((np.eye(state_count), self.sx[:-state_count]))
        before_increment = np.vstack((np.zeros((state_count, horizon)), sv[:-state_count]))
        gains = np.kron(np.eye(horizon), gain)
        # 力矩 u₀ 到 u_{N-1} 由当前状态及各步修正量共同决定。
        self.state_to_torque = -gains @ before_state
        self.increment_to_torque = np.eye(horizon) - gains @ before_increment

        weights = np.kron(np.eye(horizon), q)
        weights[-state_count:, -state_count:] = terminal_weight  # 用黎卡提方程的 P 为末端状态计分。
        self.weighted_sv = sv.T @ weights
        self.weighted_torques = torque_weight * self.increment_to_torque.T
        hessian = 2.0 * (self.weighted_sv @ sv + self.weighted_torques @ self.increment_to_torque)
        hessian = (hessian + hessian.T) / 2.0
        self.qp = BoxConstrainedQp(hessian, self.increment_to_torque)
        self.upper = np.full(horizon, torque_limit)

    def plan(self, state, future_reference):
        # 每一拍只更新初始状态和参考。H 与预测模型矩阵都可复用。
        free_error = future_reference.ravel() - self.sx @ state
        free_torques = self.state_to_torque @ state
        # 预测误差是参考减预测状态，所以加入修正量 V 后，误差为 free_error - Sv @ V。
        linear_cost = 2.0 * (-(self.weighted_sv @ free_error) + self.weighted_torques @ free_torques)
        # 把已知的 free_torques 移到边界右边，得到修正量必须满足的约束。
        increments, iterations = self.qp.solve(linear_cost, -self.upper - free_torques, self.upper - free_torques)
        torques = free_torques + self.increment_to_torque @ increments
        return torques, iterations


def make_future_reference(target, lookahead_times):
    # 以本拍目标速度等速外推参考位置，目标速度和俯仰状态在这个预测窗口内保持给定值。
    future = np.repeat(np.asarray(target)[None, :], len(lookahead_times), axis=0)
    future[:, 0] += lookahead_times * target[1]
    return future


class SpeedMpcPolicy:
    """用当拍指令预测纵向运动，执行优化序列的第一拍。"""

    def __init__(self, plant, a, b, gain, terminal_weight, torque_limit):
        self.plant = plant
        self.controller = TorqueMpc(
            a, b, gain, np.diag(State_Weights), Torque_Weight, terminal_weight, Prediction_Horizon, torque_limit,
        )
        self.lookahead_times = plant.dt * np.arange(1, Prediction_Horizon + 1)
        self.constrained_solves = 0

    def __call__(self, state, target):
        future = make_future_reference(target, self.lookahead_times)
        try:
            torques, iterations = self.controller.plan(state, future)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            raise RuntimeError(f"第 {self.plant.data.time:.2f} 秒的 MPC 求解失败。{error}") from error
        self.constrained_solves += int(iterations > 0)
        return torques[0]
