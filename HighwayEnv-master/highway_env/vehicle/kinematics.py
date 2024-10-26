from __future__ import annotations
from typing import Union
import copy
from collections import deque
import math
import numpy as np
import ray

from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.objects import RoadObject


class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0

    def update(self, target_heading: float, current_heading: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be greater than zero")

        error = target_heading - current_heading
        self.integral += error * dt

        derivative = (error - self.last_error) / dt
        self.last_error = error

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        """重置积分和误差"""
        self.integral = 0
        self.last_error = 0


def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class DynamicReferencePath:
    def __init__(self, length=30, num_points=300, lane_width=4, safety_distance=2.5):
        self.length = length  # 参考路径的长度
        self.num_points = num_points  # 参考路径的点数
        self.lane_width = lane_width  # 车道宽度
        self.safety_distance = safety_distance  # 安全距离
        self.refer_path = np.zeros((num_points, 4))  # 初始化参考路径

    def generate_path(self, vehicles):
        """Generate a dynamic reference trajectory based on vehicle positions.

        Args:
            vehicles (list): A list of vehicle objects with a position attribute.
        """
        vx, vy = vehicles[0].position  # 假设我们用第一个车辆的位置作为基准

        # 生成基础路径并加上车辆位置偏移
        self.refer_path[:, 0] = np.linspace(vx, vx + self.length, self.num_points)
        self.refer_path[:, 1] = vy + 2 * np.sin(self.refer_path[:, 0] / 3.0) + 2.5 * np.cos(self.refer_path[:, 0] / 2.0)

        # 计算切线方向和曲率
        for i in range(self.num_points):
            dx = self.refer_path[i, 0] - self.refer_path[i-1, 0] if i > 0 else 0.01
            dy = self.refer_path[i, 1] - self.refer_path[i-1, 1] if i > 0 else 0.01
            self.refer_path[i, 2] = math.atan2(dy, dx)  # yaw
            if i > 0:
                curvature = dy / (dx ** 2 + dy ** 2) ** (3 / 2)
                self.refer_path[i, 3] = curvature  # 曲率k

        # 避障逻辑
        for vehicle in vehicles:
            vx, vy = vehicle.position  # 获取车辆的当前位置
            # 计算与参考路径的距离并调整
            for j in range(self.num_points):
                dist = np.sqrt((self.refer_path[j, 0] - vx) ** 2 + (self.refer_path[j, 1] - vy) ** 2)
                if dist < self.lane_width / 2 + self.safety_distance:  # 如果距离小于车道宽度加安全距离
                    # 调整参考路径
                    shift = self.lane_width / 2 + self.safety_distance - dist
                    # 计算路径调整量，基于距离和震荡因子
                    if dist < self.lane_width / 2:  # 距离非常近
                        adjustment_factor = 0.5  # 完全调整
                    else:  # 距离稍远
                        adjustment_factor = 0.01  # 减少调整幅度
                    # 限制最大调整幅度
                    max_shift = 1.0  # 可以根据需求调整最大限制
                    shift = min(shift * adjustment_factor, max_shift)
                    angle = self.refer_path[j, 2] + np.pi / 2  # 计算垂直于路径的方向
                    self.refer_path[j, 0] += shift * np.cos(angle)
                    self.refer_path[j, 1] += shift * np.sin(angle)

    def calc_track_error(self, x, y):
        """Calculate tracking error.

        Args:
            x (float): Current vehicle position x.
            y (float): Current vehicle position y.

        Returns:
            tuple: (error, curvature, yaw, index)
        """
        # 寻找参考轨迹最近目标点
        d_x = self.refer_path[:, 0] - x
        d_y = self.refer_path[:, 1] - y
        d = np.sqrt(d_x ** 2 + d_y ** 2)
        s = np.argmin(d)  # 最近目标点索引

        yaw = self.refer_path[s, 2]
        k = self.refer_path[s, 3]
        angle = normalize_angle(yaw - math.atan2(d_y[s], d_x[s]))
        e = d[s]  # 误差
        if angle < 0:
            e *= -1

        return e, k, yaw, s


class LQRController:
    def __init__(self, N=100, EPS=1e-4):
        """
        初始化LQR控制器
        :param N: 最大迭代次数
        :param EPS: 迭代精度
        """
        self.Q = np.eye(3) * 3
        self.R = np.eye(2) * 2.
        self.N = N
        self.EPS = EPS

    def solve_riccati(self, A, B):
        """
        解代数Riccati方程
        :param A: 状态矩阵A
        :param B: 输入矩阵B
        :return: P矩阵
        """
        P = self.Q
        for _ in range(self.N):
            P_next = self.Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.pinv(self.R + B.T @ P @ B) @ B.T @ P @ A
            if np.abs(P_next - P).max() < self.EPS:
                break
            P = P_next
        return P

    def compute_control(self, x, A, B):
        """
        计算LQR控制输入
        :param x: 状态误差
        :param A: 状态矩阵A
        :param B: 输入矩阵B
        :return: 控制输入u
        """
        P = self.solve_riccati(A, B)
        K = -np.linalg.pinv(self.R + B.T @ P @ B) @ B.T @ P @ A
        u = K @ x
        return u[0, 1]


class Vehicle(RoadObject):
    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It is state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    L = 2.0
    DEFAULT_INITIAL_SPEEDS = [20, 25]  # 论文要求的初始车速
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 30.0
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = 20.0
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""
    MAX_STEERING_CHANGE = 0.1
    MAX_ACC_CHANGE = 0.01

    def __init__(
            self,
            road: Road,
            position: Vector,
            heading: float = 0,
            speed: float = 0,
            prediction_type: str = "constant_steering",
            pid_acceleration: PIDController = None,
            is_observed: bool = False
    ):
        super().__init__(road, position, heading, speed)
        self.target_heading = None
        self.target_lane_center_y = None
        self.jerk_y = None
        self.jerk_x = None
        self.previous_acceleration_y = 0
        self.previous_acceleration_x = 0
        self.prediction_type = prediction_type
        self.action: dict[str, Union[float, np.ndarray]] = {"steering": 0.0, "acceleration": 0.0}
        self.crashed = False
        self.is_observed = is_observed
        self.is_changing_lane = None
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.acceleration1 = 0.0
        self.previous_acceleration = 0.0  # 前一时刻的加速度
        self.jerk = 0.0  # 当前加加速度
        self.pid_controller_acceleration = pid_acceleration if pid_acceleration else PIDController(3, 0.05, 0.2)
        self.dy_ref_path = DynamicReferencePath()
        self.lqr_controller = LQRController()

    @classmethod
    def create_random(
        cls,
        road: Road,
        speed: float = None,
        lane_from: str | None = None,
        lane_to: str | None = None,
        lane_id: int | None = None,
        spacing: float = 1,
        is_observed: bool = False
    ) -> Vehicle:
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :param is_observed: 是否是观测车辆
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7 * lane.speed_limit, 0.8 * lane.speed_limit
                )
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]
                )
        default_spacing = 12 + 1.0 * speed
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            if len(road.vehicles)
            else 3 * offset
        )
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed, is_observed)
        return v

    @classmethod
    def create_from(cls, vehicle: Vehicle) -> Vehicle:
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        if hasattr(vehicle, "color"):
            v.color = vehicle.color
        return v

    def act(self, action: dict | str = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def get_nearby_obstacles(self, distance_threshold: float = LENGTH) -> list[RoadObject]:
        nearby_obstacles = []  # 将当前车辆的位置 self.position 转换为一个 NumPy 数组，确保后续可以进行矢量运算。self.position 应该是当前车辆在道路上的二维坐标。
        # 观测车辆的坐标
        self_pos = np.array(self.position)  # 确保是 numpy 数组
        # 假设车辆也算作障碍物，这个循环遍历 self.road.vehicles 中的所有车辆。self.road 表示当前车辆所在的道路，
        # self.road.vehicles 是该道路上所有车辆的列表。
        for obj_id, obj in enumerate(
                self.road.vehicles):
            # 如果是非观测车辆
            if not obj.is_observed:
                # 获取非观测车辆的坐标
                obj_pos = np.array(obj.position)  # 确保是 numpy 数组
                # 计算记录
                distance = np.linalg.norm(obj_pos - self_pos)
                # 这里不能简单给10，如果是以像素为单位，直线上建议把这个距离给一个车的长度，考虑到变道后隔壁车道也有车，应该在求一个三角形斜边（有兴趣你自己加）
                # print(f"观测车辆车道{self.lane_index[2]}-要避障的车辆的车道-{obj.lane_index[2]}")
                if distance < distance_threshold * 2.5 and self.lane_index[2] == obj.lane_index[2]:
                    nearby_obstacles.append(obj)
        return nearby_obstacles

    def lqr_compute(self, dt, steering):
        robot_state = np.zeros(4)
        robot_state[0] = self.position[0]
        robot_state[1] = self.position[1]
        robot_state[2] = steering
        robot_state[3] = self.speed
        self.dy_ref_path.generate_path(self.road.vehicles)
        e, k, ref_yaw, s0 = self.dy_ref_path.calc_track_error(
            robot_state[0], robot_state[1])
        ref_delta = math.atan2(self.L*k, 1)
        A = np.matrix([
            [1.0, 0.0, -self.speed * dt * math.sin(ref_yaw)],
            [0.0, 1.0, self.speed * dt * math.cos(ref_yaw)],
            [0.0, 0.0, 1.0]])

        B = np.matrix([
            [dt * math.cos(ref_yaw), 0],
            [dt * math.sin(ref_yaw), 0],
            [dt * math.tan(ref_delta) / self.L, self.speed * dt /
             (self.L * math.cos(ref_delta) * math.cos(ref_delta))]
        ])

        x = robot_state[0:3]-self.dy_ref_path.refer_path[s0, 0:3]
        delta = self.lqr_controller.compute_control(x, A, B)
        return delta+ref_delta

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.
        """
        # 使用LQR平滑角度
        if self.is_observed:
            steering_control = self.lqr_compute(dt, self.action['steering'])
            self.action["steering"] = np.clip(steering_control,
                                              self.action["steering"] - self.MAX_STEERING_CHANGE,
                                              self.action["steering"] + self.MAX_STEERING_CHANGE)
            # PID平滑加速度
            acceleration_control = self.pid_controller_acceleration.update(self.MAX_SPEED, self.speed, dt)
            # 限制变化量
            self.action["acceleration"] = np.clip(
                acceleration_control,
                self.action["acceleration"] - self.MAX_ACC_CHANGE,
                self.action["acceleration"] + self.MAX_ACC_CHANGE
            )
            obstacles = self.get_nearby_obstacles()  # 获取障碍物
            if obstacles:
                if self.lane_index[2] == 3 and steering_control > 0:  # 避免向左转，保持直行或向右
                    steering_control = -steering_control
                elif self.lane_index[2] == 0 and steering_control < 0:  # 避免向左转，保持直行或向右
                    steering_control = -steering_control
                else:
                    steering_control = steering_control
                self.action['steering'] = steering_control
            # else:
            #     self.action["steering"] = 0
        else:
            self.action['steering'] = 0
        self.clip_actions()

        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta), np.sin(self.heading + beta)])
        self.position += v * dt

        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt
        # 处理换道逻辑
        # 更新车辆的朝向
        # 初始化换道参数
        if not hasattr(self, 'is_changing_lane'):
            self.is_changing_lane = False
            self.target_lane_center_y = None
        new_lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)

        # 检查是否需要更改车道
        if new_lane_index[2] != self.lane_index[2] and not self.is_changing_lane:
            # 设置换道目标位置
            self.target_lane_center_y = (new_lane_index[2] + 0.5) * 4 - 2  # 车道宽度为4
            self.target_heading = self.lane.heading_at(self.position[0])  # 目标车道的航向
            self.is_changing_lane = True  # 标记正在换道
            self.lane_index = new_lane_index
            self.lane = self.road.network.get_lane(self.lane_index)

        # 换道过程平滑处理
        if self.is_changing_lane:
            # 逐步调整位置和车头方向
            delta_y = (self.target_lane_center_y - self.position[1]) * 0.4  # 小步移动
            self.position[1] += delta_y
            delta_heading = (self.target_heading - self.heading) * 0.4  # 小步调整航向
            self.heading += delta_heading
            # 当接近目标位置和目标朝向时，结束换道过程
            if abs(self.position[1] - self.target_lane_center_y) < 0.1 and abs(
                    self.heading - self.target_heading) < 0.1:
                self.position[1] = self.target_lane_center_y
                self.heading = self.target_heading
                self.is_changing_lane = False  # 换道完成
        # 调用状态更新
        self.collect_jerk_message(dt)
        # 调用状态更新
        self.on_state_update()

    def collect_jerk_message(self, dt):
        # 计算当前时刻的横向和纵向加速度
        current_acceleration_x = self.action["acceleration"] * np.cos(self.heading)
        current_acceleration_y = self.action["acceleration"] * np.sin(self.heading)
        # 计算横向和纵向加加速度（jerk），jerk = 加速度的变化 / 时间差
        jerk_x = (current_acceleration_x - self.previous_acceleration_x) / dt
        jerk_y = (current_acceleration_y - self.previous_acceleration_y) / dt
        # 更新前一时刻的加速度值
        self.previous_acceleration_x = current_acceleration_x
        self.previous_acceleration_y = current_acceleration_y
        # 输出当前横向和纵向的加加速度
        self.jerk_x = jerk_x
        self.jerk_y = jerk_y

    @property
    def get_jerk_x(self) -> float:
        """返回当前的加加速度"""
        return self.jerk_x

    @property
    def get_jerk_y(self) -> float:
        """返回当前的加加速度"""
        return self.jerk_y

    @property
    def get_verb_x(self) -> float:
        """获取规划此刻的横向速度"""
        return self.velocity[0]

    @property
    def get_verb_y(self) -> float:
        """获取此刻的纵向速度"""
        return self.velocity[1]

    def clip_actions(self) -> None:
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = -1.0 * self.speed
        self.action["steering"] = float(self.action["steering"])
        self.action["acceleration"] = float(self.action["acceleration"])
        if self.speed > self.MAX_SPEED:
            self.action["acceleration"] = min(
                self.action["acceleration"], 1.0 * (self.MAX_SPEED - self.speed)
            )
        elif self.speed < self.MIN_SPEED:
            self.action["acceleration"] = max(
                self.action["acceleration"], 1.0 * (self.MIN_SPEED - self.speed)
            )

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(
                self.position, self.heading
            )
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(
            self, times: np.ndarray
    ) -> tuple[list[np.ndarray], list[float]]:
        if self.prediction_type == "zero_steering":
            action = {"acceleration": 0.0, "steering": 0.0}
        elif self.prediction_type == "constant_steering":
            action = {"acceleration": 0.0, "steering": self.action["steering"]}
        else:
            raise ValueError("Unknown prediction type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return positions, headings

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = (
                last_lane_index
                if last_lane_index[-1] is not None
                else (*last_lane_index[:-1], 0)
            )
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(
                self.destination - self.position
            )
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang])
        else:
            return np.zeros((3,))

    def to_dict(
            self, origin_vehicle: Vehicle = None, observe_intentions: bool = True
    ) -> dict:
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": self.velocity[0],
            "vy": self.velocity[1],
            "heading": self.heading,
            "cos_h": self.direction[0],
            "sin_h": self.direction[1],
            "cos_d": self.destination_direction[0],
            "sin_d": self.destination_direction[1],
            "long_off": self.lane_offset[0],
            "lat_off": self.lane_offset[1],
            "ang_off": self.lane_offset[2],
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(
            self.__class__.__name__, id(self) % 1000, self.position
        )

    def __repr__(self):
        return self.__str__()

    def predict_trajectory(
            self,
            actions: list,
            action_duration: float,
            trajectory_timestep: float,
            dt: float,
    ) -> list[Vehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # Low-level control action
            for _ in range(int(action_duration / dt)):
                t += 1
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
