from __future__ import annotations

import copy
from collections import deque

import numpy as np
# import ray
from ray import logger
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.objects import RoadObject


class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float, integral_limit: float = None) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0
        self.integral_limit = integral_limit

    def update(self, target_heading: float, current_heading: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be greater than zero")

        error = target_heading - current_heading
        self.integral += error * dt

        if self.integral_limit is not None:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        derivative = (error - self.last_error) / dt
        self.last_error = error

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


class Vehicle(RoadObject):
    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_INITIAL_SPEEDS = [20, 25]  # 论文要求的初始车速
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 30.0
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = 20.0
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        predition_type: str = "constant_steering",
        pid_controller: PIDController = None,
        is_observed: bool = False
    ):
        super().__init__(road, position, heading, speed)
        self.jerk_y = None
        self.jerk_x = None
        self.previous_acceleration_y = None
        self.previous_acceleration_x = None
        self.prediction_type = predition_type
        self.action = {"steering": 0, "acceleration": 0}
        self.crashed = False
        self.is_observed = is_observed
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.acceleration1 = 0.0
        self.previous_acceleration = 0.0  # 前一时刻的加速度
        self.jerk = 0.0  # 当前加加速度
        self.pid_controller = pid_controller if pid_controller else PIDController(3, 0.05, 0.2)  # 默认 PID 参数

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
        nearby_obstacles = []#将当前车辆的位置 self.position 转换为一个 NumPy 数组，确保后续可以进行矢量运算。self.position 应该是当前车辆在道路上的二维坐标。
        # 观测车辆的坐标
        self_pos = np.array(self.position)  # 确保是 numpy 数组
        for obj_id, obj in enumerate(self.road.vehicles):  # 假设车辆也算作障碍物，这个循环遍历 self.road.vehicles 中的所有车辆。self.road 表示当前车辆所在的道路，self.road.vehicles 是该道路上所有车辆的列表。
            # 如果是非观测车辆
            if not obj.is_observed:
                # 获取非观测车辆的坐标
                obj_pos = np.array(obj.position)  # 确保是 numpy 数组
                # 计算记录
                distance = np.linalg.norm(obj_pos - self_pos)
                # 这里不能简单给10，如果是以像素为单位，直线上建议把这个距离给一个车的长度，考虑到变道后隔壁车道也有车，应该在求一个三角形斜边（有兴趣你自己加）
                # print(f"观测车辆车道{self.lane_index[2]}-要避障的车辆的车道-{obj.lane_index[2]}")
                if distance < distance_threshold*2.5 and self.lane_index[2] == obj.lane_index[2]:
                    nearby_obstacles.append(obj)
        return nearby_obstacles

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.
        """

        if self.is_observed:
            logger.info(f"观测车辆当前车速：{self.speed}")
            obstacles = self.get_nearby_obstacles()  # 获取障碍物
            if obstacles:
                closest_obstacle = min(obstacles, key=lambda obs: np.linalg.norm(obs.position - self.position))
                direction_to_obstacle = closest_obstacle.position - self.position
                target_heading = np.arctan2(direction_to_obstacle[1], direction_to_obstacle[0]) + np.pi/2
                # print(f"{self.lane_index[2]}--{type(self.lane_index[2])}--{target_heading}--{type(target_heading)}")
                if self.lane_index[2] == 0:
                    if target_heading < 0:  # 避免向左转，保持直行或向右
                        target_heading = -target_heading
                elif self.lane_index[2] == 3:
                    if target_heading > 0:  # 避免向左转，保持直行或向右
                        target_heading = -target_heading
                # print(f"1、看看有没有正确进if{self.action['steering']}：观测车辆车道{self.lane_index[2]}")
                self.action["steering"] = self.pid_controller.update(target_heading, self.heading, dt)
                # print("2、看看有没有正确进if：", self.action["steering"])
            else:
                self.action["steering"] = 0  # 无障碍物时，保持直线行驶
        else:
            self.action["steering"] = 0  # 非观察车辆时，保持直线行驶

        self.clip_actions()
        delta_f = self.action["steering"]  # 使用 PID 控制的 steering
        beta = np.arctan(1 / 2 * np.tan(delta_f))  # 侧滑角
        self.heading += self.action["steering"] * dt
        v = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt

        # 碰撞检测
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None

        # 处理换道逻辑
        new_lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
        if self.is_observed:
            logger.info(f"当前车道：{self.lane_index[2]}。当前位置：{self.position[1]} 换道目标车道：{new_lane_index[2]}.")
        if new_lane_index[2] != self.lane_index[2]:
            # 计算新车道中心位置，假设车道宽度为4
            """
            0
            --1     1车道中心坐标=（0+0.5）*4=0.5*4=2
            4
            --2     2车道中心坐标=（1+0.5）*4=1.5*4=6
            8
            --3     3车道中心坐标=（2+0.5）*4=2.5*4-2=10-2 = 8
            12
            --4     4车道中心坐标=（3+0.5）*4-2=3.5*4-2=14-2=12
            16
            """
            target_lane_center_y = (new_lane_index[2] + 0.5) * 4 - 2  # 车道宽度为4
            self.position[1] = target_lane_center_y  # 移动车辆到新车道的中心位置
            self.lane_index = new_lane_index
            self.lane = self.road.network.get_lane(self.lane_index)
            self.heading = self.lane.heading_at(self.position[0])  # 将航向调整为车道的方向
            logger.info(f"{self.position[1]}--{target_lane_center_y}--{self.lane_index}")
        # 更新速度
        self.speed += self.action["acceleration"] * dt
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
        # 调用状态更新
        self.on_state_update()

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
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

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
