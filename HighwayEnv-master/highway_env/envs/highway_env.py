from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from ray import logger
import threading

lock = threading.Lock()
from queue import Queue

# 使用线程局部存储
thread_local = threading.local()
state_queue = Queue()
Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,
                # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )

        return config

    # def _reset(self) -> None:
    #     self._create_road()
    #     self._create_vehicles()
    #     #self.previous_state = self._save_vehicle_state()  # 保存初始状态
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.state_saved = False
        self._initialize_thread_state()  # 初始化线程状态
        self.previous_state = self._save_vehicle_state()  # 保存初始状态

    def _initialize_thread_state(self) -> None:
        """初始化线程的状态存储"""
        if not hasattr(thread_local, 'previous_state'):
            thread_local.previous_state = self._save_vehicle_state()

    def _save_vehicle_state(self) -> Dict:
        """保存车辆的状态"""
        state = {
            'lane_index': self.vehicle.lane_index[:],
            'position': self.vehicle.position.copy()
        }
        return state
        # return {
        #     'lane_index': self.vehicle.lane_index[:],
        #     'position': self.vehicle.position.copy()  # 保存车辆位置
        # }

    # def _restore_vehicle_state(self, state: Dict) -> None:
    #     """恢复车辆的状态"""
    #     print("瞅瞅stste里都有啥：",state)
    #     self.vehicle.lane_index = state['lane_index']
    #     self.vehicle.position = state['position']
    #     print("看一下新的state：",self.vehicle.lane_index,self.vehicle.position)
    #     print(f"恢复后是否在车道上: {self.vehicle.on_road}")

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def get_nearest_road_position(self, current_position) -> np.ndarray:
        """
        获取车辆最接近的车道中心位置，以便将车辆重新放回道路上。
        :param current_position: 车辆当前的位置
        :return: 车辆应该被重置到的道路上的位置
        """
        road_network = self.road.network
        lane_index = self.vehicle.lane_index  # 获取当前车道的索引
        lane = road_network.get_lane(lane_index)  # 获取当前车道对象
        # 返回车道中心线最接近的点
        nearest_position = lane.position(current_position[0], current_position[1])
        # 调试信息
        # print(f"当前位置: {current_position}, 最近车道位置: {nearest_position}")
        return nearest_position

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)

        if self.vehicle.on_road:
            if not self.state_saved:  # 检查标志变量
                # logger.info(f"是否在车道上: {self.vehicle.on_road}，是的话保存当前状态")
                self.previous_state = self._save_vehicle_state()
                # logger.info(f"看一下保存的state：{self.previous_state}")
                self.state_saved = True  # 一旦保存，设置标志为True
            rewards["right_lane_reward"] = 0.5
        else:
            # 将车辆重新放回道路上，恢复到之前保存的状态{'lane_index': ('0', '1', 3), 'position': array([138.129816  ,  13.11193569])}
            self.vehicle.lane_index = self.previous_state["lane_index"]
            self.vehicle.position = self.previous_state["position"]
            # print("看一下新的state：",self.vehicle.lane_index,self.vehicle.position)
            # print(f"恢复后是否在车道上: {self.vehicle.on_road}")
            # self._restore_vehicle_state(self.previous_state)
            rewards["lane_change_reward"] = -3.0  # 更大的越界惩罚

        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["lane_change_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        # 添加时间步惩罚，防止车辆拖延时间
        time_step_penalty = -0.01  # 每个时间步增加轻微惩罚
        reward += time_step_penalty
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # scaled_speed = utils.lmap(
        #     forward_speed, self.config["reward_speed_range"], [0, 1]
        # )
        # high_speed_reward = np.clip(scaled_speed, 0, 1)  # 保证结果在 [0, 1] 范围内
        v_min, v_max = self.config["reward_speed_range"]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        high_speed_reward = -1 + (forward_speed - v_min) / (v_max - v_min)
        high_speed_reward = np.clip(high_speed_reward, -1, 1)  # 保证结果在 [-1, 1] 范围内

        # 引入低速惩罚
        low_speed_penalty = 0
        min_speed_threshold = self.config.get("min_speed_threshold", 5)  # 最低速度阈值
        if forward_speed < min_speed_threshold:
            low_speed_penalty = -1.0  # 低速时的惩罚


        # 判断是否进行了换道操作
        # 判断是否进行了换道操作并且保持在道路上
        lane_change_reward = 0
        if hasattr(self.vehicle, 'last_lane_index'):
            if self.vehicle.lane_index[2] != self.vehicle.last_lane_index:
                lane_change_reward = self.config["lane_change_reward"]
        # 更新 last_lane_index
        self.vehicle.last_lane_index = self.vehicle.lane_index[2]
        # 计算并返回各项奖励
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": high_speed_reward,
            "lane_change_reward": lane_change_reward,
            "on_road_reward": float(self.vehicle.on_road) + 1,
            "low_speed_penalty": low_speed_penalty,
            # "acceleration_reward": acceleration_reward,
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
                self.vehicle.crashed
                or self.config["offroad_terminal"]
                and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
