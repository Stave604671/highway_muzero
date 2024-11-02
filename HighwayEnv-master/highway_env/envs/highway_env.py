from __future__ import annotations

import numpy as np
import ray

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

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
        config.update(  # highway_env环境的默认参数
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
                'safe_distance_reward': 0.5,
                'on_road_reward': 1,
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

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
        """使用指定的车辆类型，在道路上创建车辆对象"""
        # 获取非观测车辆的类型
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # 获取其余的观测车辆
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )
        # 存储受控制的观测车辆
        self.controlled_vehicles = []
        for others in other_per_controlled:
            # 创建观测车辆，注意这里是用哪个类创建的
            vehicle = Vehicle.create_random(
                self.road,
                lane_id=self.config["initial_lane_id"],
                speed=25,  # 观测车辆的初始速度，注释掉后，内部有默认的初始速度，在20-25
                spacing=self.config["ego_spacing"],
                is_observed=True,
            )
            # 传递给Action类，设置观测车辆的属性
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed, is_observed=True
            )
            self.controlled_vehicles.append(vehicle)
            # 讲观测车辆添加到道路上
            self.road.vehicles.append(vehicle)
            # 遍历其余的非观测车辆
            for _ in range(others):
                # 创建非观测车辆
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"], is_observed=False
                )
                vehicle.randomize_behavior()
                # 将非观测车辆添加到道路上
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # 计算reward数值
        rewards = self._rewards(action)
        # 日志记录
        if rewards['lane_change_reward']:
            ray.logger.info(
                f"车辆在尝试脱轨而得到了惩罚{self.config['lane_change_reward']}:{rewards['lane_change_reward']}")
        if rewards['collision_reward']:
            ray.logger.info(
                f"车辆在发生碰撞而得到了惩罚{self.config['collision_reward']}:{rewards['collision_reward']}")
        # 根据配置文件记录的权重，配置奖励数值
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        # 奖励数值正则化
        if self.config["normalize_reward"]:
            # 当奖励的数值处于[min,max]范围内时，将奖励的数值映射到[0，1]
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["lane_change_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        # 如果车辆脱离道路，代表当前脱轨，不予提供奖励
        reward *= rewards["on_road_reward"]
        # 将奖励返回给外部环境，和self_play里面的self.game.steps方法的返回值
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        # 获取道路状态
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        # 获取观测车辆的车道索引，第四车道的话，代表车辆在靠右行
        lane = (
            self.vehicle.target_lane_index[2]  # 不同的观测车辆获取车道编号的方法的差异
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # 获取横向速度
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # 当横向速度在self.config的范围内时，将速度奖励映射到[0,1]
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        # 更新 last_lane_index获取观测车辆的上一个车道号
        self.vehicle.last_lane_index = self.vehicle.lane_index[2]
        # 汇总奖励数值，
        # 碰撞"collision_reward"返回0，其余返回1
        # 靠右行返回"right_lane_reward"奖励
        # 车辆jerk大于2给一个惩罚lane_change_reward，这里需要返回正数，因为后面会和权重相乘，如果返回负数，会乘两次变正数
        # "on_road_reward"车辆是否在道路上
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "lane_change_reward": -self.config["lane_change_reward"] if (isinstance(self.vehicle.get_jerk_x, (
            int, float)) and self.vehicle.get_jerk_x > 2) or (isinstance(self.vehicle.get_jerk_y, (
            int, float)) and self.vehicle.get_jerk_y > 2) else 0,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
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
    高速公路环境继承前面的环境，只是修改部分参数，并且只检查观测车辆和其余车辆是否碰撞
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
