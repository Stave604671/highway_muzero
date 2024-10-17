from __future__ import annotations

import numpy as np
import ray

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
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
                "collision_reward": -5,  # The reward received when colliding with a vehicle.
                # zero for other lanes.
                "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": -1,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "safe_distance_reward": 2,
                'not_collision_reward': 5,
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
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = MDPVehicle.create_random(
                self.road,
                lane_id=self.config["initial_lane_id"],
                speed=25,
                spacing=self.config["ego_spacing"],
                is_observed=True,
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed, is_observed=True
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"], is_observed=False
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        if self.vehicle.is_observed:
            ray.logger.info(f"观测车辆车速{self.vehicle.speed}--{rewards}----name")
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config['collision_reward'] + self.config["lane_change_reward"] - 1,
                 self.config['collision_reward'] + 1 + self.config["safe_distance_reward"]],  # reward 的最小值和最大值
                [0, 1],
            )
        # reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        speed_min, speed_max = self.config['reward_speed_range'][0], self.config['reward_speed_range'][1]
        speed_reward = -1 + 2 * (self.vehicle.speed * np.cos(self.vehicle.heading) - speed_min) / (
                    speed_max - speed_min)

        # 换道惩罚
        lane_change_reward = 0
        if hasattr(self.vehicle, 'last_lane_index'):
            if self.vehicle.lane_index[2] != self.vehicle.last_lane_index:
                lane_change_reward = self.config["lane_change_reward"]
        self.vehicle.last_lane_index = self.vehicle.lane_index[2]

        # 安全距离奖励
        min_distance_to_other_vehicles = min(
            [np.linalg.norm(v.position - self.vehicle.position) for v in self.road.vehicles if v != self.vehicle]
        )

        # 计算动态的安全距离阈值
        dynamic_safe_distance = 5 + 0.5 * self.vehicle.speed  # d_min = 5, β = 0.5

        # 使用非线性函数计算安全距离奖励
        safe_distance_reward = 1 - np.exp(-min_distance_to_other_vehicles / dynamic_safe_distance)

        return {
            "not_collision_reward": self.config['not_collision_reward'] if not self.vehicle.crashed else 0,
            "collision_reward": self.config['collision_reward'] if self.vehicle.crashed else 0,
            "lane_change_reward": lane_change_reward,
            "high_speed_reward": speed_reward,
            "safe_distance_reward": safe_distance_reward,
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
                "lanes_count": 4,
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
