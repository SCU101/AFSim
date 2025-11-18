import gymnasium as gym
import numpy as np
from typing import Dict, Any

from core.base.environment_base import AirCombatEnvironmentBase
from .feature_extractor import BVRCombatFeatureExtractor
from .reward_calculator import BVRCombatRewardCalculator
from .termination_checker import BVRCombatTerminationChecker


class BVRCombatEnvironment(AirCombatEnvironmentBase):
    """超视距空战环境"""

    def __init__(self,
                 sim_client,
                 render: bool = False,
                 save_acmi: bool = False,
                 acmi_file_path: str = None):
        super().__init__(
            env_name="bvr_combat",
            sim_client=sim_client,
            render=render,
            save_acmi=save_acmi,
            acmi_file_path=acmi_file_path
        )

        self._init_components()
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

    def _init_components(self):
        """初始化BVR环境组件"""
        self.feature_extractor = BVRCombatFeatureExtractor()
        self.reward_calculator = BVRCombatRewardCalculator()
        self.termination_checker = BVRCombatTerminationChecker()

    def _define_action_space(self) -> gym.Space:
        """定义BVR空战动作空间 - 更复杂的雷达和武器控制"""
        return gym.spaces.Dict({
            "flight_controls": gym.spaces.Box(
                low=np.array([0, -1, -1, -1]),
                high=np.array([1, 1, 1, 1]),
                dtype=np.float32
            ),
            "sensor_controls": gym.spaces.Discrete(4),  # 雷达模式
            "weapon_controls": gym.spaces.MultiBinary(3)  # 锁定, 发射, 对抗
        })

    def _define_observation_space(self) -> gym.Space:
        """定义BVR观察空间 - 包含雷达接触信息"""
        obs_dim = self.feature_extractor.get_feature_dimension()
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _convert_action_to_dict(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """转换BVR动作为仿真格式"""
        flight = action["flight_controls"]
        return {
            "throttle": float(flight[0]),
            "pitch": float(flight[1]),
            "roll": float(flight[2]),
            "yaw": float(flight[3]),
            "radar_mode": int(action["sensor_controls"]),
            "lock_target": bool(action["weapon_controls"][0]),
            "fire_missile": bool(action["weapon_controls"][1]),
            "deploy_countermeasures": bool(action["weapon_controls"][2])
        }