import gymnasium as gym
import numpy as np
from typing import Dict, Any

from core.base.environment_base import AirCombatEnvironmentBase
from .feature_extractor import BasicCombatFeatureExtractor
from .reward_calculator import BasicCombatRewardCalculator
from .termination_checker import BasicCombatTerminationChecker


class BasicCombatEnvironment(AirCombatEnvironmentBase):
    """基础空战环境"""

    def __init__(self,
                 sim_client,
                 render: bool = False,
                 save_acmi: bool = False,
                 acmi_file_path: str = None):
        super().__init__(
            env_name="basic_combat",
            sim_client=sim_client,
            render=render,
            save_acmi=save_acmi,
            acmi_file_path=acmi_file_path
        )

        self._init_components()
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

    def _init_components(self):
        """初始化基础空战环境组件"""
        self.feature_extractor = BasicCombatFeatureExtractor()
        self.reward_calculator = BasicCombatRewardCalculator()
        self.termination_checker = BasicCombatTerminationChecker()

    def _define_action_space(self) -> gym.Space:
        """定义基础空战动作空间"""
        return gym.spaces.Box(
            low=np.array([0, -1, -1, -1, 0, 0]),  # 油门, 俯仰, 滚转, 偏航, 武器, 对抗
            high=np.array([1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )

    def _define_observation_space(self) -> gym.Space:
        """定义基础空战观察空间"""
        obs_dim = self.feature_extractor.get_feature_dimension()
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _convert_action_to_dict(self, action: np.ndarray) -> Dict[str, Any]:
        """转换动作为仿真格式"""
        return {
            "throttle": float(action[0]),
            "pitch": float(action[1]),
            "roll": float(action[2]),
            "yaw": float(action[3]),
            "weapon_control": int(action[4] > 0.5),  # 二值化
            "countermeasures": int(action[5] > 0.5)  # 二值化
        }
