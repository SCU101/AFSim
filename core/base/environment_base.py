import gymnasium as gym
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

from communication.http_client import SimulationClient
from visualization.tacview_handler import TacViewHandler


class AirCombatEnvironmentBase(gym.Env, ABC):
    """空战环境基类"""

    def __init__(self,
                 env_name: str,
                 sim_client: SimulationClient,
                 render: bool = False,
                 save_acmi: bool = False,
                 acmi_file_path: str = None):

        self.env_name = env_name
        self.sim_client = sim_client
        self.render = render
        self.save_acmi = save_acmi
        self.acmi_file_path = acmi_file_path

        # 初始化组件 - 由子类实现具体实例
        self.feature_extractor = None
        self.reward_calculator = None
        self.termination_checker = None

        # TacView处理器
        self.tacview_handler = TacViewHandler() if render or save_acmi else None

        # 连接仿真服务端
        if not self.sim_client.connect(env_name):
            raise ConnectionError(f"无法连接到环境: {env_name}")

    @abstractmethod
    def _init_components(self):
        """初始化环境特定组件"""
        pass

    @abstractmethod
    def _define_action_space(self) -> gym.Space:
        """定义动作空间"""
        pass

    @abstractmethod
    def _define_observation_space(self) -> gym.Space:
        """定义观察空间"""
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境 - 通用实现"""
        super().reset(seed=seed)

        if not self.sim_client.reset_environment():
            raise RuntimeError("环境重置失败")

        # 获取初始环境数据
        env_data = self.sim_client.get_environment_data()
        if env_data is None:
            raise RuntimeError("获取初始环境数据失败")

        # 特征提取
        observation = self.feature_extractor.extract(env_data)
        info = {"raw_data": env_data}

        # TacView处理
        self._handle_visualization(env_data)

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步动作 - 通用实现"""
        # 发送动作到仿真服务端
        action_dict = self._convert_action_to_dict(action)
        if not self.sim_client.send_action(action_dict):
            raise RuntimeError("发送动作失败")

        # 获取新的环境数据
        env_data = self.sim_client.get_environment_data()
        if env_data is None:
            raise RuntimeError("获取环境数据失败")

        # 特征提取
        observation = self.feature_extractor.extract(env_data)

        # 计算奖励
        reward = self.reward_calculator.calculate(env_data, action_dict)

        # 检查终止条件
        terminated = self.termination_checker.is_terminated(env_data)
        truncated = self.termination_checker.is_truncated(env_data)

        info = {
            "raw_data": env_data,
            "action": action_dict,
            "reward_components": self.reward_calculator.get_reward_components()
        }

        # TacView处理
        self._handle_visualization(env_data)

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _convert_action_to_dict(self, action: np.ndarray) -> Dict[str, Any]:
        """将numpy动作数组转换为字典格式"""
        pass

    def _handle_visualization(self, env_data: Dict[str, Any]):
        """处理可视化"""
        if self.tacview_handler:
            if self.render:
                self.tacview_handler.send_to_tacview(env_data)
            if self.save_acmi and self.acmi_file_path:
                self.tacview_handler.save_acmi(env_data, self.acmi_file_path)

    def close(self):
        """关闭环境"""
        if self.tacview_handler:
            self.tacview_handler.close()