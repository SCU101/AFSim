import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from utils.tools import RAMathUtil
import math
from communication.tcp_client import SimulationClient


class PointTrackingEnv(gym.Env):
    """
    点跟踪环境，用于与仿真平台交互 (Gymnasium版本)
    """

    def __init__(self, simulation_client, max_steps: int = 200, render_mode: Optional[str] = None):
        """
        初始化环境

        Args:
            simulation_client: 仿真平台客户端
            max_steps: 每个episode的最大步数
            render_mode: 渲染模式，可选'human'或None
        """
        super(PointTrackingEnv, self).__init__()

        self.simulation = simulation_client
        self.simulation.connection(scenario="testWzz")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0

        # 定义动作空间：连续动作，控制点的移动 [x, y, z, 其他参数?]
        # 根据你的仿真平台调整维度
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),  # 最小控制值
            high=np.array([1.0, 1.0, 1.0, 1.0]),  # 最大控制值
            shape=(4,),
            dtype=np.float64
        )

        # 定义观测空间：环境返回的数据
        # 这里需要根据simulation.get_environment_data返回的实际结构调整
        # 假设观测是包含位置、速度等信息的向量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),  # 根据实际观测维度调整
            dtype=np.float64
        )

        # 目标位置（可以根据任务动态设置）
        self.target_position = np.array([12000.0, 0.0, 0.0])  # x, y, z

        # 中心点的经、纬、高度
        self.center_position = {'alt': 0.0, 'lat': 0.0, 'lon': 0.0}

        # 用于跟踪episode信息
        self.episode_reward = 0
        self.episode_length = 0

        # Gymnasium的metadata格式
        self.metadata = {"render_modes": ["human"], "render_fps": 30}

    def _process_observation(self, observation) -> np.ndarray:
        """
        处理原始观测数据，转换为numpy数组

        Args:
            raw_observation: 仿真平台返回的原始观测

        Returns:
            处理后的观测数组
        """
        plane_info = observation["data"]["0"]["obs"]['platforms'][0]
        delta_x, delta_y = RAMathUtil.convert_lat_long_to_xy(plane_info, self.target_position)
        delta_z = self.target_position["alt"] - plane_info["alt"]

        state = np.array([delta_x, delta_y, delta_z, plane_info["heading"],
                          plane_info["pitch"], plane_info["roll"], plane_info["speed"],
                          plane_info["vx"], plane_info["vy"], plane_info["vz"]], dtype=np.float64)
        return np.array(state)

    def _calculate_reward(self, observation, state) -> float:
        """
        计算奖励值

        Args:
            observation: 当前观测

        Returns:
            奖励值
        """
        # 提取当前位置（假设前3个元素是位置）
        delta_x, delta_y, delta_z = state[0], state[1], state[2]

        distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        # 奖励函数设计
        # 1. 距离惩罚（负奖励）
        distance_penalty = -distance * 0.0001

        # 2. 成功到达目标的奖励
        success_reward = 10.0 if distance < 500 else 0.0

        # 3. 时间惩罚（鼓励快速到达）
        time_penalty = -0.01

        # 4. 平滑性奖励（可选，需要速度信息）
        # velocity = observation[3:6]
        # smooth_penalty = -0.001 * np.linalg.norm(velocity)

        # 5.超出高度限制，判定飞机坠毁
        plane_info = observation["data"]["0"]["obs"]['platforms'][0]
        if plane_info['alt'] < 1000.0:
            return float(-10.0)

        total_reward = distance_penalty + success_reward + time_penalty

        return float(total_reward)

    def _check_terminated(self, observation, state) -> bool:
        """
        检查是否终止（任务完成）

        Args:
            observation: 当前观测

        Returns:
            bool: 是否终止
        """
        # 成功到达目标
        delta_x, delta_y, delta_z = state[0], state[1], state[2]
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        if distance < 500:  # 到达目标的阈值
            return True

        # 飞机坠毁
        plane_info = observation["data"]["0"]["obs"]['platforms'][0]
        if plane_info['alt'] < 1000.0:
            return True

        return False

    def _check_truncated(self, observation, state) -> bool:
        """
        检查是否截断（时间耗尽等）

        Args:
            observation: 当前观测

        Returns:
            bool: 是否截断
        """
        # 达到最大步数
        if self.current_step >= self.max_steps:
            return True

        return False

    def step(self, action: np.ndarray, slice: int = 60) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作向量

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 确保动作在合法范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i in range(slice):
            # 连续多少帧再重新生成一个新的动作
            observation = self.simulation.get_environment_data([action.tolist()])

        # 处理观测
        state = self._process_observation(observation)

        # 计算奖励
        reward = self._calculate_reward(observation, state)

        # 检查是否终止和截断
        terminated = self._check_terminated(observation, state)
        truncated = self._check_truncated(observation, state)

        # 更新步数
        self.current_step += 1
        self.episode_reward += reward
        self.episode_length += 1

        # 信息字典
        info = {
            'episode': {
                'r': self.episode_reward,
                'l': self.episode_length
            },
        }

        # 如果需要渲染
        if self.render_mode == "human":
            self.render()

        return state, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 重置选项，可以包含scenario和target_position

        Returns:
            tuple: (observation, info)
        """
        # 设置随机种子
        super().reset(seed=seed)

        # 从options中获取参数
        scenario = "testWzz"
        target_position = None

        if options is not None:
            scenario = options.get("scenario", "testWzz")
            target_position = options.get("target_position", None)

        # 重置仿真
        try:
            self.simulation.reset()
        except Exception as e:
            # 返回零观测和错误信息
            info = {"error": str(e)}
            return np.zeros(self.observation_space.shape, dtype=np.float64), info

        # 传入默认初始动作
        observation = self.simulation.get_environment_data([[0.5, 0.0, 0.0, 1.0]])

        # 设置新的目标位置
        if target_position is not None:
            self.target_position = np.array(target_position, dtype=np.float64)
        else:
            # 随机生成目标位置（可选）
            random_target_position = RAMathUtil.generate_target_arc()
            self.target_position = RAMathUtil.convert_xy_to_lat_long(
                observation["data"]["0"]["obs"]['platforms'][0],
                random_target_position[0],
                random_target_position[1],
                delta_z=random_target_position[2]
            )

        # 重置步数和奖励
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0

        self.center_position = observation["data"]["0"]["obs"]['platforms'][0]

        # 处理观测
        state = self._process_observation(observation)

        # 构建信息字典
        info = {
            "initial_target_position": self.target_position.copy(),
            "initial_observation": observation
        }

        # 如果需要渲染
        if self.render_mode == "human":
            self.render()

        return state, info

    def render(self):
        """
        渲染环境（可选）待完成
        """
        pass
        # if self.render_mode == "human":
        #     # 这里可以添加可视化代码
        #     print(f"Step: {self.current_step}, Reward: {self.episode_reward:.2f}")
        # elif self.render_mode is not None:
        #     # Gymnasium要求如果render_mode不为None，必须实现render方法
        #     raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

    def close(self):
        """
        关闭环境
        """
        if hasattr(self, 'simulation') and self.simulation:
            self.simulation.close()


# 使用示例
if __name__ == "__main__":
    # 1. 创建环境（Gymnasium版本）
    simulation = SimulationClient(host='127.0.0.1', port=8888)
    env = PointTrackingEnv(simulation_client=simulation, max_steps=200, render_mode="human")

    # 重置环境，现在返回两个值
    state, info = env.reset()

    for i in range(1000):
        action = np.array([0.5, 0.0, 0.0, 1.0])
        # Gymnasium的step返回5个值
        state, reward, terminated, truncated, info = env.step(action)

        # 检查是否结束（终止或截断）
        done = terminated or truncated

        if done:
            print(f"Episode ended: reward={reward}, terminated={terminated}, truncated={truncated}")
            state, info = env.reset()

    env.close()
