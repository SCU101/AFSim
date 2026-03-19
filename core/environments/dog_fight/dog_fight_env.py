import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from utils.tools import RAMathUtil
import math, os, json
from communication.tcp_client import SimulationClient
from datetime import datetime, timedelta
from visualization.tacview_handler import TacView


class DogFightEnv(gym.Env):
    """
    1v1缠斗环境，用于与仿真平台交互 (Gymnasium版本)
    """

    def __init__(self, simulation_client, max_steps: int = 200, render_mode: Optional[str] = None):
        """
        初始化环境

        Args:
            simulation_client: 仿真平台客户端
            max_steps: 每个episode的最大步数
            render_mode: 渲染模式，可选'human'或None
        """
        super(DogFightEnv, self).__init__()

        self.view_server = None
        self.simulation = simulation_client
        self.simulation.connection(scenario="test1v1")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0
        self.action_pre = np.zeros(4)
        self.observation = None

        # 定义动作空间：连续动作，控制我方飞机 [升降舵, 副翼, 方向舵, 油门]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),  # 最小控制值
            high=np.array([1.0, 1.0, 1.0, 1.0]),  # 最大控制值
            shape=(4,),
            dtype=np.float64
        )

        # 定义观测空间：环境返回的数据
        # 包含我方飞机和敌方飞机的状态信息
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(31,),  # 我方12维 + 敌方12维 + 相对位置3维 + 上一次动作4维
            dtype=np.float64
        )

        # 敌方飞机的固定策略
        self.enemy_action = [0.5, 0.0, 0.0, 1.0]  # 直飞

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
        platforms = observation["data"]["0"]["obs"]['platforms']
        
        # 找到我方飞机和敌方飞机
        my_plane = None
        enemy_plane = None
        
        for platform in platforms:
            if platform["name"] == "1001":
                my_plane = platform
            elif platform["name"] == "5001":
                enemy_plane = platform
        
        if my_plane is None or enemy_plane is None:
            return np.zeros(self.observation_space.shape, dtype=np.float64)
        
        # 计算相对位置
        delta_x, delta_y = RAMathUtil.convert_lat_long_to_xy(my_plane, enemy_plane)
        delta_z = enemy_plane["alt"] - my_plane["alt"]
        
        # 构建状态向量
        # 我方飞机状态：[heading, pitch, roll, speed, vx, vy, vz, alt, beta, pitch_rate, roll_rate, mass]
        my_state = np.array([
            my_plane["heading"], my_plane["pitch"], my_plane["roll"], my_plane["speed"],
            my_plane["vx"], my_plane["vy"], my_plane["vz"], my_plane["alt"], my_plane["beta"],
            my_plane["pitch_rate"], my_plane["roll_rate"], my_plane["mass"]
        ], dtype=np.float64)
        
        # 敌方飞机状态：[heading, pitch, roll, speed, vx, vy, vz, alt, beta, pitch_rate, roll_rate, mass]
        enemy_state = np.array([
            enemy_plane["heading"], enemy_plane["pitch"], enemy_plane["roll"], enemy_plane["speed"],
            enemy_plane["vx"], enemy_plane["vy"], enemy_plane["vz"], enemy_plane["alt"], enemy_plane["beta"],
            enemy_plane["pitch_rate"], enemy_plane["roll_rate"], enemy_plane["mass"]
        ], dtype=np.float64)
        
        # 相对位置：[delta_x, delta_y, delta_z]
        relative_pos = np.array([delta_x, delta_y, delta_z], dtype=np.float64)
        
        # 上一次动作：[elevator, aileron, rudder, throttle]
        prev_action = self.action_pre
        
        # 组合所有状态
        state = np.concatenate([my_state, enemy_state, relative_pos, prev_action])
        
        return state

    def _calculate_reward(self, observation, state) -> float:
        """
        计算奖励值

        Args:
            observation: 当前观测
            state: 处理后的状态向量

        Returns:
            奖励值
        """
        platforms = observation["data"]["0"]["obs"]['platforms']
        
        # 找到我方飞机和敌方飞机
        my_plane = None
        enemy_plane = None
        
        for platform in platforms:
            if platform["name"] == "1001":
                my_plane = platform
            elif platform["name"] == "5001":
                enemy_plane = platform
        
        if my_plane is None or enemy_plane is None:
            return 0.0
        
        # 计算相对距离
        delta_x, delta_y = RAMathUtil.convert_lat_long_to_xy(my_plane, enemy_plane)
        delta_z = enemy_plane["alt"] - my_plane["alt"]
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        
        # 计算角度优势（我方飞机朝向敌方飞机的角度）
        my_heading = my_plane["heading"]
        dx, dy = delta_x, delta_y
        angle_to_enemy = math.atan2(dx, dy)
        angle_diff = abs(my_heading - angle_to_enemy)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        
        # 奖励函数设计
        # 1. 距离奖励（越近越好，但不能太近）
        if distance < 500:
            distance_reward = -0.1  # 太近了，危险
        elif distance < 2000:
            distance_reward = 0.01 * (2000 - distance) / 1500  # 2000米内，越近越好
        else:
            distance_reward = -0.000005 * (distance - 2000)  # 太远了，惩罚（系数调小）
        
        # 2. 角度优势奖励（我方朝向敌方飞机）
        angle_reward = 0.1 * (1 - angle_diff / math.pi)
        
        # 3. 高度优势奖励（我方在敌方上方）
        altitude_diff = my_plane["alt"] - enemy_plane["alt"]
        altitude_reward = 0.001 * min(altitude_diff, 1000) / 1000
        
        # 4. 速度优势奖励
        speed_diff = my_plane["speed"] - enemy_plane["speed"]
        speed_reward = 0.001 * min(speed_diff, 100) / 100
        
        # 5. 时间惩罚（鼓励快速战斗）
        time_penalty = -0.001
        
        # 6. 坠毁惩罚
        if my_plane['alt'] < 1000.0:
            return -10.0
        
        # 7. 击杀奖励（敌方坠毁）
        if enemy_plane['alt'] < 1000.0:
            return 100.0
        
        total_reward = distance_reward + angle_reward + altitude_reward + speed_reward + time_penalty
        
        return float(total_reward)

    def _check_terminated(self, observation, state) -> bool:
        """
        检查是否终止（任务完成）

        Args:
            observation: 当前观测
            state: 处理后的状态向量

        Returns:
            bool: 是否终止
        """
        platforms = observation["data"]["0"]["obs"]['platforms']
        
        # 找到我方飞机和敌方飞机
        my_plane = None
        enemy_plane = None
        
        for platform in platforms:
            if platform["name"] == "1001":
                my_plane = platform
            elif platform["name"] == "5001":
                enemy_plane = platform
        
        if my_plane is None or enemy_plane is None:
            return True
        
        # 我方飞机坠毁
        if my_plane['alt'] < 1000.0:
            print("terminated: my plane crashed")
            return True
        
        # 敌方飞机坠毁（击杀成功）
        if enemy_plane['alt'] < 1000.0:
            print("terminated: enemy plane crashed (kill success)")
            return True
        
        # 先不要因为距离大就直接终止
        return False

    def _check_truncated(self, observation, state) -> bool:
        """
        检查是否截断（时间耗尽等）

        Args:
            observation: 当前观测
            state: 处理后的状态向量

        Returns:
            bool: 是否截断
        """
        # 达到最大步数
        if self.current_step >= self.max_steps:
            return True
        
        # 距离太远（脱离战斗）- 放宽到50km避免过早截断
        platforms = observation["data"]["0"]["obs"]['platforms']
        my_plane = None
        enemy_plane = None
        
        for platform in platforms:
            if platform["name"] == "1001":
                my_plane = platform
            elif platform["name"] == "5001":
                enemy_plane = platform
        
        if my_plane is not None and enemy_plane is not None:
            delta_x, delta_y = RAMathUtil.convert_lat_long_to_xy(my_plane, enemy_plane)
            delta_z = enemy_plane["alt"] - my_plane["alt"]
            distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
            
            if distance > 50000:  # 50km，避免过早截断
                print(f"truncated: distance too large = {distance}")
                return True
        
        return False

    def step(self, action: np.ndarray, slice: int = 1) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作向量（我方飞机的动作）
            slice: 连续多少帧执行同一个动作

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 确保动作在合法范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.action_pre = action
        
        # 构建两架飞机的动作：我方飞机使用RL动作，敌方飞机使用固定策略
        actions = [action.tolist(), self.enemy_action]
        
        for i in range(slice):
            # 连续多少帧再重新生成一个新的动作
            observation = self.simulation.get_environment_data(actions)
            self.observation = observation
        
        # 处理观测
        state = self._process_observation(observation)
        
        # 计算奖励
        reward = self._calculate_reward(observation, state)
        
        # 检查是否终止和截断
        terminated = self._check_terminated(observation, state)
        if terminated:
            print(f"Episode ended (terminated) with reward: {reward}")
        truncated = self._check_truncated(observation, state)
        if truncated:
            print(f"Episode ended (truncated) with reward: {reward}")
        
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
            self.log_save()
        elif self.render_mode == "real":
            self.real_time_view()
        
        return state, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 重置选项

        Returns:
            tuple: (observation, info)
        """
        # 设置随机种子
        super().reset(seed=seed)
        
        self.observation = None
        self.reset_logs()
        
        # 重置仿真
        try:
            self.simulation.reset()
        except Exception as e:
            # 返回零观测和错误信息
            info = {"error": str(e)}
            return np.zeros(self.observation_space.shape, dtype=np.float64), info
        
        # 传入初始动作（我方飞机和敌方飞机）
        actions = [[0.5, 0.0, 0.0, 1.0], self.enemy_action]
        observation = self.simulation.get_environment_data(actions)
        self.observation = observation
        
        # 重置步数和奖励
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.action_pre = np.zeros(4)
        
        # 处理观测
        state = self._process_observation(observation)
        
        # 构建信息字典
        info = {
            "initial_observation": observation
        }
        
        # 如果需要渲染
        if self.render_mode == "human":
            self.log_save()
        elif self.render_mode == "real":
            self.real_time_view()
        
        return state, info

    def log_save(self, output_dir='logs', output_file='dogfight.acmi'):
        """
        保存ACMI格式的日志文件
        """
        output_path = os.path.join(output_dir, output_file)
        
        if not hasattr(self, 'observation') or self.observation is None:
            return None
        
        try:
            # 解析数据
            if isinstance(self.observation, str):
                data = json.loads(self.observation)
            else:
                data = self.observation
            
            platforms = data.get('data', {}).get('0', {}).get('obs', {}).get('platforms', [])
            sim_time = data.get('data', {}).get('0', {}).get('obs', {}).get('sim_time', 0)
            
            if not platforms:
                return None
            
            # 初始化（如果是第一次调用）
            if not hasattr(self, '_base_time'):
                self._base_time = datetime.now()
                self._frame_count = 0
                self._platform_ids = {'1001': '5160', '5001': '5161'}  # 我方和敌方飞机的ID
            
            # 以追加模式打开文件
            with open(output_path, 'a', encoding='utf-8') as f:
                # 如果是新文件，写入头信息
                if self._frame_count == 0:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.2\n")
                    f.write(f"0,ReferenceTime={self._base_time.strftime('%Y-%m-%dT%H:%M:%S')}Z\n")
                
                # 写入时间戳
                f.write(f"#{sim_time:.2f}\n")
                
                # 为每个平台写入数据
                for i, platform in enumerate(platforms):
                    name = platform.get('name', '1001')
                    
                    # 获取ID映射
                    if name in self._platform_ids:
                        object_id = self._platform_ids[name]
                    else:
                        object_id = str(5160 + i)
                        self._platform_ids[name] = object_id
                    
                    # 获取数据
                    lat = platform.get('lat', 0)
                    lon = platform.get('lon', 0)
                    alt = platform.get('alt', 0)
                    roll = platform.get('roll', 0)
                    pitch = platform.get('pitch', 0)
                    heading = platform.get('heading', 0)
                    
                    # 构建数据行
                    # 格式: ID,T=时间戳|经度|纬度|高度|滚转|俯仰|偏航,Name=名称,Type=类型,CallSign=呼号,Color=颜色
                    if name == "1001":
                        data_line = (f"{object_id},T={lon:.8f}|{lat:.8f}|{alt:.2f}|"
                                     f"{roll:.12f}|{pitch:.12f}|{heading:.6f},"
                                     f"Name=F-16,Type=Air+FixedWing,CallSign=Friend,Color=Red")
                    else:
                        data_line = (f"{object_id},T={lon:.8f}|{lat:.8f}|{alt:.2f}|"
                                     f"{roll:.12f}|{pitch:.12f}|{heading:.6f},"
                                     f"Name=F-16,Type=Air+FixedWing,CallSign=Enemy,Color=Blue")
                    
                    f.write(data_line + "\n")
                
                self._frame_count += 1
            
            return output_path
        
        except Exception as e:
            print(f"错误: {e}")
            return None

    def reset_logs(self, output_dir='logs', output_file='dogfight.acmi'):
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")
        # 构建完整的输出文件路径
        output_path = os.path.join(output_dir, output_file)
        
        # 如果文件已存在，先删除
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"删除已存在的文件: {output_path}")

    def real_time_view(self, host='127.0.0.1', port=42674):
        """
        实时可视化
        """
        if self.view_server is None:
            self.view_server = TacView(host=host, port=port)
            self._platform_ids = {'1001': '5160', '5001': '5161'}  # 我方和敌方飞机的ID
        
        if not hasattr(self, 'observation') or self.observation is None:
            return None
        try:
            # 解析数据
            if isinstance(self.observation, str):
                data = json.loads(self.observation)
            else:
                data = self.observation
            
            platforms = data.get('data', {}).get('0', {}).get('obs', {}).get('platforms', [])
            
            if not platforms:
                return None
            
            # 为每个平台写入数据
            for i, platform in enumerate(platforms):
                name = platform.get('name', '1001')
                # 获取ID映射
                if name in self._platform_ids:
                    object_id = self._platform_ids[name]
                else:
                    object_id = str(5160 + i)
                    self._platform_ids[name] = object_id
                
                # 获取数据
                lat = platform.get('lat', 0)
                lon = platform.get('lon', 0)
                alt = platform.get('alt', 0)
                roll = platform.get('roll', 0)
                pitch = platform.get('pitch', 0)
                heading = platform.get('heading', 0)
                
                # 构建数据行
                # 格式: ID,T=时间戳|经度|纬度|高度|滚转|俯仰|偏航,Name=名称,Type=类型,CallSign=呼号,Color=颜色
                if name == "1001":
                    data_line = (f"{object_id},T={lon:.8f}|{lat:.8f}|{alt:.2f}|"
                                 f"{roll:.12f}|{pitch:.12f}|{heading:.6f},"
                                 f"Name=F-16,Type=Air+FixedWing,CallSign=Friend,Color=Red")
                else:
                    data_line = (f"{object_id},T={lon:.8f}|{lat:.8f}|{alt:.2f}|"
                                 f"{roll:.12f}|{pitch:.12f}|{heading:.6f},"
                                 f"Name=F-16,Type=Air+FixedWing,CallSign=Enemy,Color=Blue")
                
                self.view_server.send_data_to_client((data_line + "\n").encode())
                time.sleep(0.01)
        
        except Exception as e:
            print(f"错误: {e}")
            return None

    def close(self):
        """
        关闭环境
        """
        if hasattr(self, 'simulation') and self.simulation:
            self.simulation.close()


# 使用示例
if __name__ == "__main__":
    # 1. 创建环境（Gymnasium版本）
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=1)
    env = DogFightEnv(simulation_client=simulation, max_steps=2000, render_mode="human")
    # env = DogFightEnv(simulation_client=simulation, max_steps=2000, render_mode="real")
    
    # 重置环境，现在返回两个值
    state, info = env.reset()
    
    for i in range(2000):
        action = np.array([0.5, 0.0, 0.0, 1.0])
        # Gymnasium的step返回5个值
        state, reward, terminated, truncated, info = env.step(action)
        
        # 检查是否结束（终止或截断）
        done = terminated or truncated
        
        if done:
            print(f"Episode ended: reward={reward}, terminated={terminated}, truncated={truncated}")
            # state, info = env.reset()
            break
    
    env.close()
