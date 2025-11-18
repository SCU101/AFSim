import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from core.environment import AirCombatEnv
from communication.http_client import SimulationClient


def train_single_agent_combat():
    """单智能体对抗训练"""

    # 初始化仿真客户端
    sim_client = SimulationClient("http://simulation-server:8080")

    # 创建环境
    env = AirCombatEnv(
        sim_client=sim_client,
        env_name="air_combat_basic",
        render=False,
        save_acmi=True,
        acmi_file_path="./logs/training.acmi"
    )

    # 检查环境兼容性
    check_env(env)

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )

    # 训练模型
    model.learn(
        total_timesteps=1_000_000,
        log_interval=10,
        tb_log_name="single_agent_combat"
    )

    # 保存模型
    model.save("./models/single_agent_combat")

    env.close()


if __name__ == "__main__":
    train_single_agent_combat()
