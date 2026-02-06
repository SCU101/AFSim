from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from communication.tcp_client import SimulationClient
from core.environments.point_tracking.point_tracking_env import PointTrackingEnv


def make_env():
    simulation = SimulationClient(host='127.0.0.1', port=8888)
    env = PointTrackingEnv(simulation_client=simulation, max_steps=200)
    return env


# 创建向量化环境
env = DummyVecEnv([make_env])

# 创建模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./ppo_tracking_tensorboard/"
)

# 训练
model.learn(total_timesteps=100000)
model.save("ppo_point_tracking")

# 测试
# env = make_env()
# obs, info = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()

env.close()
