from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from communication.tcp_client import SimulationClient
from core.environments.dog_fight.dog_fight_env import DogFightEnv


def make_env():
    simulation = SimulationClient(host='127.0.0.1', port=8888)
    env = DogFightEnv(simulation_client=simulation, max_steps=200)
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
    tensorboard_log="./ppo_dogfight_tensorboard/"
)

# 训练
model.learn(total_timesteps=100000)
model.save("ppo_dogfight")

# 测试
# env = make_env()
# obs, info = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()

env.close()
