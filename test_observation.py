import numpy as np
from communication.tcp_client import SimulationClient

# 连接服务器
simulation = SimulationClient(host='127.0.0.1', port=8888, steps=1)
simulation.connection(scenario="test1v1")

# 重置环境
simulation.reset()

# 获取一次观测
actions = [[0.5, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]
observation = simulation.get_environment_data(actions)

# 打印所有字段
platforms = observation["data"]["0"]["obs"]['platforms']
print(f"飞机数量: {len(platforms)}")
print("\n" + "="*50)

for i, platform in enumerate(platforms):
    print(f"\n飞机 {i+1} (ID: {platform['name']}):")
    print(f"所有字段: {list(platform.keys())}")
    print(f"字段数量: {len(platform)}")
    print("\n字段详情:")
    for key, value in platform.items():
        print(f"  {key}: {value}")

# 计算需要的维度
print("\n" + "="*50)
print("维度分析:")
print(f"如果包含所有字段: {len(platforms[0])} 维/架")
print(f"两架飞机总维度: {len(platforms[0]) * 2} 维")
print(f"加上相对位置 (3维): {len(platforms[0]) * 2 + 3} 维")
print(f"加上上一次动作 (4维): {len(platforms[0]) * 2 + 3 + 4} 维")

simulation.close()
