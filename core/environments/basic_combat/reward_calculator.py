from typing import Dict, Any
from core.base.reward_calculator_base import RewardCalculatorBase
from utils.math_functions import calculate_distance, calculate_angles


class BasicCombatRewardCalculator(RewardCalculatorBase):
    """基础空战奖励计算器"""

    def calculate(self, env_data: Dict[str, Any], action: Dict[str, Any]) -> float:
        """计算基础空战奖励"""
        self.reward_components = {}

        # 生存奖励
        survival_reward = self._calculate_survival_reward(env_data)

        # 距离奖励 - 保持在理想交战距离
        distance_reward = self._calculate_distance_reward(env_data)

        # 角度奖励 - 占据有利位置
        angle_reward = self._calculate_angle_reward(env_data)

        # 能量奖励 - 保持能量优势
        energy_reward = self._calculate_energy_reward(env_data)

        # 动作惩罚
        action_penalty = self._calculate_action_penalty(action)

        # 战斗结果奖励
        combat_reward = self._calculate_combat_reward(env_data)

        total_reward = (
                survival_reward +
                distance_reward +
                angle_reward +
                energy_reward +
                action_penalty +
                combat_reward
        )

        self.reward_components = {
            "survival": survival_reward,
            "distance": distance_reward,
            "angle": angle_reward,
            "energy": energy_reward,
            "action_penalty": action_penalty,
            "combat": combat_reward,
            "total": total_reward
        }

        return total_reward

    def _calculate_survival_reward(self, env_data: Dict) -> float:
        """生存奖励 - 每步给予小奖励"""
        return 0.01

    def _calculate_distance_reward(self, env_data: Dict) -> float:
        """距离奖励 - 鼓励保持在理想交战距离"""
        ownship = env_data.get("ownship", {})
        enemy = env_data.get("enemies", [{}])[0] if env_data.get("enemies") else {}

        if not enemy:
            return 0.0

        distance = calculate_distance(
            ownship.get("position", {}),
            enemy.get("position", {})
        )

        # 理想交战距离为5-10公里
        ideal_min, ideal_max = 5000, 10000
        if ideal_min <= distance <= ideal_max:
            return 0.1
        elif distance < ideal_min:
            # 太近惩罚
            return -0.05 * (ideal_min - distance) / 1000
        else:
            # 太远惩罚
            return -0.02 * (distance - ideal_max) / 1000

    def _calculate_angle_reward(self, env_data: Dict) -> float:
        """角度奖励 - 占据有利攻击位置"""
        ownship = env_data.get("ownship", {})
        enemy = env_data.get("enemies", [{}])[0] if env_data.get("enemies") else {}

        if not enemy:
            return 0.0

        angles = calculate_angles(ownship, enemy)
        bearing = abs(angles.get("bearing", 0))
        elevation = abs(angles.get("elevation", 0))

        # 奖励良好的攻击角度（敌机在正前方小角度内）
        bearing_reward = max(0, 1.0 - bearing / 30)  # 30度内最佳
        elevation_reward = max(0, 1.0 - elevation / 15)  # 15度内最佳

        return (bearing_reward + elevation_reward) * 0.1

    def _calculate_energy_reward(self, env_data: Dict) -> float:
        """能量奖励 - 保持速度和高度优势"""
        ownship = env_data.get("ownship", {})
        enemy = env_data.get("enemies", [{}])[0] if env_data.get("enemies") else {}

        if not enemy:
            return 0.0

        # 速度优势
        velocity_advantage = ownship.get("velocity", 0) - enemy.get("velocity", 0)
        velocity_reward = velocity_advantage / 100  # 每100m/s优势给0.1奖励

        # 高度优势
        altitude_advantage = ownship.get("altitude", 0) - enemy.get("altitude", 0)
        altitude_reward = altitude_advantage / 1000  # 每1000米优势给0.1奖励

        return (velocity_reward + altitude_reward) * 0.1

    def _calculate_action_penalty(self, action: Dict) -> float:
        """动作惩罚 - 惩罚剧烈操作"""
        penalty = 0.0
        # 惩罚剧烈俯仰和滚转
        if abs(action.get("pitch", 0)) > 0.8:
            penalty -= 0.02
        if abs(action.get("roll", 0)) > 0.8:
            penalty -= 0.02
        return penalty

    def _calculate_combat_reward(self, env_data: Dict) -> float:
        """战斗结果奖励"""
        combat_results = env_data.get("combat_results", {})
        if combat_results.get("hit", False):
            return 5.0  # 命中奖励
        elif combat_results.get("kill", False):
            return 20.0  # 击毁奖励
        return 0.0