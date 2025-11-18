import numpy as np
from typing import Dict, Any

from core.base.feature_extractor_base import FeatureExtractorBase
from utils.math_functions import calculate_distance, calculate_angles


class BasicCombatFeatureExtractor(FeatureExtractorBase):
    """基础空战特征提取器"""

    def __init__(self):
        super().__init__()
        self.feature_dim = 15  # 基础空战特征维度

    def extract(self, env_data: Dict[str, Any]) -> np.ndarray:
        """提取基础空战特征"""
        features = []

        # 提取己方状态
        ownship = env_data.get("ownship", {})
        features.extend([
            self._normalize_value(ownship.get("velocity", 0), 0, 500),
            self._normalize_value(ownship.get("altitude", 0), 0, 15000),
            self._normalize_value(ownship.get("heading", 0), 0, 360),
            self._normalize_value(ownship.get("pitch", 0), -90, 90),
            self._normalize_value(ownship.get("roll", 0), -180, 180),
            ownship.get("fuel_remaining", 0) / ownship.get("max_fuel", 1)
        ])

        # 提取敌方相对状态
        enemy = env_data.get("enemies", [{}])[0] if env_data.get("enemies") else {}
        if enemy:
            rel_features = self._extract_relative_features(ownship, enemy)
            features.extend(rel_features)
        else:
            # 无敌人时的默认值
            features.extend([0, 0, 0, 0, 0])

        # 武器状态
        weapons = env_data.get("weapons", {})
        features.extend([
            weapons.get("missiles_remaining", 0) / 4,  # 假设最大4枚
            weapons.get("gun_ammo", 0) / 500  # 假设最大500发
        ])

        # 健康状态
        damage = env_data.get("damage", {})
        features.append(1.0 - damage.get("total_damage", 0))

        return np.array(features, dtype=np.float32)

    def _extract_relative_features(self, ownship: Dict, enemy: Dict) -> list:
        """提取相对特征"""
        own_pos = ownship.get("position", {})
        enemy_pos = enemy.get("position", {})

        # 计算相对距离
        distance = calculate_distance(own_pos, enemy_pos)
        norm_distance = self._normalize_value(distance, 0, 50000)

        # 计算相对角度
        angles = calculate_angles(ownship, enemy)
        norm_bearing = self._normalize_value(angles.get("bearing", 0), -180, 180)
        norm_elevation = self._normalize_value(angles.get("elevation", 0), -90, 90)

        # 相对速度
        rel_velocity = enemy.get("velocity", 0) - ownship.get("velocity", 0)
        norm_rel_velocity = self._normalize_value(rel_velocity, -500, 500)

        # 是否在武器射程内
        in_weapon_range = 1.0 if distance < 10000 else 0.0

        return [norm_distance, norm_bearing, norm_elevation, norm_rel_velocity, in_weapon_range]

    def get_feature_dimension(self) -> int:
        return self.feature_dim
