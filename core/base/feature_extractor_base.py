from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class FeatureExtractorBase(ABC):
    """特征提取器基类"""

    def __init__(self):
        self.feature_dim = 0

    @abstractmethod
    def extract(self, env_data: Dict[str, Any]) -> np.ndarray:
        """从环境数据中提取特征"""
        pass

    @abstractmethod
    def get_feature_dimension(self) -> int:
        """返回特征维度"""
        pass

    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """归一化数值到[-1, 1]范围"""
        return 2 * (value - min_val) / (max_val - min_val) - 1