import requests
import json
from typing import Dict, Any, Optional


class SimulationClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def connect(self, env_name: str) -> bool:
        """连接仿真服务端并选择环境"""
        payload = {"environment": env_name}
        try:
            response = self.session.post(
                f"{self.base_url}/connect",
                json=payload,
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def get_environment_data(self) -> Optional[Dict[str, Any]]:
        """从仿真服务端获取环境数据"""
        try:
            response = self.session.get(
                f"{self.base_url}/environment",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"获取环境数据失败: {e}")
        return None

    def send_action(self, action: Dict[str, Any]) -> bool:
        """发送动作到仿真服务端"""
        try:
            response = self.session.post(
                f"{self.base_url}/action",
                json=action,
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"发送动作失败: {e}")
            return False

    def reset_environment(self) -> bool:
        """重置环境"""
        try:
            response = self.session.post(
                f"{self.base_url}/reset",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"重置环境失败: {e}")
            return False