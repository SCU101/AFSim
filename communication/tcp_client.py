import socket
import json
import struct
import time


class SimulationClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.scenario = None
        self.target_ids = None
        self.init_actions = None  # 升降舵、副翼、方向舵、油门
        # 油门范围是[0,1]，其他范围是[-1,1]

    def connection(self, scenario):
        """单次通信仿真步长是16ms"""
        try:
            self.scenario = scenario
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            resp = self.send_request("init", {"count": 1, "scenario": scenario})
            if resp.get("status") != "ok":
                print(f"Init 失败: {resp.get('msg')}")
                return
            if scenario == "testWzz":
                self.target_ids = ["1001"]
                self.init_actions = [[0.5, 0.0, 0.0, 1.0]]  # 升降舵、副翼、方向舵、油门
            else:
                self.target_ids = []
                self.init_actions = [[0.5, 0.0, 0.0, 1.0]]  # 升降舵、副翼、方向舵、油门
            for target_id in self.target_ids:
                is_ready, _ = self.wait_for_platform_ready(target_id)
                if not is_ready:
                    return
            return resp

        except ConnectionRefusedError:
            print(f"\n❌ 连接失败！请检查 C++ Server 是否正在监听端口 {self.port}")
        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_environment_data(self, actions):
        step_params = {
            "actions": {"0": {"objID": self.target_ids[0], "vals": actions[0]}}
        }
        resp = self.send_request("step", step_params)
        return resp

    def reset(self):
        try:
            self.send_request("reset", {"env_ids": [0]})
        except Exception as e:
            import traceback
            traceback.print_exc()

    def close(self):
        try:
            self.send_request("close", {"env_ids": [0]})
            self.socket.close()
            print("\n🔌 连接已关闭")
        except Exception as e:
            import traceback
            traceback.print_exc()

    def send_request(self, command, params):
        """封装好的发送函数"""
        req_id = f"{command}_{int(time.time())}"
        payload = {
            "req_id": req_id, "cmd": command, "params": params
        }
        json_str = json.dumps(payload)
        body_bytes = json_str.encode('utf-8')
        header = struct.pack('<I', len(body_bytes))
        self.socket.sendall(header + body_bytes)

        header_recv = self.socket.recv(4)
        if not header_recv:
            raise ConnectionError("Connection closed")
        body_len = struct.unpack('<I', header_recv)[0]

        body_recv = b""
        while len(body_recv) < body_len:
            packet = self.socket.recv(body_len - len(body_recv))
            if not packet:
                break
            body_recv += packet

        return json.loads(body_recv)

    def wait_for_platform_ready(self, target_id, timeout=10):
        """轮询等待飞机上线"""
        print(f"🕵️‍♂️ 正在轮询等待飞机 [{target_id}] 上线 (超时: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                resp = self.send_request("reset", {"env_ids": [0]})
                if resp.get("status") == "ok":
                    obs = resp["data"]["0"]["obs"]
                    platforms = obs.get("platforms", [])
                    # 遍历查找目标飞机 (不依赖 ID，只看 name)
                    for p in platforms:
                        if p.get("name") == target_id:
                            print(f"✅ 成功捕获目标！飞机 [{target_id}] 已就绪。")
                            return True, obs
                    if not platforms:
                        print(f"   ... AFSIM 正在加载模型 ...")
            except Exception as e:
                print(f"   轮询错误: {e}")
            time.sleep(0.5)
        print(f"❌ 等待超时！")
        return False, None


if __name__ == "__main__":
    simulation = SimulationClient(host='127.0.0.1', port=8888)
    observation = simulation.connection(scenario="testWzz")
    for i in range(100):
        observation = simulation.get_environment_data([[0.5, 0.0, 0.0, 1.0]])
    simulation.reset()
    observation = simulation.get_environment_data([[0.5, 0.0, 0.0, 1.0]])
    simulation.close()
