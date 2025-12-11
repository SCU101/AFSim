import socket
import json
import time

# --- 配置 ---
# 服务器IP地址，请修改为对应的地址
HOST = '192.168.43.173'
# 服务器监听的端口，请修改为对应的端口
PORT = 8888

class SimClient:
    def __init__(self):
        self.sock = None
        self.req_counter = 0

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((HOST, PORT))
            print(f"✅ [Py] 已连接服务器 {HOST}:{PORT}")
            return True
        except Exception as e:
            print(f"❌ [Py] 连接失败: {e}")
            print("   -> 请检查 SimLauncher.exe 是否已运行")
            return False

    def close(self):
        if self.sock:
            self.sock.close()
            print("✅ [Py] 断开连接")

    def recv_and_parse(self):
        """
        接收服务器返回数据并解析（简化版本）
        """
        try:
            resp_bytes = self.sock.recv(8192)
            if not resp_bytes:
                print("❌ [Py] 未收到服务器数据或服务器断开")
                return None

            resp_str = resp_bytes.decode("utf-8").strip()

            # 尝试 JSON 解析
            try:
                resp = json.loads(resp_str)
            except json.JSONDecodeError:
                print("❌ [Py] 回包不是合法的 JSON：")
                print(f"    原始数据: {resp_str}")
                return None

            # 简易格式检测
            if not isinstance(resp, dict):
                print("❌ [Py] 回包 JSON 不是对象类型: ")
                print(resp)
                return None

            # 格式正常 -> 打印结构
            print(f"📥 [Py] 收到回包：")
            print(f"    - keys: {list(resp.keys())}")

            if "status" in resp:
                print(f"    - status: {resp['status']}")

            if "msg" in resp:
                print(f"    - msg: {resp['msg']}")

            if "data" in resp:
                if isinstance(resp["data"], dict):
                    print(f"    - data keys: {list(resp['data'].keys())}")
                else:
                    print(f"    - data: (非 dict 类型)")

            return resp

        except Exception as e:
            print(f"❌ [Py] 接收或解析异常: {e}")
            return None

    def send_cmd(self, cmd, params={}):
        """
        核心发送函数：负责封装协议头、序列化、发送、接收、反序列化
        """
        self.req_counter += 1
        req_id = str(self.req_counter)

        # 1. 构造请求包
        payload = {
            "cmd": cmd,
            "req_id": req_id,
            "params": params
        }

        json_str = json.dumps(payload)
        print(f"\n🚀 [Py] 发送 ({cmd}) ID={req_id} >>>")
        # print(f"    {json_str}") # 打开这行看原始数据

        try:
            # 2. 发送
            self.sock.sendall(json_str.encode('utf-8'))

            # 3. 接收 (简单起见读 4096 字节，实际项目可能需要循环读)
            # resp_bytes = self.sock.recv(8192)
            # if not resp_bytes:
            #     print("❌ [Py] 服务器意外断开")
            #     return None
            #
            # resp_str = resp_bytes.decode('utf-8')
            # resp = json.loads(resp_str)
            #
            # # 4. 打印回包
            # if resp["status"] == "ok":
            #     print(f"🟢 [Py] 成功 (ID={resp['req_id']})")
            #     if "msg" in resp:
            #         print(f"    Msg: {resp['msg']}")
            #     if "data" in resp and resp["data"]:
            #         # 简单打印 data 的 keys，防止刷屏
            #         print(f"    Data Keys: {list(resp['data'].keys())}")
            # else:
            #     print(f"🔴 [Py] 失败: {resp.get('msg', 'Unknown Error')}")
            #
            # return resp
            resp = self.recv_and_parse()

            return resp

        except Exception as e:
            print(f"❌ [Py] 通信异常: {e}")
            return None

# ==========================================
# 主测试流程
# ==========================================
def main():
    client = SimClient()
    if not client.connect():
        return

    # --- 1. 测试 Init ---
    print("\n=== 测试 1: 初始化 ===")
    client.send_cmd("init", {
        "count": 2,
        "scenario": "BVR_Combat"
    })

    # --- 2. 测试 Pause/Resume ---
    print("\n=== 测试 2: 暂停与继续 ===")
    client.send_cmd("pause", {"state": True})  # 暂停
    time.sleep(0.5)
    client.send_cmd("pause", {"state": False}) # 继续

    # --- 3. 测试 Reset (带自定义位置) ---
    print("\n=== 测试 3: 重置环境 ===")
    client.send_cmd("reset", {
        "env_ids": [0, 1],
        "custom_states": {
            "0": {
                "lon": 120.5, "lat": 24.0, "alt": 8000.0,
                "hp": 100, "fuel": 1000
            }
            # 环境 1 使用默认
        }
    })

    # --- 4. 测试 Step (循环 3 次) ---
    print("\n=== 测试 4: 物理步进 (3次) ===")
    for i in range(3):
        client.send_cmd("step", {
            "actions": {
                "0": {"throttle": 1.0, "pitch": 0.5},
                "1": {"throttle": 0.0, "pitch": 0.0}
            }
        })
        time.sleep(0.2)

    # --- 5. 测试 Close ---
    print("\n=== 测试 5: 关闭服务器 ===")
    client.send_cmd("close", {})

    client.close()
    print("\n✨ 所有测试完成！")

if __name__ == "__main__":
    main()