import socket
import json
import time
import struct


# --- 配置 ---
# 服务器IP地址，请修改为对应的地址
HOST = '192.168.43.173'
# 服务器监听的端口，请修改为对应的端口
PORT = 8888

class SimClient:
    """
    通信客户端类，功能有建立连接、断开连接、发送命令（暂停、继续、动作命令）、接受服务器返回并解析
    """

    def __init__(self):
        self.sock = None
        self.req_counter = 0  # 请求序号

    def connect(self):
        """
        与服务器建立连接的功能，返回值为布尔值，代表是否成功建立连接
        :return: 布尔值，代表是否成功建立连接
        """

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((HOST, PORT))
            print(f"√ [Py] 已连接服务器 {HOST}:{PORT}")
            return True
        except Exception as e:
            print(f"× [Py] 连接失败: {e}")
            print("   -> 请检查 SimLauncher.exe 是否已运行")
            return False

    def close(self):
        """
        与服务器断开连接的功能
        """
        if self.sock:
            self.sock.close()
            print("√ [Py] 断开连接")

    def recv_and_parse(self):
        """
        接收服务器返回数据并解析，目前仅打印
        :return: None 出现异常/resp 正常返回的数据
        """

        try:
            # 接收 4 字节包头
            header_bytes = self.sock.recv(4)
            if not header_bytes or len(header_bytes) < 4:
                print("× [Py] 未收到完整包头")
                return None

            # 解析包体长度
            body_len = struct.unpack("!I", header_bytes)[0]
            print(f"√ [Py] 收到包头，包体长度 = {body_len} 字节")

            # 2. 按长度接收包体
            resp_bytes = b""
            while len(resp_bytes) < body_len:
                chunk = self.sock.recv(body_len - len(resp_bytes))
                if not chunk:
                    print("× [Py] 接收包体过程中连接断开")
                    return None
                resp_bytes += chunk

            # if not resp_bytes:
            #     print("× [Py] 未收到服务器数据或服务器断开")
            #     return None

            resp_str = resp_bytes.decode("utf-8").strip()

            # 尝试 JSON 解析
            try:
                resp = json.loads(resp_str)
            except json.JSONDecodeError:
                print("× [Py] 回包不是合法的 JSON：")
                print(f"    原始数据: {resp_str}")
                return None

            # 格式检测
            if not isinstance(resp, dict):
                print("× [Py] 回包 JSON 不是对象类型: ")
                print(resp)
                return None

            # 格式正常 -> 打印结构
            print(f"√ [Py] 收到回包：")
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
            print(f"× [Py] 接收或解析异常: {e}")
            return None

    def send_cmd(self, cmd_type, params={}):
        """
        核心发送函数：负责封装协议头、序列化、发送、接收、反序列化
        cmd_type: 命令类型，例如step步进/reset重置环境/pause暂停等
        params: json格式的具体命令
        """

        self.req_counter += 1
        req_id = str(self.req_counter)

        # 构造请求包
        payload = {
            "cmd_type": cmd_type,
            "req_id": req_id,
            "params": params
        }

        json_str = json.dumps(payload)
        print(f"\n[Py] 发送 ({cmd_type}) ID={req_id} >>>")
        # print(f"    {json_str}") # 打开这行看原始数据

        try:
            # 发送
            body_bytes = json_str.encode("utf-8")
            body_len = len(body_bytes)

            # 32bit 无符号整型，网络字节序
            header = struct.pack("I", body_len)

            # 先发头，再发数据
            self.sock.sendall(header + body_bytes)

            # 接收返回并解析
            resp = self.recv_and_parse()

            return resp

        except Exception as e:
            print(f"× [Py] 通信异常: {e}")
            return None

# ==========================================
# 主测试流程
# ==========================================
def main():
    """
    主测试流程
    :return:
    """
    # 初始化一个客户端通信实例
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
    # 客户端先发送关闭请求，之后客户端关闭连接
    print("\n=== 测试 5: 关闭服务器 ===")
    client.send_cmd("close", {})

    client.close()
    print("\n√ 所有测试完成！")

if __name__ == "__main__":
    main()
