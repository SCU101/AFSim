import socket
import json
import struct
import time

# ================= 配置区域 =================
HOST = '127.0.0.1'  # C++ Server IP
PORT = 8888         # ✅ 修正：根据你的最新指示，使用 8888 端口
SCENARIO_NAME = "testWzz" # ⚠️ 注意大小写，需与文件名一致
TARGET_ID = "1001"  # 🎯 目标飞机 ID
MAX_WAIT_SEC = 10   # 最大等待加载时间 (秒)
# ===========================================

def send_request(sock, command, params):
    """封装好的发送函数"""
    req_id = f"{command}_{int(time.time())}"
    payload = {
        "req_id": req_id, "cmd": command, "params": params
    }
    json_str = json.dumps(payload)
    body_bytes = json_str.encode('utf-8')
    header = struct.pack('<I', len(body_bytes))
    sock.sendall(header + body_bytes)
    
    header_recv = sock.recv(4)
    if not header_recv: raise ConnectionError("Connection closed")
    body_len = struct.unpack('<I', header_recv)[0]
    
    body_recv = b""
    while len(body_recv) < body_len:
        packet = sock.recv(body_len - len(body_recv))
        if not packet: break
        body_recv += packet
        
    return json.loads(body_recv)

def wait_for_platform_ready(client, target_id, timeout=10):
    """轮询等待飞机上线"""
    print(f"🕵️‍♂️ 正在轮询等待飞机 [{target_id}] 上线 (超时: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            resp = send_request(client, "reset", {"env_ids": [0]})
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

def find_plane_state(obs, plane_id):
    """根据 name 查找飞机状态"""
    for p in obs.get("platforms", []):
        if p.get("name") == plane_id: return p
    return None

def main():
    print(f"🔌 连接服务器 {HOST}:{PORT}...")
    client = None
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((HOST, PORT))
        print("✅ TCP 连接建立成功！")

        # --- 1. INIT ---
        print(f"\n[1] 执行 Init ({SCENARIO_NAME})...")
        resp = send_request(client, "init", {"count": 1, "scenario": SCENARIO_NAME})
        if resp.get("status") != "ok":
            print(f"Init 失败: {resp.get('msg')}")
            return

        # --- 2. 轮询等待 ---
        is_ready, _ = wait_for_platform_ready(client, TARGET_ID, MAX_WAIT_SEC)
        if not is_ready: return

        # --- 3. STEP 循环 (50帧) ---
        print("\n[3] 开始 50 次 Step 循环 (拉起 + 满油门) ...")
        
        # 动作：拉杆 0.5，满油门 1.0
        action_vals = [0.5, 0.0, 0.0, 1.0] 

        for i in range(1, 51):
            step_params = {
                "steps": 1,
                "actions": { "0": { TARGET_ID: action_vals } }
            }

            start_t = time.time()
            resp = send_request(client, "step", step_params)
            cost_ms = (time.time() - start_t) * 1000

            if resp.get("status") == "ok":
                obs = resp["data"]["0"]["obs"]
                plane = find_plane_state(obs, TARGET_ID)
                
                if plane:
                    # === 打印核心飞行参数 ===
                    print(f"\n🚀 Frame {i:02d} | Time: {obs['sim_time']:.2f}s | Cost: {cost_ms:.1f}ms")
                    
                    # 1. 位置 (Lat/Lon/Alt)
                    print(f"   [位置] Lat: {plane['lat']:.6f}  Lon: {plane['lon']:.6f}  Alt: {plane['alt']:.2f} m")
                    
                    # 2. 姿态 (Heading/Pitch/Roll)
                    if 'heading' in plane:
                        print(f"   [姿态] Hdg: {plane['heading']:.2f}°  Pitch: {plane['pitch']:.2f}°  Roll: {plane['roll']:.2f}°")
                    
                    # 3. 速度 (Speed + NED分量)
                    if 'speed' in plane:
                        print(f"   [速度] Spd: {plane['speed']:.1f} m/s  (Vx:{plane['vx']:.1f}, Vy:{plane['vy']:.1f}, Vz:{plane['vz']:.1f})")
                    
                    # 4. 质量/油量 (Mass)
                    if 'mass' in plane:
                        print(f"   [质量] Mass: {plane['mass']:.1f} kg (油耗监控)")

                else:
                    print(f"⚠️ 警告: Step 成功但未找到飞机 [{TARGET_ID}]")
            else:
                print(f"❌ Step 报错: {resp.get('msg')}")
                break
            
            # 0.1秒一帧，方便观察
            time.sleep(0.1)

        # --- 4. 结束清理 ---
        print("\n[4] 测试结束，发送 close 指令...")
        send_request(client, "close", {"env_ids": [0]})

    except ConnectionRefusedError:
        print(f"\n❌ 连接失败！请检查 C++ Server 是否正在监听端口 {PORT}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if client: client.close()
        print("\n🔌 连接已关闭")

if __name__ == "__main__":
    main()