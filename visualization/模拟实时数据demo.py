import time
import os
# 构建文件的完整路径
from tacview_handler import TacView
tacview=TacView()

script_dir = os.path.dirname(os.path.abspath(__file__))

# 该acmi进行了处理 将acmi文件头：FileType=text/acmi/tacviewFileVersion=2.20,ReferenceTime=2025-11-12T14:48:44Z 剔除，只传输数据部分
# acmi游戏头在 tacview_handler中与 tacview建立连接时传输，详细可见tacview_handler.py中connect函数
# 开发者 只需调用 tacview.send_data_to_client(data.encode()) 方法传输数据即可
acmi_file_path = os.path.join(script_dir, 'F-16_2025-11-12_14-48-44_group1_episode50_acmi.acmi')
with open(acmi_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tacview.send_data_to_client(line.encode())
            print(line.encode())
            # 每 0.1s 发送一行数据
            time.sleep(0.1)