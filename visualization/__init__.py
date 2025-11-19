"""
TacView 数据发送模块
负责与TacView实时数据记录软件建立连接并发送仿真数据
"""

from visualization.tacview_handler import TacView
__all__ = [
    'TacViewHandler',
    'get_version'
]
__version__ = "1.0.0"
__author__ = ""
__description__ = "TacView仿真数据发送模块"
# 包级别配置
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 42674
PROTOCOL_VERSION = "1.0"
def create_tacview(host=None, port=None):
    """
    创建TacView处理器实例的便捷函数
    
    Args:
        host: TacView监听地址
        port: TacView监听端口
        protocol_version: 协议版本
    
    Returns:
        TacViewHandler: 配置好的处理器实例
    """
    host = host or DEFAULT_HOST
    port = port or DEFAULT_PORT

    
    return TacView(host, port)

def get_version():
    """获取模块版本信息"""
    return {
        "version": __version__,
        "protocol": PROTOCOL_VERSION,
        "description": __description__
    }

# 包导入时的初始化
print(f"Initialized TacView Sender Module v{__version__}")