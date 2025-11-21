import socket
from log_handler import setup_logger
class TacView(object):
    """ TacView 实时数据传输处理类 
    arguments:
        host: 服务器主机地址，默认本地地址 '
        port: 服务器端口，默认42674
    """
    def __init__(self, host:str='127.0.0.1', port:int=42674):
        self.host = host
        self.port = port
        self.logger = setup_logger()
        self.setup_server()
        

       
    def setup_server(self):
        """ 设置服务器套接字并开始监听连接"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.logger.info(f"Server listening on {self.host}:{self.port}")
            self.logger.info("IMPORTANT: Please open Tacview Advanced, click Record -> Real-time Telemetry, and input the IP address and port !")
            self.connect()
        except Exception as e:
            self.logger.error(f"Setup error: {e}")
            self.cleanup()
            raise

    def send_data_to_client(self, data):

        try:
            self.client_socket.send(data)
        except Exception as e:
            self.logger.error(f"Send error: {e}")
            self.reconnect()
    
    def connect(self):
        """ 等待客户端连接并进行握手 """
        try:
            self.logger.info("Waiting for connection...")
            self.client_socket, self.address = self.server_socket.accept()
            self.logger.info(f"Accepted connection from {self.address}")
            
            # 发送握手数据
            handshake_data = f"XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\n{socket.gethostname()}\n\x00"
            self.client_socket.send(handshake_data.encode())
            
            # 接收客户端响应
            data = self.client_socket.recv(1024)
            self.logger.info(f"Received data from {self.address}: {data.decode()}")
            
            # 发送头部数据
            header_data = ("FileType=text/acmi/tacview\nFileVersion=2.2\n"
                          "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n")
            self.client_socket.send(header_data.encode())
            self.logger.info("Connection established")
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self.cleanup()
            raise

    def reconnect(self):
        """ 尝试重新连接客户端 """
        self.logger.info("Attempting to reconnect...")
        self.cleanup()
        self.setup_server()

    def cleanup(self):
        try:
            if hasattr(self, 'client_socket') and self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            if hasattr(self, 'server_socket') and self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def __del__(self):
        self.cleanup()

