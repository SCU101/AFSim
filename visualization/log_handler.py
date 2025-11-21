import logging
import os
from datetime import datetime
import sys
def setup_logger():
    """
    设置项目日志系统
    """
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'log')
    os.makedirs(log_dir, exist_ok=True)
    logfile_name='afsim_tacview_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.log'
    # 日志文件路径
    log_file = os.path.join(log_dir, logfile_name)
    
    # 创建logger
    logger = logging.getLogger('AFSim_Tacview')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 添加handler到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger has been set up successfully.")