# src/utils/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name="emotion_system", log_file="logs/system.log"):
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 文件处理器
    fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

logger = setup_logger()
