# src/hardware/drivers/music_driver.py
import time
import json
import os
import subprocess
import roslibpy
from src.core.config import ROS_BRIDGE_URI, TOPIC_ACTION, AUDIO_OUTPUT_DEVICE
from src.utils.logger import logger

def on_action_received(msg_data):
    try:
        msg = json.loads(msg_data['data'])
        execution = msg.get("execution", {})
        action = execution.get("action") or "none"

        # 映射动作到音乐文件
        music_map = {
            "music_happy": "music/happy.mp3",
            "music_calm": "music/calm.mp3",
            "test": "music/test.mp3"
        }
        
        target_file = music_map.get(action, "music/test.mp3")
        
        if not os.path.exists(target_file):
            target_file = "music/test.mp3" # 再次兜底
            
        if os.path.exists(target_file):
            logger.info(f"[Music Driver] 正在播放: {target_file}")
            subprocess.Popen(["mpg123", "-a", AUDIO_OUTPUT_DEVICE, "-q", target_file])
            else:
                logger.warning(f"[Music Driver] 找不到文件: {file_path}")
    except Exception as e:
        logger.error(f"[Music Driver] 错误: {e}")

def main():
    logger.info("--- 音乐播放驱动 (ROS版) 启动 ---")
    host_port = ROS_BRIDGE_URI.replace("ws://", "").split(":")
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 9090
    
    client = roslibpy.Ros(host=host, port=port)
    listener = roslibpy.Topic(client, TOPIC_ACTION, 'std_msgs/String')
    
    client.on('ready', lambda: logger.info("[Music Driver] 已连接到 ROSBridge"))
    listener.subscribe(on_action_received)
    
    client.run_forever()

if __name__ == "__main__":
    main()
