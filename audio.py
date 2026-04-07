# vision.py - 表情识别模块
import cv2
from fer.fer import FER

# 初始化检测器（全局，只加载一次）
detector = FER(mtcnn=True)


def get_current_emotion() -> str:
    """
    实时从摄像头获取当前表情
    返回值示例: 'happy', 'sad', 'angry', 'neutral', 'surprise' 等
    如果未检测到人脸，返回 'neutral'
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ 无法打开摄像头")
        return "neutral"

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "neutral"

    # 检测表情
    emotions = detector.detect_emotions(frame)

    if not emotions:
        return "neutral"

    # 获取主导表情
    emo_dict = emotions[0]["emotions"]
    dominant_emotion = max(emo_dict, key=emo_dict.get)

    return dominant_emotion


# 测试函数（直接运行本文件时执行）
if __name__ == "__main__":
    print("正在检测表情... 请看向摄像头")
    emotion = get_current_emotion()
    print(f"当前检测到的表情: {emotion}")