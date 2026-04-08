# vision.py
import cv2
from fer.fer import FER

# 全局初始化一次，避免重复加载模型
detector = FER(mtcnn=True)


def process_image(frame) -> str:
    """
    接收 Gradio 传入的图像帧（numpy array），返回主导表情
    参数:
        frame: Gradio 传来的图像（BGR 格式的 numpy array）
    返回:
        'happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust' 之一
    """
    if frame is None:
        return "neutral"

    try:
        # FER 直接分析传入的帧
        emotions = detector.detect_emotions(frame)

        if not emotions:
            return "neutral"

        # 取第一个人脸的主导表情
        emo_dict = emotions[0]["emotions"]
        dominant = max(emo_dict, key=emo_dict.get)
        return dominant

    except Exception as e:
        print(f"表情识别出错: {e}")
        return "neutral"


# ========== 保留原函数（备用/本地测试）==========
def get_current_emotion() -> str:
    """
    自己打开摄像头检测（仅用于本地测试，Gradio 不要用这个）
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "neutral"

    return process_image(frame)  # 复用上面的处理逻辑


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 本地测试：自己打开摄像头
    print("测试表情识别（按 Ctrl+C 退出）")
    try:
        while True:
            emotion = get_current_emotion()
            print(f"当前表情: {emotion}", end="\r")
    except KeyboardInterrupt:
        print("\n测试结束")