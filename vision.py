import cv2
from fer.fer import FER
import time


def detect_emotion_from_camera():
    """实时摄像头表情识别测试"""
    detector = FER(mtcnn=True)  # mtcnn 更准确，但稍慢
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

    print("按 'q' 键退出摄像头表情识别...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FER 检测表情
        emotions = detector.detect_emotions(frame)

        if emotions:
            for emotion in emotions:
                box = emotion["box"]  # 人脸位置
                emo_dict = emotion["emotions"]
                dominant_emotion = max(emo_dict, key=emo_dict.get)
                score = emo_dict[dominant_emotion]

                # 画框和文字
                cv2.rectangle(frame, (box[0], box[1]),
                              (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{dominant_emotion}: {score:.2f}",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("FER - Facial Expression Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)  # 控制帧率

    cap.release()
    cv2.destroyAllWindows()


# 测试运行
if __name__ == "__main__":
    detect_emotion_from_camera()