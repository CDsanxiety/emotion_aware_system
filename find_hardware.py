import cv2
import pyaudio

def test_cameras():
    print("--- 正在检测摄像头 ---")
    available_cams = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ 找到可用摄像头，索引: {i}")
                available_cams.append(i)
            cap.release()
    if not available_cams:
        print("❌ 没有找到任何能输出画面的摄像头！请检查 USB 连接。")
    return available_cams

def test_microphones():
    print("\n--- 正在检测麦克风 ---")
    p = pyaudio.PyAudio()
    available_mics = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # 检查是否有输入通道
        if info['maxInputChannels'] > 0:
            print(f"麦克风选项 [索引 {i}]: {info['name']} (支持最大采样率: {info['defaultSampleRate']}Hz)")
            available_mics.append(i)
    p.terminate()
    if not available_mics:
        print("❌ 没有找到任何麦克风！请检查 USB 连接。")
    return available_mics

if __name__ == "__main__":
    test_cameras()
    test_microphones()
