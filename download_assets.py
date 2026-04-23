# download_assets.py
import os
import requests

# 情绪与音乐链接映射
ASSETS = {
    "happy.mp3": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", # 测试用稳定链接
    "sad.mp3": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
    "thinking.mp3": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"
}

MUSIC_DIR = "music"

def download_music():
    if not os.path.exists(MUSIC_DIR):
        os.makedirs(MUSIC_DIR)
        print(f"Created directory: {MUSIC_DIR}")

    for filename, url in ASSETS.items():
        path = os.path.join(MUSIC_DIR, filename)
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Success: {path}")
            else:
                print(f"Failed {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error {filename}: {e}")

if __name__ == "__main__":
    download_music()
    print("\n--- Done ---")
