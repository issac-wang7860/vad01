import subprocess
import os

def extract_audio(video_path, wav_path):
    """用ffmpeg从视频中提取音频为wav文件。"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"[extract_audio] Failed: {e}")
        return False 