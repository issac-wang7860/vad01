import os
from yt_dlp import YoutubeDL, DownloadError
from filelock import FileLock

def get_local_or_stream(url: str, tmp_dir: str, prefer_stream: bool = True):
    """
    优先下载，失败则回退到流式播放，支持tiktok/douyin等。
    返回本地文件路径或{'stream': url, 'fps': fps}
    """
    os.makedirs(tmp_dir, exist_ok=True)
    lock_path = os.path.join(tmp_dir, 'yt_dlp.lock')
    out_path = os.path.join(tmp_dir, 'video.mp4')
    with FileLock(lock_path):
        ydl_opts = {
            'outtmpl': out_path,
            'quiet': True,
            'noplaylist': True,
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if os.path.exists(out_path):
                return out_path
        except DownloadError:
            try:
                with YoutubeDL({'skip_download': True, 'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                best = info['url']
                proto = info.get('protocol')
                fps = info.get('fps', 30)
                if prefer_stream or proto in ('m3u8_native','m3u8','https'):
                    return {'stream': best, 'fps': fps}
                else:
                    raise
            except Exception as e:
                print(f"[stream_or_download] Failed: {e}")
                raise 