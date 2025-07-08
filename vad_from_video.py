import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import mediapipe as mp
import cv2
import tempfile
from utils.stream_or_download import get_local_or_stream

# ========== EmoNet 相关 =============
class EmoNetWrapper:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = self._load_emonet()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def _download_emonet_weights(self, url, out_path):
        import requests
        if not os.path.exists(out_path):
            print(f"Downloading EmoNet weights to {out_path}...")
            r = requests.get(url, allow_redirects=True)
            with open(out_path, 'wb') as f:
                f.write(r.content)

    def _load_emonet(self):
        # EmoNet repo: https://github.com/face-analysis/emonet
        # 权重下载地址（官方发布的 .pth）
        url = 'https://github.com/face-analysis/emonet/raw/master/pretrained/emonet_8.pth'
        weight_path = 'emonet_8.pth'
        self._download_emonet_weights(url, weight_path)
        # EmoNet 是 ResNet18 backbone，输出 valence/arousal
        class EmoNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
                self.backbone.fc = nn.Linear(512, 2)  # 2: valence, arousal
            def forward(self, x):
                return self.backbone(x)
        model = EmoNet()
        state = torch.load(weight_path, map_location=self.device)
        # 兼容性处理
        if 'state_dict' in state:
            state = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(self.device)
        return model

    def detect_face(self, frame):
        # mediapipe 检测人脸，返回最大人脸的 bbox
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            # 取最大人脸
            det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            # 边界修正
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return frame[y1:y2, x1:x2]
        return None

    def predict(self, frame):
        face = self.detect_face(frame)
        if face is None or face.size == 0:
            return np.nan, np.nan
        img = self.transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img)
            valence, arousal = out[0].cpu().numpy()
        return float(valence), float(arousal)

# ========== 其余代码保持不变 =============
# 1. extract_frames(video_path, fps=5) → yield (timestamp_s, frame_bgr)
def extract_frames(video_path, fps=5):
    """Yield (timestamp_s, frame_bgr) at given fps from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(video_fps / fps))
    frame_idx = 0
    sampled = 0
    # 计算采样帧数
    if step > 0:
        num_samples = total_frames // step
    else:
        num_samples = total_frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / video_fps
            yield (timestamp, frame)
            sampled += 1
        frame_idx += 1
    cap.release()

# 2. estimate_va(frame) → (valence, arousal)
def estimate_va(frame):
    """Stub: Estimate (valence, arousal) from frame using EmoNet. To be implemented."""
    # TODO: Replace with actual EmoNet inference
    return np.nan, np.nan

# 3. Discrete emotion to D mapping
discrete_to_D = {'happy':0.8,'surprise':0.4,'neutral':0.0,'sad':-0.5,'fear':-0.2,'disgust':-0.4,'anger':-0.6}

def estimate_D(valence, arousal, emotion=None):
    if emotion in discrete_to_D:
        return discrete_to_D[emotion]
    if np.isnan(valence) or np.isnan(arousal):
        return np.nan
    return 0.5 * valence + 0.5 * arousal * np.sign(valence)

# 4. Exponential smoothing
def smooth_series(series, alpha=0.3):
    return series.ewm(alpha=alpha).mean()

# 5. Plot 3D trajectory
def plot_3d(csv_path, out_path='plot.png'):
    df = pd.read_csv(csv_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['V'], df['A'], df['D'], label='V-A-D trajectory')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.legend()
    plt.savefig(out_path)
    plt.close()

def plot_3d_animation(csv_path, out_gif_path='vad_trajectory.gif', out_mp4_path='vad_trajectory.mp4'):
    import matplotlib.animation as animation
    df = pd.read_csv(csv_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_xlim(df['V'].min(), df['V'].max())
    ax.set_ylim(df['A'].min(), df['A'].max())
    ax.set_zlim(df['D'].min(), df['D'].max())
    line, = ax.plot([], [], [], 'b-', label='V-A-D trajectory')
    point, = ax.plot([], [], [], 'ro')
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    def update(frame):
        x = df['V'][:frame+1]
        y = df['A'][:frame+1]
        z = df['D'][:frame+1]
        line.set_data(x, y)
        line.set_3d_properties(z)
        point.set_data([x.iloc[-1]], [y.iloc[-1]])
        point.set_3d_properties([z.iloc[-1]])
        return line, point
    ani = animation.FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=50)
    # 保存gif
    ani.save(out_gif_path, writer='pillow', fps=20)
    # 保存mp4
    try:
        ani.save(out_mp4_path, writer='ffmpeg', fps=20)
    except Exception as e:
        print(f'Warning: mp4 export failed: {e}')
    plt.close()

# 6. Main pipeline
try:
    from vad_multimodal import extract_audio, extract_audio_features, extract_body_bbox, load_mlp, estimate_D_mod, backup_estimate_D
    MULTIMODAL_OK = True
except ImportError:
    MULTIMODAL_OK = False
    print('[Warning] vad_multimodal not available, fallback to legacy D.')

def main(args):
    # 输出文件夹，按视频名和模式分子文件夹
    video_basename = os.path.splitext(os.path.basename(args.input))[0]
    mode = args.mode if hasattr(args, 'mode') else 'accurate'
    output_dir = os.path.join('outputs', f'{video_basename}_{mode}')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'results_{video_basename}_{mode}.csv')
    png_path = os.path.join(output_dir, f'plot_{video_basename}_{mode}.png')
    gif_path = os.path.join(output_dir, f'vad_trajectory_{video_basename}_{mode}.gif')
    mp4_path = os.path.join(output_dir, f'vad_trajectory_{video_basename}_{mode}.mp4')
    # 先统计采样帧数用于 tqdm
    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(video_fps / args.fps))
    if step > 0:
        num_samples = total_frames // step
    else:
        num_samples = total_frames
    cap.release()
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emonet = EmoNetWrapper(device=device)
    # 多模态Dominance准备
    if MULTIMODAL_OK:
        # 1. 提取音频特征
        wav_path = os.path.join(output_dir, f'{video_basename}.wav')
        extract_audio(args.input, wav_path)
        pitch_z, loud_z = extract_audio_features(wav_path, fps=args.fps)
        # 2. 加载MLP
        mlp = load_mlp(weights_path=os.path.join('vad_multimodal','weights','mlp_d.pth'), device=device)
    else:
        pitch_z, loud_z, mlp = None, None, None
    for idx, (timestamp, frame) in enumerate(tqdm(extract_frames(args.input, args.fps), total=num_samples, desc='Processing frames', unit='frame')):
        valence, arousal = emonet.predict(frame)
        D = estimate_D(valence, arousal)
        # 多模态Dominance
        if MULTIMODAL_OK:
            # 音频特征
            pz = pitch_z[idx] if pitch_z is not None and idx < len(pitch_z) else np.nan
            lz = loud_z[idx] if loud_z is not None and idx < len(loud_z) else np.nan
            # 人体bbox
            try:
                bbox_norm = extract_body_bbox(frame, frame.shape)
            except Exception:
                bbox_norm = np.nan
            features = [valence, arousal, pz, lz, bbox_norm]
            if all(np.isnan(f) for f in features[2:]):
                D_mod = np.nan
                D_final = D
            else:
                try:
                    D_mod = estimate_D_mod(features, mlp, device=device)
                except Exception:
                    D_mod = np.nan
                D_final = D_mod if not np.isnan(D_mod) else D
        else:
            D_mod = np.nan
            D_final = D
        results.append({'timestamp': timestamp, 'V': valence, 'A': arousal, 'D': D, 'D_mod': D_mod, 'D_final': D_final})
    df = pd.DataFrame(results)
    for col in ['V', 'A', 'D']:
        df[f'{col}_smooth'] = smooth_series(df[col], alpha=args.smooth)
    df.to_csv(csv_path, index=False)
    plot_3d(csv_path, out_path=png_path)
    plot_3d_animation(csv_path, out_gif_path=gif_path, out_mp4_path=mp4_path)
    print(f'Done. Results saved to {csv_path}, {png_path}, {gif_path} and {mp4_path}')

    # 自动生成3D交互HTML和快捷方式
    import subprocess, sys
    viewer_py = os.path.join(os.path.dirname(__file__), 'vad_viewer.py')
    if os.path.exists(viewer_py):
        subprocess.run([
            sys.executable, viewer_py,
            '--csv', csv_path,
            '--port', '0',  # 0端口只生成文件不启动服务
            '--export'
        ], check=False)
    else:
        print('Warning: vad_viewer.py not found, skip 3D HTML export.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Framewise V-A-D extractor from video')
    parser.add_argument('--input', type=str, help='Input video file (.mp4/.mov)')
    parser.add_argument('--url', type=str, help='Input video url (tiktok/douyin/HLS)')
    parser.add_argument('--prefer_stream', action='store_true', help='Prefer streaming mode if available')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to sample')
    parser.add_argument('--smooth', type=float, default=0.3, help='Exponential smoothing alpha')
    parser.add_argument('--mode', choices=['accurate', 'fast'], default='accurate', help='accurate = highest precision; fast = precision/speed trade-off')
    args = parser.parse_args()
    # 处理url或本地文件
    if args.url:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = get_local_or_stream(args.url, tmp_dir, args.prefer_stream)
            if isinstance(source, dict):
                # 流式处理
                def process_stream(stream_url, fps):
                    import cv2
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        raise IOError(f"Cannot open stream: {stream_url}")
                    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                    step = int(round(video_fps / fps))
                    frame_idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_idx % step == 0:
                            timestamp = frame_idx / video_fps
                            yield (timestamp, frame)
                        frame_idx += 1
                    cap.release()
                # 用process_stream替换extract_frames
                extract_frames = lambda path, fps: process_stream(source['stream'], source['fps'])
                args.input = source['stream']
            else:
                args.input = source
    main(args) 