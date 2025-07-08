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

def plot_3d_animation(csv_path, out_path='vad_trajectory.gif'):
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
    ani.save(out_path, writer='pillow', fps=20)
    plt.close()

# 6. Main pipeline
def main(args):
    # 输出文件夹
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    # 根据输入视频名生成前缀
    video_basename = os.path.splitext(os.path.basename(args.input))[0]
    csv_path = os.path.join(output_dir, f'results_{video_basename}.csv')
    png_path = os.path.join(output_dir, f'plot_{video_basename}.png')
    gif_path = os.path.join(output_dir, f'vad_trajectory_{video_basename}.gif')
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
    for timestamp, frame in tqdm(extract_frames(args.input, args.fps), total=num_samples, desc='Processing frames', unit='frame'):
        valence, arousal = emonet.predict(frame)
        D = estimate_D(valence, arousal)
        results.append({'timestamp': timestamp, 'V': valence, 'A': arousal, 'D': D})
    df = pd.DataFrame(results)
    for col in ['V', 'A', 'D']:
        df[f'{col}_smooth'] = smooth_series(df[col], alpha=args.smooth)
    df.to_csv(csv_path, index=False)
    plot_3d(csv_path, out_path=png_path)
    plot_3d_animation(csv_path, out_path=gif_path)
    print(f'Done. Results saved to {csv_path}, {png_path} and {gif_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Framewise V-A-D extractor from video')
    parser.add_argument('--input', type=str, required=True, help='Input video file (.mp4/.mov)')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to sample')
    parser.add_argument('--smooth', type=float, default=0.3, help='Exponential smoothing alpha')
    args = parser.parse_args()
    main(args) 