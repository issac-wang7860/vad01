# 逐帧 V-A-D 提取器

## 功能
- 输入任意 .mp4 / .mov 视频
- 每秒采 5 帧
- 对每帧人脸估计 (Valence, Arousal)，并映射 Dominance
- 输出 results.csv (timestamp, V, A, D, V_smooth, A_smooth, D_smooth)
- 生成 3D 轨迹 plot.png

## 安装依赖
```bash
pip install -r requirements.txt
```

## 用法
```bash
python vad_from_video.py --input your_video.mp4 --fps 5 --smooth 0.3
```

## 说明
- 需补全 estimate_va(frame) 以调用 EmoNet 进行情感估计
- 若无检测结果，输出 NaN
- 支持指数平滑 