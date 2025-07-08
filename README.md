# 逐帧 V-A-D 提取器

## 功能
- 输入任意 .mp4 / .mov 视频
- 每秒采 5 帧
- 对每帧人脸估计 (Valence, Arousal)，并映射 Dominance
- 输出 results_视频名_模式.csv (timestamp, V, A, D, V_smooth, A_smooth, D_smooth) 到 outputs/视频名_模式/ 文件夹
- 生成 3D 轨迹 plot_视频名_模式.png 到 outputs/视频名_模式/ 文件夹
- 生成 3D 动画 vad_trajectory_视频名_模式.gif 和 vad_trajectory_视频名_模式.mp4 到 outputs/视频名_模式/ 文件夹

## 安装依赖
```bash
pip install -r requirements.txt
```

## 用法
```bash
# 精度优先
python vad_from_video.py --input your_video.mp4 --fps 5 --smooth 0.3 --mode accurate

# 速度优先
python vad_from_video.py --input your_video.mp4 --fps 5 --smooth 0.3 --mode fast
```

## 说明
- 需补全 estimate_va(frame) 以调用 EmoNet 进行情感估计
- 若无检测结果，输出 NaN
- 支持指数平滑
- 所有输出文件均保存在 outputs/视频名_模式/ 文件夹，文件名自动带上视频名和模式，避免覆盖。
- --mode accurate 追求最高精度，--mode fast 追求速度（未来可扩展为不同模型/推理方案） 