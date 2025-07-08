# 逐帧 V-A-D 提取器

## 功能
- 输入任意 .mp4 / .mov 视频
- 每秒采 5 帧
- 对每帧人脸估计 (Valence, Arousal)，并映射 Dominance
- 输出 results_视频名_模式.csv (timestamp, V, A, D, V_smooth, A_smooth, D_smooth) 到 outputs/视频名_模式/ 文件夹
- 生成 3D 轨迹 plot_视频名_模式.png 到 outputs/视频名_模式/ 文件夹
- 生成 3D 动画 vad_trajectory_视频名_模式.gif 和 vad_trajectory_视频名_模式.mp4 到 outputs/视频名_模式/ 文件夹
- 生成交互式3D分析 viewer_视频名_模式.html 和快捷方式 .url 到同一文件夹

## 3D交互体验升级
- 左键拖拽 = 平移，右键/CTRL = 旋转
- 滚轮缩放支持 scrollZoom，缩放更顺滑
- 视角持久化：UI交互不再复位，支持每个CSV记忆默认视角
- 下拉切换「Pan / Orbit」模式，双击复位，R键快捷复位
- 支持导出离线HTML，随时本地交互查看

## 安装依赖
```bash
pip install -r requirements.txt
pip install -r requirements_viewer.txt
```

## 用法
```bash
# 精度优先
python vad_from_video.py --input your_video.mp4 --fps 5 --smooth 0.3 --mode accurate

# 速度优先
python vad_from_video.py --input your_video.mp4 --fps 5 --smooth 0.3 --mode fast

# 交互式3D分析
python vad_viewer.py --csv outputs/your_video_模式/results_your_video_模式.csv --port 8050
```

## 说明
- 需补全 estimate_va(frame) 以调用 EmoNet 进行情感估计
- 若无检测结果，输出 NaN
- 支持指数平滑
- 所有输出文件均保存在 outputs/视频名_模式/ 文件夹，文件名自动带上视频名和模式，避免覆盖。
- --mode accurate 追求最高精度，--mode fast 追求速度（未来可扩展为不同模型/推理方案）
- 3D分析支持平移/旋转/缩放/快捷键/视角记忆/导出HTML等高级交互体验 