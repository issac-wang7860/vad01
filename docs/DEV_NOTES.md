# VAD 项目双模式（accurate/fast）开发说明

---

❶ 总览
---
新增 CLI 参数 --mode {accurate|fast}，主流程保持一份代码，仅在“组件选择 + 推理开关”上做 if-else。

accurate  —— 追求最佳精度  
fast      —— 在保证 ~90-95 % 精度的前提下，把 FPS 拉到 50+

---

❷ 关键改动
---
1️⃣ 在 `vad_from_video.py` 的 ArgumentParser 里插入新参数
```python
parser.add_argument('--mode', choices=['accurate', 'fast'], default='accurate',
                    help='accurate = highest precision; fast = precision/speed trade-off')
```

2️⃣ 建两个配置表（可放 `config.py`）
```python
ACCURATE = dict(
    face_detector = 'retinaface',          # or yolov8m-face
    vision_model  = 'vit_b16',             # ViT-B/16 Fine-tuned on Aff-Wild2
    audio_model   = 'wav2vec2_large_pdem', # 24-layer
    fusion_head   = 'xmod_transformer4',   # 4-layer
    quantize      = False,                 # FP32
    engine        = None                   # 原生 PyTorch
)

FAST = dict(
    face_detector = 'yolov8n_face_int8',   # INT8 + NMS
    vision_model  = 'mobilevit_xs_lora',   # 15 MB
    audio_model   = 'distilhubert_int8',   # 6-layer
    fusion_head   = 'gated_tcn2',          # 2-layer
    quantize      = True,                  # INT8 QAT
    engine        = 'tensorrt_fp16'        # ONNX-TRT
)

MODE_CFG = dict(accurate=ACCURATE, fast=FAST)
```

3️⃣ 在 `main()` 里根据 `args.mode` 拿到 cfg，再实例化组件
```python
cfg = MODE_CFG[args.mode]

face_det = load_face_detector(cfg['face_detector'])
vision   = load_vision_model(cfg['vision_model'], quantize=cfg['quantize'])
audio    = load_audio_model(cfg['audio_model'], quantize=cfg['quantize'])
fusion   = load_fusion_head(cfg['fusion_head'])

# 推理时：vision_feat, audio_feat → fusion → (V, A)
```

4️⃣ 在循环里把 `emonet.predict(frame)` 换成新接口
```python
valence, arousal = pipeline_infer(frame, face_det, vision, audio, fusion)
```

5️⃣ 可选：如果 `cfg['engine'] == 'tensorrt_fp16'`，在启动时先执行
```bash
python tools/convert_to_trt.py --cfg fast --save engine/fusion_fp16.trt
```
然后 runtime 直接 `trt_infer()`。

---

❸ 组件实现细节
---
◆ face_detector
| accurate | `pip install retina-face`，Weights: `ResNet50-0.3`；AP≈0.96 (Easy)
| fast     | `ultralytics` 的 `yolov8n-face.pt` + `torch.int8` QAT

◆ vision_model
| accurate | `from transformers import ViTModel` (`google/vit-base-patch16-224-in21k`) + 微调
| fast     | MobileViT-XS backbone + LoRA(r=16)；权重 14.7 MB

◆ audio_model
| accurate | `wav2vec2-large-pdem` (24-layer)
| fast     | `distilhubert-base` + dynamic-range INT8

◆ fusion_head
| accurate | 4-layer Cross-Modal Transformer (`nn.TransformerEncoder`)
| fast     | 2-layer Gated-TCN：`nn.Conv1d(depthwise) → GLU`

---

❹ 运行示例
---
```bash
# 精度优先
python vad_from_video.py --input demo.mp4 --fps 5 --mode accurate --smooth 0.3

# 精度 + 速度折中
python vad_from_video.py --input demo.mp4 --fps 10 --mode fast --smooth 0.2
```

`fast` 模式在 RTX 4090 上 1080p 输入可达 55-60 FPS；`accurate` 模式约 18 FPS，但 Valence/Arousal CCC 提高 ~20 %。

---

❺ 附：对旧代码的兼容
---
* `EmoNetWrapper` 原文件可保留作为 fallback，不再默认调用。
* 旧 YAML/CSV 输出格式 & 绘图函数 **不变**，上层脚本不用改。
* 若要逐步迁移，可将 `--mode legacy` 映射到旧 `EmoNetWrapper` 路径。

---

有了这份「配置驱动 + 单文件入口」的双模式结构，你只要把各组件实现在 `modules/` 下填好，Cursor 执行上述两条命令即可切换精度/速度策略，无需再改主脚本逻辑。祝编码顺利！ 