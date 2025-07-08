import pytest
import numpy as np
import pandas as pd
import os
import sys
import cv2

# 确保可以导入桌面主脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vad_from_video

def test_extract_frames_empty():
    # 用空视频文件测试 extract_frames
    # 这里只做接口存在性和异常处理测试
    with pytest.raises(IOError):
        list(vad_from_video.extract_frames('nonexistent.mp4'))

def test_estimate_va_stub():
    # 测试 stub 返回 nan
    dummy = np.zeros((224,224,3), dtype=np.uint8)
    v, a = vad_from_video.estimate_va(dummy)
    assert np.isnan(v) and np.isnan(a)

def test_estimate_D_mapping():
    # 测试离散情绪映射
    assert vad_from_video.estimate_D(0,0,'happy') == 0.8
    assert vad_from_video.estimate_D(0,0,'anger') == -0.6
    assert np.isnan(vad_from_video.estimate_D(np.nan, np.nan))

def test_smooth_series():
    s = pd.Series([1,2,3,4,5])
    smoothed = vad_from_video.smooth_series(s, alpha=0.5)
    assert len(smoothed) == 5
    assert smoothed.iloc[0] == 1 