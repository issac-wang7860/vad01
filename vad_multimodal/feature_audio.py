import numpy as np
import librosa

def extract_audio_features(wav_path, sr=16000, fps=5, fmin=90, fmax=300):
    """
    提取音频的每帧pitch和loudness（z-score归一化），对齐视频fps。
    返回: pitch_z, loud_z (均为长度N的np.ndarray)
    """
    try:
        y, _ = librosa.load(wav_path, sr=sr)
        hop_length = int(sr / fps)
        # pitch
        f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        # loudness
        rms = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)[0]
        loud = 20 * np.log10(rms + 1e-7)
        # z-score
        pitch_z = (f0 - np.nanmean(f0)) / (np.nanstd(f0) + 1e-7)
        loud_z = (loud - np.nanmean(loud)) / (np.nanstd(loud) + 1e-7)
        return pitch_z, loud_z
    except Exception as e:
        print(f"[feature_audio] Failed: {e}")
        return None, None 