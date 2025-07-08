import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None

def extract_body_bbox(frame, frame_shape):
    """
    mediapipe pose检测，返回bbox归一化面积（bbox/frame_area），无检测返回nan。
    """
    if mp is None:
        return np.nan
    pose = mp.solutions.pose.Pose(static_image_mode=True)
    results = pose.process(frame)
    if results.pose_landmarks:
        xs = [lm.x for lm in results.pose_landmarks.landmark]
        ys = [lm.y for lm in results.pose_landmarks.landmark]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        h, w = frame_shape[:2]
        bbox_area = (max_x - min_x) * (max_y - min_y) * w * h
        frame_area = w * h
        return bbox_area / frame_area
    return np.nan 