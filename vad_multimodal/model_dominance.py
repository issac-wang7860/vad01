import torch
import torch.nn as nn
import numpy as np
import os

def backup_estimate_D(valence, arousal):
    if np.isnan(valence) or np.isnan(arousal):
        return np.nan
    return 0.5 * valence + 0.5 * arousal * np.sign(valence)

class TinyMLP(nn.Module):
    def __init__(self, input_dim=5, hidden=[64,128], output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], output_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

def load_mlp(weights_path=None, device='cpu'):
    model = TinyMLP()
    model.eval()
    model.to(device)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
    return model

def estimate_D_mod(features, model, device='cpu'):
    """
    features: [V, A, pitch_z, loud_z, bbox_norm], nan自动补零+mask
    """
    feats = np.array(features, dtype=np.float32)
    mask = ~np.isnan(feats)
    feats_nan_to_zero = np.nan_to_num(feats)
    input_vec = np.concatenate([feats_nan_to_zero, mask.astype(np.float32)])  # shape (10,)
    input_tensor = torch.from_numpy(input_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        d_mod = model(input_tensor).cpu().item()
    return d_mod 