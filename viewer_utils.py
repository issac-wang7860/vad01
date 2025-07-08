import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from typing import Dict
from sklearn.decomposition import PCA

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    读取CSV，返回补齐NaN的DataFrame，包含timestamp, V_smooth, A_smooth, D_smooth。
    """
    df = pd.read_csv(csv_path)
    for col in ['V_smooth', 'A_smooth', 'D_smooth']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return df

def kde_mesh(df: pd.DataFrame, grid_n: int = 28) -> Dict[str, np.ndarray]:
    """
    基于窗口内VAD点，构建三维/二维/一维KDE密度网格。
    返回dict: x, y, z, density (shape: [grid_n, grid_n, grid_n] 或 [grid_n, grid_n] 或 [grid_n])
    """
    points = np.vstack([
        df['V_smooth'].values,
        df['A_smooth'].values,
        df['D_smooth'].values
    ])
    # 检查有效维度
    pca = PCA()
    pca.fit(points.T)
    n_dim = np.sum(pca.explained_variance_ > 1e-8)
    grid = np.linspace(-1, 1, grid_n)
    if n_dim == 3:
        try:
            kde = gaussian_kde(points)
            x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')
            coords = np.vstack([x.ravel(), y.ravel(), z.ravel()])
            dens = kde(coords).reshape((grid_n, grid_n, grid_n))
            return dict(x=x, y=y, z=z, density=dens)
        except Exception:
            pass
    elif n_dim == 2:
        # 投影到主平面
        pts2d = pca.transform(points.T)[:,:2].T
        try:
            kde = gaussian_kde(pts2d)
            x, y = np.meshgrid(grid, grid, indexing='ij')
            coords = np.vstack([x.ravel(), y.ravel()])
            dens = kde(coords).reshape((grid_n, grid_n))
            # 补z为0平面
            z = np.zeros_like(x)
            return dict(x=x, y=y, z=z, density=dens)
        except Exception:
            pass
    elif n_dim == 1:
        # 投影到主轴
        pts1d = pca.transform(points.T)[:,0]
        try:
            kde = gaussian_kde(pts1d)
            x = grid
            dens = kde(x)
            y = np.zeros_like(x)
            z = np.zeros_like(x)
            return dict(x=x, y=y, z=z, density=dens)
        except Exception:
            pass
    # fallback: 空密度
    x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')
    dens = np.zeros((grid_n, grid_n, grid_n))
    return dict(x=x, y=y, z=z, density=dens) 