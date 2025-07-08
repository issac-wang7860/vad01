import viewer_utils
import pandas as pd
import numpy as np

def test_load():
    df = viewer_utils.load_dataframe('outputs/results_happy.csv')
    assert df.shape[0] > 0
    assert {'V_smooth','A_smooth','D_smooth'}.issubset(df.columns)

def test_kde():
    df = viewer_utils.load_dataframe('outputs/results_happy.csv')
    mesh = viewer_utils.kde_mesh(df, grid_n=28)
    assert mesh['density'].shape == (28,28,28)

def test_figure():
    from vad_viewer import update_figure
    df = viewer_utils.load_dataframe('outputs/results_happy.csv')
    fig, info = update_figure(0, 3, 'Local')
    assert fig is not None
    assert isinstance(info, str) 