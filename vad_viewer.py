import argparse
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import viewer_utils
import os
from dash_extensions import EventListener

# ========== CLI ===========
parser = argparse.ArgumentParser(description='VAD 3D Interactive Viewer')
parser.add_argument('--csv', type=str, required=True, help='输入CSV文件')
parser.add_argument('--video', type=str, default=None, help='可选：同步视频文件')
parser.add_argument('--port', type=int, default=8050, help='Dash服务端口')
parser.add_argument('--export', action='store_true', help='导出静态HTML')
args = parser.parse_args()

# ========== 数据加载 ===========
df = viewer_utils.load_dataframe(args.csv)
csv_basename = os.path.splitext(os.path.basename(args.csv))[0]

# ========== Dash App ===========
app = dash.Dash(__name__)

# 预计算全局KDE
kde_global = viewer_utils.kde_mesh(df)

# ========== Layout ===========
def get_layout(init_cam=None):
    return html.Div([
        html.H2('V-A-D 3D Analytics Suite'),
        dcc.Store(id="cam-store"),
        dcc.RadioItems(id="drag-mode", options=[{"label":x,"value":x} for x in ["pan","orbit"]], value="pan", inline=True),
        EventListener(
            id="key-listener",
            events=[{"event":"keydown","props":["key"]}],
            children=[
                dcc.Graph(
                    id='vad-3d-graph',
                    config={
                        "scrollZoom": True,
                        "scrollZoomSpeed": 0.2,
                        "doubleClick": "reset+autosize",
                        "modeBarButtonsToRemove": ["resetCameraDefault3d"],
                        "displaylogo": False
                    }
                )
            ]
        ),
        dcc.Slider(id='frame-slider', min=0, max=len(df)-1, value=0, step=1, marks=None, tooltip={"placement": "bottom"}),
        html.Div([
            html.Label('Window width (s):'),
            dcc.Dropdown(id='window-width', options=[{'label': f'{w} s', 'value': w} for w in [1,3,5]], value=3),
            html.Label('Density overlay:'),
            dcc.Dropdown(id='density-overlay', options=[{'label': x, 'value': x} for x in ['Local','Global','Off']], value='Local'),
            html.Label('Playback speed:'),
            dcc.Dropdown(id='playback-speed', options=[{'label': f'{s}x', 'value': s} for s in [0.5,1,2]], value=1),
            html.Button('Play', id='play-btn', n_clicks=0),
            html.Button('Pause', id='pause-btn', n_clicks=0),
            dcc.Interval(id='play-interval', interval=500, n_intervals=0, disabled=True)
        ], style={'display':'flex','gap':'2em'}),
        html.Div(id='current-timestamp'),
        html.Button('Export HTML', id='export-btn', n_clicks=0),
        dcc.Download(id='download-html'),
        html.Div([dcc.Video(id='video-player', src=args.video, controls=True)]) if args.video else None
    ])

app.layout = get_layout()

# ========== Camera Store回调 ===========
@app.callback(
    Output("cam-store","data"),
    Input("vad-3d-graph","relayoutData"),
    State("cam-store","data"),
    prevent_initial_call=True
)
def save_cam(redata, camdata):
    if redata and "scene.camera" in redata:
        return {"csv": csv_basename, "camera": redata["scene.camera"]}
    return dash.no_update

# ========== Figure回调 ===========
@app.callback(
    Output('vad-3d-graph', 'figure'),
    Output('current-timestamp', 'children'),
    Input('frame-slider', 'value'),
    Input('window-width', 'value'),
    Input('density-overlay', 'value'),
    Input('drag-mode', 'value'),
    State('cam-store', 'data')
)
def update_figure(frame_idx, window_width, density_overlay, drag_mode, camdata):
    t0 = df.iloc[frame_idx]['timestamp']
    mask = (df['timestamp'] >= t0 - window_width/2) & (df['timestamp'] <= t0 + window_width/2)
    df_win = df[mask]
    trace_all = go.Scatter3d(x=df['V_smooth'], y=df['A_smooth'], z=df['D_smooth'], mode='lines', line=dict(color='gray', width=2), name='All')
    trace_win = go.Scatter3d(x=df_win['V_smooth'], y=df_win['A_smooth'], z=df_win['D_smooth'], mode='lines', line=dict(color='red', width=6), name='Window')
    traces = [trace_all, trace_win]
    if density_overlay == 'Local':
        mesh = viewer_utils.kde_mesh(df_win)
    elif density_overlay == 'Global':
        mesh = kde_global
    else:
        mesh = None
    if mesh is not None:
        traces.append(go.Isosurface(
            x=mesh['x'].flatten(), y=mesh['y'].flatten(), z=mesh['z'].flatten(),
            value=mesh['density'].flatten(),
            isomin=0.2*mesh['density'].max(), isomax=mesh['density'].max(),
            opacity=0.2, surface_count=2, colorscale='Blues', showscale=False, name='KDE'))
    # camera持久化
    init_cam = camdata["camera"] if camdata and camdata.get("csv") == csv_basename else None
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='Valence', yaxis_title='Arousal', zaxis_title='Dominance',
            dragmode=drag_mode,
            camera=init_cam,
            projection={"type":"perspective"}
        ),
        uirevision="lock",
        transition={"duration":300,"easing":"cubic-in-out"}
    )
    return fig, f"Current: t={t0:.2f}s, frame={frame_idx}"

# ========== 键盘快捷键回调 ===========
@app.callback(
    Output('vad-3d-graph', 'relayoutData'),
    Input('key-listener', 'event'),
    prevent_initial_call=True
)
def reset_camera(event):
    if event and event.get('key') == 'r':
        return {"scene.camera": None}  # 复位到初始视角
    return dash.no_update

# ========== 导出HTML ===========
@app.callback(
    Output('download-html', 'data'),
    Input('export-btn', 'n_clicks'),
    prevent_initial_call=True
)
def export_html(n_clicks):
    import plotly.io as pio
    fig, _ = update_figure(0, 3, 'Local', 'pan', None)
    html_str = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
    fname = f"viewer_{csv_basename}.html"
    return dict(content=html_str, filename=fname)

# ========== 启动 ===========
if __name__ == '__main__':
    if args.export:
        import plotly.io as pio
        fig, _ = update_figure(0, 3, 'Local', 'pan', None)
        fname = f"viewer_{csv_basename}.html"
        pio.write_html(fig, fname, include_plotlyjs="cdn", full_html=True)
        print(f"导出静态HTML: {fname}")
    else:
        # 自动本地导出HTML和快捷方式
        import plotly.io as pio
        fig, _ = update_figure(0, 3, 'Local', 'pan', None)
        out_dir = os.path.dirname(args.csv)
        html_path = os.path.join(out_dir, f"viewer_{csv_basename}.html")
        url_path = os.path.join(out_dir, f"viewer_{csv_basename}.url")
        pio.write_html(fig, html_path, include_plotlyjs="cdn", full_html=True)
        with open(url_path, 'w', encoding='utf-8') as f:
            f.write(f"[InternetShortcut]\nURL=http://127.0.0.1:{args.port}\n")
        print(f"已自动导出本地HTML: {html_path} 和快捷方式: {url_path}")
        app.layout = get_layout()
        app.run_server(debug=True, port=args.port) 