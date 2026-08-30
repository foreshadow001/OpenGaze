"""标准系人脸模型 3D 交互可视化（纯 HTML 输出，2026-08-30 自 canonical_model.py 分离）

标准系（解剖轴）定义见 CLAUDE.md 约定 9 与 utils/normalization.py
canonicalize_face_model（唯一定义，本脚本只做展示）：
全部被试真实模型（true6 → 标准系）+ 标准模型 canonical_mean6.txt + gen6
（标准化后）同图对比，并报告 gen6 自身坐标系相对标准的偏差。

输出：metrics/true_model/canonical_standard_3d.html
用法：python preprocess/zhang2015-specific-face-model/get_face_model/xgaze/viz_canonical_models.py
"""
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.normalization import canonicalize_face_model as canonicalize  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path('/media/yanglinxuan/sfm/xgaze_specific_face_model/face_models')
GEN6_FILE = HERE.parents[4] / 'preprocess' / 'zhang2015-insightface' / 'face_model_xgaze.txt'
OUT = HERE / 'canonical_standard_3d.html'
LBL = ['eye_out_L', 'eye_in_L', 'eye_in_R', 'eye_out_R', 'nose_L', 'nose_R']


def rpy_of(R):
    """R（标准系轴在旧系方向）的旋转角分解：新系相对旧系的 pitch/yaw/roll（度）。"""
    rv = np.degrees(np.linalg.norm(__import__('cv2').Rodrigues(R.T)[0]))
    return rv


gen6 = np.loadtxt(GEN6_FILE)[[20, 23, 26, 29, 15, 19], :]
subs = sorted(p.parent.name for p in MODELS_DIR.glob('*/true6.txt'))
models = {s: np.loadtxt(MODELS_DIR / s / 'true6.txt') for s in subs}

canon = {}
Rs = {}
for s, m in models.items():
    canon[s], Rs[s], _ = canonicalize(m)
gen6_c, R_g, _ = canonicalize(gen6)
allc = np.stack(list(canon.values()))
mean_c = np.loadtxt(MODELS_DIR / 'canonical_mean6.txt')   # 由 true_face_model.py 产出

# gen6 与各被试原坐标系相对标准的偏差
import cv2
ang_g = np.degrees(np.linalg.norm(cv2.Rodrigues(R_g.T)[0]))
angs = [np.degrees(np.linalg.norm(cv2.Rodrigues(Rs[s].T)[0])) for s in subs]
# gen6 的三轴偏差角
x_dev = np.degrees(np.arccos(np.clip(R_g[0] @ np.array([1, 0, 0.]), -1, 1)))
y_dev = np.degrees(np.arccos(np.clip(R_g[1] @ np.array([0, 1, 0.]), -1, 1)))
z_dev = np.degrees(np.arccos(np.clip(R_g[2] @ np.array([0, 0, 1.]), -1, 1)))
print(f'gen6 坐标系相对解剖标准的偏差: 总旋转 {ang_g:.2f}°（x 轴偏 {x_dev:.2f}°, '
      f'y 轴偏 {y_dev:.2f}°, z 轴偏 {z_dev:.2f}°）')
print(f'80 被试真实模型原系（gen6 对齐系）相对标准: 总旋转中位 '
      f'{np.median(angs):.2f}°（p90 {np.percentile(angs, 90):.2f}°）')

# 可视化
fig = go.Figure()
for j, l in enumerate(LBL):
    fig.add_trace(go.Scatter3d(
        x=allc[:, j, 0], y=allc[:, j, 1], z=allc[:, j, 2],
        mode='markers', name=f'true/{l}', marker=dict(size=3, opacity=0.45),
        hovertext=[f'{s} {l}' for s in subs], hoverinfo='text'))
for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)):
    fig.add_trace(go.Scatter3d(
        x=[mean_c[a, 0], mean_c[b, 0]], y=[mean_c[a, 1], mean_c[b, 1]],
        z=[mean_c[a, 2], mean_c[b, 2]],
        mode='lines', line=dict(color='lime', width=5, dash='dot'),
        hoverinfo='skip', showlegend=False))
fig.add_trace(go.Scatter3d(
    x=mean_c[:, 0], y=mean_c[:, 1], z=mean_c[:, 2],
    mode='markers+text', name='standard model (true mean)',
    text=LBL, textposition='top center',
    textfont=dict(size=10, color='rgb(0,160,0)'),
    marker=dict(size=6, color='lime')))
for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)):
    fig.add_trace(go.Scatter3d(
        x=[gen6_c[a, 0], gen6_c[b, 0]], y=[gen6_c[a, 1], gen6_c[b, 1]],
        z=[gen6_c[a, 2], gen6_c[b, 2]],
        mode='lines', line=dict(color='red', width=7), hoverinfo='skip',
        showlegend=False))
fig.add_trace(go.Scatter3d(
    x=gen6_c[:, 0], y=gen6_c[:, 1], z=gen6_c[:, 2],
    mode='markers+text', name='gen6 (canonicalized)',
    text=LBL, textposition='bottom center',
    textfont=dict(size=10, color='darkred'),
    marker=dict(size=7, color='red', symbol='square')))
L = 55
for name, color, vec in (('x (roll=0: eye line)', 'blue', [1, 0, 0]),
                         ('y (pitch=0: eye->nose)', 'purple', [0, 1, 0]),
                         ('z (face normal)', 'orange', [0, 0, 1])):
    fig.add_trace(go.Scatter3d(
        x=[0, vec[0] * L], y=[0, vec[1] * L], z=[0, vec[2] * L],
        mode='lines+text', line=dict(color=color, width=6),
        text=['', name], textposition='top right',
        textfont=dict(size=11, color=color), name=name, hoverinfo='skip'))

iod = np.linalg.norm(allc[:, 0] - allc[:, 3], axis=1)
nw = np.linalg.norm(allc[:, 4] - allc[:, 5], axis=1)
fig.update_layout(
    title=('Canonical (anatomical) frame: eye line ∥ x (roll=0), eye->nose ∥ y (pitch=0) '
           '& ⊥ x (yaw=0), origin=eye center, z=x×y — '
           f'IOD {iod.mean():.1f}±{iod.std():.1f}, nose {nw.mean():.1f}±{nw.std():.1f}, '
           f'gen6 frame offset {ang_g:.1f}°'),
    scene=dict(xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)',
               aspectmode='data'),
    width=1300, height=950)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(OUT, include_plotlyjs='cdn')
print(f'输出 {OUT}')
