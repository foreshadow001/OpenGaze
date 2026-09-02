"""v3 共享几何恢复链（ours-without-roll 协议唯一实现，四个 normalize_* 共用）

从 v2 产物逐样本精确恢复归一化输入（无 DLT / 无 PnP，协议 §3）：
  hR, t        = Kabsch(model, face_landmarks_3d)     # 刚体反解 ~1e-12
  gp 方向       = face_mat_norm^T · unit(face_gaze)    # face_mat_norm = R_v2
  face_center  = 6 点加权中心（与 normalizeData_face 同式）
归一化 fixed_forward=True（roll-only）：hR_norm = Rz(−γ)·hR，欧拉 (α,β)
归一化前后不变、γ≡0（协议 §2，已数值验证到 1e-15）。
世界系头姿（协议 §4）：elev = arcsin(−v_w[1])−30°、azim = atan2(−v_w[0],−v_w[2])，
v_w = R_head_world·[0,0,−1]；ext = 世界←相机旋转（None 时世界=相机）。
"""
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT),
           str(_PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

import face_model_core as core
from utils.normalization import HEAD_PITCH_OFFSET, normalizeData_face, \
    vector_to_angles

GP_SCALE = 300.0     # gp = face_center + dir·300（normalizeData_face 只用方向）


def unit_from_angles(theta, phi):
    """vector_to_angles 的逆：单位方向向量"""
    return np.array([-np.cos(theta) * np.sin(phi), -np.sin(theta),
                     -np.cos(theta) * np.cos(phi)])


def face_center_of(X_cam):
    """normalizeData_face 同式的 6 点加权中心"""
    two_eye = X_cam[0:4].mean(axis=0)
    nose = X_cam[4:6].mean(axis=0)
    return (two_eye + nose) / 2.0


def world_elev_azim(hR, ext=None):
    """世界系头姿 (elev, azim) 度；ext=None 时世界=相机（GC/MPII）"""
    R_hw = ext @ hR if ext is not None else hR
    t_w, p_w = vector_to_angles(R_hw @ np.array([0., 0., -1.]))
    return float(np.degrees(t_w) + HEAD_PITCH_OFFSET), float(np.degrees(p_w))


def v3_sample(img, model, X_cam, matn, gaze_rad, K, ext=None):
    """一个样本的完整 v3 计算（fixed_forward=True）→ dict"""
    hR, t = core.kabsch(model, X_cam)
    rvec = cv2.Rodrigues(hR)[0]
    tvec = t.reshape(3, 1)
    gp_dir = matn.T @ unit_from_angles(gaze_rad[0], gaze_rad[1])
    gp = (face_center_of(X_cam) + gp_dir * GP_SCALE).reshape(3, 1)
    img_w, hr_norm, gc = normalizeData_face(
        img, model, rvec, tvec, gp, K, fixed_forward=True)[:3]
    hR_n = cv2.Rodrigues(hr_norm)[0]
    gc_hcs = hR_n.T @ gc
    gc_hcs /= np.linalg.norm(gc_hcs)
    theta, phi = vector_to_angles(gc.flatten())
    hcs_t, hcs_p = vector_to_angles(gc_hcs.flatten())
    return {'patch': img_w,
            'R': hR_n @ hR.T,                    # R_v3 = Rz(−γ)
            'gaze': (theta, phi),                # v3 CCS（roll 修正、头姿保留）
            'hcs': (hcs_t, hcs_p),               # ≡ v2 face_gaze_hcs
            'world': world_elev_azim(hR, ext)}   # face_head_elev_azim
