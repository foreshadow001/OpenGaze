"""ours-without-roll 抽样可视化（2026-08-29）：CCS / HCS 双视线标注

随机抽 10 个 xgaze 样本（个性化模型覆盖的相机），按第三版协议归一化：
  个性化 model6 PnP -> normalizeData_face(fixed_forward=True)
每行输出归一化 patch + 三组角度（deg）：
  head    头姿 (pitch, yaw)——归一化系（横滚已随头稳定）
  gaze_CCS 归一化相机系视线 (pitch, yaw) = h5 的 face_gaze 字段
  gaze_HCS 头部坐标系视线 (pitch, yaw) = h5 的 face_gaze_head 字段（新标注）
不使用任何官方标注数据（rvec/tvec/检测点）。

输出: viz/xgaze_ours_without_roll_samples.png + .json
用法（仓库根目录）: python preprocess/ours-without-roll/sample_viz.py
"""
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))

from utils.logger import get_logger                                   # noqa: E402
from utils.normalization import estimateHeadPose, normalizeData_face, vector_to_angles

log = get_logger('preprocess.ours_without_roll.sample_viz')

RAW = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/train')
ANN = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/annotation_train')
CALIB = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')
LM = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/landmarks')
FM = Path('/media/yanglinxuan/ylx/xgaze_specific_face_model/face_models')
OUT = Path(__file__).resolve().parent / 'viz'

IDX6 = [35, 39, 89, 93, 78, 84]
FLIP = [3, 6, 13]
N_SAMPLE = 10
RNG = np.random.default_rng(20260829)


def load_K(cam):
    fs = cv2.FileStorage(str(CALIB / f'cam{cam:02d}.xml'), cv2.FILE_STORAGE_READ)
    K = fs.getNode('Camera_Matrix').mat()
    dist = fs.getNode('Distortion_Coefficients').mat()
    fs.release()
    return K, dist


def gaze_point_of(subject, frame, cam):
    """标注 CSV 的注视点（相机系 3D；只用第 5-7 列，不碰 rvec/tvec）"""
    with open(ANN / f'subject{subject:04d}.csv') as f:
        for line in f:
            p = line.strip().split(',')
            if p[0] == f'frame{frame:04d}' and p[1] == f'cam{cam:02d}.JPG':
                return [float(p[4]), float(p[5]), float(p[6])]
    return None


def main():
    subjects = sorted(int(p.stem[7:]) for p in LM.glob('subject*.h5'))
    picks = []
    for sid in RNG.permutation(subjects):
        with h5py.File(LM / f'subject{sid:04d}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
        rows = np.arange(len(fr))
        RNG.shuffle(rows)
        for r in rows:
            cam = int(ci[r])
            if (FM / f'subject{sid:04d}' / f'cam{cam:02d}_model6.txt').is_file():
                picks.append((sid, int(fr[r]), cam, lm[r]))
                break
        if len(picks) >= N_SAMPLE:
            break

    W_IMG, H_IMG, TXT, PAD = 224, 224, 106, 14
    canvas = np.full((len(picks) * (H_IMG + TXT) + PAD,
                      W_IMG + 2 * PAD, 3), 255, np.uint8)
    records = []

    for i, (sid, frame, cam, lm106) in enumerate(picks):
        img = cv2.imread(str(RAW / f'subject{sid:04d}' / f'frame{frame:04d}'
                             / f'cam{cam:02d}.JPG'))
        if cam in FLIP:
            img = cv2.rotate(img, cv2.ROTATE_180)
        gp = gaze_point_of(sid, frame, cam)
        K, dist = load_K(cam)
        m6 = np.loadtxt(FM / f'subject{sid:04d}' / f'cam{cam:02d}_model6.txt')

        rvec, tvec = estimateHeadPose(
            lm106[IDX6].reshape(6, 1, 2).astype(float), m6, K, dist)
        img_w, hr_norm, gc_ccs = normalizeData_face(
            img, m6, rvec, tvec, gp, K, fixed_forward=True)[:3]
        hR_norm = cv2.Rodrigues(hr_norm)[0]
        gc_hcs = hR_norm.T @ gc_ccs

        hp, hy = vector_to_angles(hR_norm @ np.array([0.0, 0.0, -1.0]))   # 模型 -z 面向外
        cp, cy = vector_to_angles(gc_ccs.ravel())
        tp, ty = vector_to_angles(gc_hcs.ravel())

        y0 = PAD + i * (H_IMG + TXT)
        canvas[y0:y0 + H_IMG, PAD:PAD + W_IMG] = img_w
        put = lambda t, dy, c: cv2.putText(canvas, t, (PAD + 2, y0 + H_IMG + dy),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        put(f'S{sid:04d} f{frame:04d} c{cam:02d}', 20, (60, 60, 60))
        put(f'head ({np.degrees(hp):+.1f},{np.degrees(hy):+.1f})', 44, (30, 30, 30))
        put(f'gaze_CCS ({np.degrees(cp):+.1f},{np.degrees(cy):+.1f})', 68, (30, 30, 30))
        put(f'gaze_HCS ({np.degrees(tp):+.1f},{np.degrees(ty):+.1f})', 92, (0, 0, 220))

        records.append({
            'subject': int(sid), 'frame': int(frame), 'cam': int(cam),
            'gaze_ccs_pitch_yaw_deg': [round(float(np.degrees(cp)), 2),
                                       round(float(np.degrees(cy)), 2)],
            'gaze_hcs_pitch_yaw_deg': [round(float(np.degrees(tp)), 2),
                                       round(float(np.degrees(ty)), 2)],
            'head_pitch_yaw_deg': [round(float(np.degrees(hp)), 2),
                                   round(float(np.degrees(hy)), 2)],
        })
        log.info(f'S{sid:04d} f{frame:04d} c{cam:02d}: CCS({np.degrees(cp):+.1f},'
                 f'{np.degrees(cy):+.1f}) HCS({np.degrees(tp):+.1f},{np.degrees(ty):+.1f}) '
                 f'head({np.degrees(hp):+.1f},{np.degrees(hy):+.1f})')

    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / 'xgaze_ours_without_roll_samples.png'), canvas)
    (OUT / 'xgaze_ours_without_roll_samples.json').write_text(
        json.dumps({'note': 'v3 ours-without-roll: personalized model6 PnP + '
                            'normalizeData_face(fixed_forward=True); '
                            'gaze_ccs = face_gaze, gaze_hcs = face_gaze_head',
                    'samples': records}, indent=2))
    log.info(f'输出 {OUT / "xgaze_ours_without_roll_samples.png"}')


if __name__ == '__main__':
    main()
