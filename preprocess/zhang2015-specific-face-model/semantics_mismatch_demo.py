"""关键点语义失配可视化（2026-08-28）：通用模型 6 点 vs insightface 检测点

同一张 XGaze 原图（近正脸样本）上并排画两种语义的关键点：
  红色方块  = 官方姿态下通用 50 点模型 6 点的投影（模型认为的解剖学眼角/鼻底位置）
  蓝色圆点  = insightface 实际检测的 6 点（我们管线 PnP 的输入）
  灰色细线  = insightface 眼轮廓多边形（33-42 / 87-96）与鼻部点群（72-86），提供解剖参照
  红→蓝箭头 = 语义偏移方向（检测 − 模型投影）
含左右眼、鼻部三个放大子图。数值闭环（机理验证）打印到日志：
  ① 官方姿态下投影 → 与检测点的逐点偏差（失配模式）
  ② 我们 PnP 姿态下投影 → 与检测点几乎重合（PnP 把失配吸收进姿态的直接证据）
  ③ 俯仰角 δ 与距离缩放的耦合算术（横向 span 被 cosδ 压缩 → PnP 拉近补偿）

输出: viz/semantics_mismatch.png
用法（仓库根目录）: python preprocess/zhang2015-specific-face-model/semantics_mismatch_demo.py
"""
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))

from utils.logger import get_logger                                   # noqa: E402
from utils.normalization import estimateHeadPose                       # noqa: E402

log = get_logger('preprocess.specific_face_model.semantics_demo')

RAW = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/train')
ANN = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/annotation_train')
CALIB = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')
LM = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/landmarks')
OUT = Path(__file__).resolve().parent / 'viz'

FM50 = np.loadtxt(_PROJECT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt')
ROWS6 = [20, 23, 26, 29, 15, 19]        # 模型行: 眼外L 眼内L 眼内R 眼外R 鼻L 鼻R
IDX6 = [35, 39, 89, 93, 78, 84]         # insightface 对应索引
EYE_CONTOUR = [[35, 41, 40, 42, 39, 37, 33, 36],       # 睑缘环（外角→上睑→内角→下睑）
               [89, 95, 94, 96, 93, 91, 87, 90]]
IRIS_CENTER = [34, 38, 88, 92]                        # 眼球中心点（不在睑缘上）
NOSE_PTS = list(range(72, 87))
LBL = ['eye_out_L', 'eye_in_L', 'eye_in_R', 'eye_out_R', 'nose_L', 'nose_R']

SID, FIDX, CAM = 0, 0, 0                # 近正脸校准帧（官方头姿 9.5°）


def main():
    fs = cv2.FileStorage(str(CALIB / f'cam{CAM:02d}.xml'), cv2.FILE_STORAGE_READ)
    K = fs.getNode('Camera_Matrix').mat()
    dist = fs.getNode('Distortion_Coefficients').mat()
    fs.release()

    with h5py.File(LM / f'subject{SID:04d}.h5', 'r') as f:
        fr = f['frame_index'][:].ravel()
        ci = f['cam_index'][:].ravel()
        lm106_all = f['facial_landmarks_2d'][:]
    r = np.where((fr == FIDX) & (ci == CAM))[0][0]
    lm106 = lm106_all[r].astype(float)

    off_rv = off_tv = gp = None
    with open(ANN / f'subject{SID:04d}.csv') as fcsv:
        for line in fcsv:
            p = line.strip().split(',')
            if p[0] == f'frame{FIDX:04d}' and p[1] == f'cam{CAM:02d}.JPG':
                gp = np.array([float(p[4]), float(p[5]), float(p[6])])
                off_rv = np.array([float(p[7]), float(p[8]), float(p[9])])
                off_tv = np.array([float(p[10]), float(p[11]), float(p[12])])
                break

    gen6 = FM50[ROWS6, :]
    det6 = lm106[IDX6]

    def project(rvec, tvec, pts3):
        pr, _ = cv2.projectPoints(np.asarray(pts3, float).reshape(-1, 1, 3),
                                  rvec, tvec, K, dist)
        return pr.reshape(-1, 2)

    proj_official = project(off_rv, off_tv, gen6)
    pnp_rv, pnp_tv = estimateHeadPose(det6.reshape(6, 1, 2), gen6, K, dist)
    proj_pnp = project(pnp_rv, pnp_tv, gen6)

    # ---------- 数值闭环 ----------
    iod = np.linalg.norm(det6[1] - det6[2])          # 内眼角距（检测）
    log.info(f'样本 subject{SID:04d}/frame{FIDX:04d}/cam{CAM:02d}，检测 IOD={iod:.0f}px')
    log.info('① 官方姿态投影 vs 检测（失配模式，px，+y=图下方）:')
    for j, l in enumerate(LBL):
        d = det6[j] - proj_official[j]
        log.info(f'   {l:10s}: dx={d[0]:+7.1f} dy={d[1]:+7.1f}')
    res_off = np.linalg.norm(det6 - proj_official, axis=1).mean()
    res_pnp = np.linalg.norm(det6 - proj_pnp, axis=1).mean()
    log.info(f'② 检测 vs 官方姿态投影 平均残差 {res_off:.1f}px | 检测 vs 我们PnP姿态投影 '
             f'平均残差 {res_pnp:.1f}px  <- PnP 把失配吸收进姿态')
    pitch_off = np.degrees(np.linalg.norm(off_rv))
    pitch_pnp = np.degrees(np.linalg.norm(pnp_rv))
    z_off, z_pnp = float(off_tv[2]), float(pnp_tv.ravel()[2])
    span_off = np.linalg.norm(proj_official[0] - proj_official[3])       # 官方姿态外角距
    span_det = np.linalg.norm(det6[0] - det6[3])
    cosd = np.cos(np.radians(pitch_pnp - pitch_off))
    log.info(f'③ 头姿角: 官方 {pitch_off:.1f}° -> PnP {pitch_pnp:.1f}° (δ={pitch_pnp-pitch_off:+.1f}°) | '
             f'z: {z_off:.0f} -> {z_pnp:.0f}mm')
    log.info(f'   外眼角距: 检测 {span_det:.0f}px, 模型在官方距离 {span_off:.0f}px, '
             f'旋转 δ 后 {span_off*cosd:.0f}px, 再按比例拉近预测 '
             f'{span_off*cosd*z_off/z_pnp:.0f}px  (cosδ 压缩 + 拉近补偿)')

    # ---------- 出图 ----------
    img = cv2.imread(str(RAW / f'subject{SID:04d}' / f'frame{FIDX:04d}'
                         / f'cam{CAM:02d}.JPG'))
    pad = int(iod * 0.9)
    x0 = int(max(0, lm106[:, 0].min() - pad)); x1 = int(min(img.shape[1], lm106[:, 0].max() + pad))
    y0 = int(max(0, lm106[:, 1].min() - pad)); y1 = int(min(img.shape[0], lm106[:, 1].max() + pad))
    crop = img[y0:y1, x0:x1].copy()
    S = 1000 / crop.shape[1]                       # 统一缩放到宽 1000
    crop = cv2.resize(crop, None, fx=S, fy=S)
    off = np.array([x0, y0])

    def P(p):                                      # 原图坐标 -> 画布坐标
        return tuple(((np.asarray(p, float) - off) * S).astype(int))

    for contour in EYE_CONTOUR:                    # 睑缘环（解剖参照）
        pts = np.array([P(lm106[i]) for i in contour])
        cv2.polylines(crop, [pts], True, (150, 150, 150), 1, cv2.LINE_AA)
    for i in IRIS_CENTER:                          # 眼球中心点（小十字）
        x, y = P(lm106[i])
        cv2.line(crop, (x - 5, y), (x + 5, y), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(crop, (x, y - 5), (x, y + 5), (150, 150, 150), 1, cv2.LINE_AA)
    for i in NOSE_PTS:
        cv2.circle(crop, P(lm106[i]), 2, (150, 150, 150), -1)

    for j in range(6):                             # 模型投影（红方）与检测（蓝圆）+ 箭头
        pa, pb = P(proj_official[j]), P(det6[j])
        cv2.arrowedLine(crop, pa, pb, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.drawMarker(crop, pa, (0, 0, 255), cv2.MARKER_SQUARE, 9, 2)
        cv2.circle(crop, pb, 6, (255, 0, 0), 2)
    cv2.putText(crop, 'red square = generic model (official pose) | blue circle = insightface',
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    # 放大子图：左眼 / 右眼 / 鼻
    panels = []
    regions = [('LEFT EYE', [33, 43]), ('RIGHT EYE', [87, 97]), ('NOSE', [72, 87])]
    for name, rng6 in regions:
        idxs = list(range(*rng6))
        pts = lm106[idxs]
        cx0 = int(pts[:, 0].min() - iod * 0.35); cx1 = int(pts[:, 0].max() + iod * 0.35)
        cy0 = int(pts[:, 1].min() - iod * 0.35); cy1 = int(pts[:, 1].max() + iod * 0.35)
        sub = img[cy0:cy1, cx0:cx1].copy()
        Z = 640 / sub.shape[1]
        sub = cv2.resize(sub, None, fx=Z, fy=Z)
        soff = np.array([cx0, cy0])
        Q = lambda p: tuple(((np.asarray(p, float) - soff) * Z).astype(int))
        for contour in EYE_CONTOUR:
            cv2.polylines(sub, [np.array([Q(lm106[i]) for i in contour])], True,
                          (150, 150, 150), 1, cv2.LINE_AA)
        for i in IRIS_CENTER:
            x, y = Q(lm106[i])
            cv2.line(sub, (x - 6, y), (x + 6, y), (150, 150, 150), 1, cv2.LINE_AA)
            cv2.line(sub, (x, y - 6), (x, y + 6), (150, 150, 150), 1, cv2.LINE_AA)
        for j, l in enumerate(LBL):
            hit = (l.split('_')[-1] in name.lower()) or (name == 'NOSE' and 'nose' in l)
            if not hit:
                continue
            pa, pb = Q(proj_official[j]), Q(det6[j])
            cv2.arrowedLine(sub, pa, pb, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)
            cv2.drawMarker(sub, pa, (0, 0, 255), cv2.MARKER_SQUARE, 11, 2)
            cv2.circle(sub, pb, 8, (255, 0, 0), 2)
            dy_mm = (det6[j][1] - proj_official[j][1]) / iod * 90
            cv2.putText(sub, f'dy={det6[j][1]-proj_official[j][1]:+.0f}px({dy_mm:+.1f}mm)',
                        (8, sub.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sub, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                    2, cv2.LINE_AA)
        panels.append(sub)

    ph = max(p.shape[0] for p in panels)
    canvas = np.full((crop.shape[0] + 40 + ph,
                      max(crop.shape[1], sum(p.shape[1] for p in panels) + 40), 3),
                     255, np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    xx = 0
    for p in panels:
        canvas[crop.shape[0] + 40:crop.shape[0] + 40 + p.shape[0],
               xx:xx + p.shape[1]] = p
        xx += p.shape[1] + 20
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / 'semantics_mismatch.png'), canvas)
    log.info(f'输出 {OUT / "semantics_mismatch.png"}')


if __name__ == '__main__':
    main()
