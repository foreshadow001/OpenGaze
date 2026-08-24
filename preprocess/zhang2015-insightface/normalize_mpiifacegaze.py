"""MPIIFaceGaze 全量归一化 -> 每人一个 h5（自 ~/data-preprocessing-gaze/normalize_mpiifacegaze_h5.py 迁移）

管线与 normalize_xgaze.py 相同（Zhang2015 虚拟相机归一化 + insightface 106 点
PnP + R 反解 + (theta,phi) 弧度 + 多线程流水线读图），只替换数据加载部分：
    XGaze 数据加载                ->  MPIIFaceGaze 数据加载
    subjectNNNN/frameNNNN/camNN   ->  pXX/dayYY/NNNN.jpg
    annotation_train/*.csv        ->  pXX/pXX.txt（28 列，见数据集 readme）
    camNN.xml (K, dist)           ->  pXX/Calibration/Camera.mat
    CSV 第 4~6 列 gaze_point      ->  pXX.txt 第 25~27 列 gt（相机系 3D 注视目标）

h5 字段（MPII 语义命名，其余与 XGaze 版同构）：
  day_index (N,) int32 | image_name (N,) string | face_patch (N,224,224,3) uint8
  face_mat_norm (N,3,3) | facial_landmarks_2d (N,106,2) | face_gaze (N,2) (theta,phi) 弧度
"""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger
from utils.normalization import estimateHeadPose, normalizeData_face, vector_to_angles

log = get_logger('preprocess.mpiifacegaze')

# ---- 管线常量（zhang2015-insightface，与 XGazePreprocessor 一致）----
# 3D face model 中 PnP 使用的点：双眼四角 + 两鼻角（对应 68 点的 36,39,42,45,48,54）
LANDMARK_USE = [20, 23, 26, 29, 15, 19]
# insightface 106 点中与 LANDMARK_USE 对应的索引
IDX6 = [35, 39, 89, 93, 78, 84]
# 本管线目录下的 3D face model
FACE_MODEL_USE = np.loadtxt(Path(__file__).parent / 'face_model_xgaze.txt') \
    .reshape(50, 1, 3)[LANDMARK_USE, :]


def load_camera(subject, raw_data_dir):
    """Calibration/Camera.mat -> (cameraMatrix, distCoeffs)"""
    c = sio.loadmat(Path(raw_data_dir) / subject / "Calibration" / "Camera.mat")
    return (np.asarray(c["cameraMatrix"], dtype=float),
            np.asarray(c["distCoeffs"], dtype=float).reshape(1, 5))


def load_annotations(subject, raw_data_dir):
    """pXX.txt -> 按天分组的样本列表 [(day_str, [(img_id, img_path, gt), ...]), ...].

    28 列布局(readme): 0=路径, 1~2 屏幕点, 3~14 六个 2D 关键点, 15~17 rvec,
    18~20 tvec, 21~23 fc, 24~26 gt(相机系 3D 注视目标), 27 评估用眼.
    """
    by_day = {}
    with open(Path(raw_data_dir) / subject / f"{subject}.txt") as f:
        for line in f:
            p = line.split()
            if len(p) < 27 or "/" not in p[0]:
                continue
            day, name = p[0].split("/")
            img_id = int(name.split(".")[0])
            gt = [float(x) for x in p[24:27]]
            by_day.setdefault(day, []).append(
                (img_id, Path(raw_data_dir) / subject / p[0], gt))
    return sorted(by_day.items())


def process_subject(subject, app, config, recorder):
    """单个受试者完整归一化 -> output_dir/pXX.h5, 返回写入样本数."""
    camera_matrix, distortion = load_camera(subject, config.raw_data_dir)
    day_groups = load_annotations(subject, config.raw_data_dir)
    if getattr(config, 'max_days', 0):
        day_groups = day_groups[:config.max_days]   # 调试：只处理前 N 天
    n_rows = sum(len(rows) for _, rows in day_groups)
    log.info(f"{subject}: {len(day_groups)} 天, {n_rows} 张标注图")

    out_path = Path(config.output_dir) / f"{subject}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(out_path, "w")
    dsets = {
        "day_index": h5.create_dataset("day_index", (n_rows,), np.int32,
                                       chunks=(1,), maxshape=(None,)),
        "image_name": h5.create_dataset("image_name", (n_rows,),
                                        h5py.string_dtype(encoding="utf-8"),
                                        chunks=(1,), maxshape=(None,)),
        "face_patch": h5.create_dataset("face_patch", (n_rows, 224, 224, 3), np.uint8,
                                        chunks=(1, 224, 224, 3),
                                        maxshape=(None, 224, 224, 3), compression="lzf"),
        "face_mat_norm": h5.create_dataset("face_mat_norm", (n_rows, 3, 3), float,
                                           chunks=(1, 3, 3), maxshape=(None, 3, 3)),
        "facial_landmarks_2d": h5.create_dataset("facial_landmarks_2d", (n_rows, 106, 2),
                                                 np.float32, chunks=(1, 106, 2),
                                                 maxshape=(None, 106, 2)),
        "face_gaze": h5.create_dataset("face_gaze", (n_rows, 2), float,
                                       chunks=(1, 2), maxshape=(None, 2)),
    }

    n = 0
    t_sub = time.time()

    # 读盘-处理流水线(单位=天): N 线程读"下一天", 主线程处理"当前天"
    q = queue.Queue(maxsize=2)

    def read_days():
        try:
            with ThreadPoolExecutor(max_workers=config.num_read_workers) as ex:
                for day, rows in day_groups:
                    imgs = dict(zip([r[0] for r in rows], ex.map(
                        lambda r: cv2.imread(str(r[1])), rows)))
                    q.put((day, rows, imgs))
            q.put(None)
        except Exception as e:  # 读线程异常传回主线程
            q.put(e)

    threading.Thread(target=read_days, daemon=True).start()

    pbar = tqdm(total=len(day_groups), desc=subject, unit="天",
                ncols=100, leave=False)
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            day, rows, imgs = item
            day_id = int(day[3:])
            n_ok = 0
            for img_id, img_path, gt in rows:
                img = imgs.get(img_id)
                if img is None:
                    recorder.add(subject, img_path.name, 'imread_failed')
                    continue
                faces = app.get(img)
                if len(faces) == 0:
                    recorder.add(subject, f"{day}/{img_path.name}", 'no_face_detected')
                    continue
                lm106 = faces[0].landmark_2d_106
                landmarks2d = lm106[IDX6, :].reshape(6, 1, 2).astype(float)
                rvec, tvec = estimateHeadPose(landmarks2d, FACE_MODEL_USE,
                                              camera_matrix, distortion)
                img_warped, hr_norm, gc_normalized = normalizeData_face(
                    img, FACE_MODEL_USE, rvec, tvec, gt, camera_matrix)[:3]
                # hR_norm = R @ hR  =>  R = hR_norm @ hR^T (旋转矩阵逆=转置)
                R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
                g_theta, g_phi = vector_to_angles(gc_normalized.flatten())

                dsets["day_index"][n] = day_id
                dsets["image_name"][n] = img_path.name
                dsets["face_patch"][n] = img_warped
                dsets["face_mat_norm"][n] = R
                dsets["facial_landmarks_2d"][n] = lm106
                dsets["face_gaze"][n] = (g_theta, g_phi)
                n += 1
                n_ok += 1
            pbar.update(1)
            pbar.set_postfix({"天": day, "入库": f"{n_ok}/{len(rows)}", "样本": n})
    finally:
        pbar.close()
        for d in dsets.values():
            d.resize((n,) + d.shape[1:])
        h5.close()

    log.info(f"{subject} 完成: {n} 样本 -> {out_path.name} "
             f"({(time.time() - t_sub) / 60:.1f} min)")
    return n


def run(config, recorder):
    """preprocess.py 入口：按配置处理全部（或指定）受试者，返回总样本数"""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis("buffalo_l", allowed_modules=["detection", "landmark_2d_106"],
                       providers=config.providers)
    app.prepare(ctx_id=0)

    raw_dir = Path(config.raw_data_dir)
    subjects = config.subjects if config.subjects else sorted(
        d.name for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("p"))
    log.info(f"MPIIFaceGaze 共 {len(subjects)} 个受试者: {subjects}")
    log.info(f"输出目录: {config.output_dir}")
    total_n = 0
    t_all = time.time()
    overall = tqdm(subjects, desc="MPII 全集", unit="人", ncols=100, leave=True)
    for subject in overall:
        try:
            total_n += process_subject(subject, app, config, recorder)
        except Exception as e:  # 单人失败不中断整体，但记录在案
            log.error(f"[错误] {subject}: {type(e).__name__}: {e}")
            recorder.add(subject, '<subject>', f'error:{type(e).__name__}:{e}')
        overall.set_postfix({"累计样本": total_n})
    overall.close()
    log.info(f"全部完成: {len(subjects)} 人, 共 {total_n} 样本, "
             f"总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    return total_n
