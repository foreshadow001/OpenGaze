"""ETH-XGaze 原始数据归一化 -> 每人一个 h5（自 ~/data-preprocessing-gaze/normalize_subject_h5.py 迁移）

管线：Zhang2015 虚拟相机归一化（utils/normalization.normalizeData_face）+
insightface 106 关键点 PnP 头姿 + R 反解 + (theta,phi) 弧度。
本模块封装为 XGazePreprocessor 类，管线常量独立配置（LANDMARK_USE / IDX6 /
FLIP_CAMERAS / EXCLUDE_CAMERAS），face model 从本目录独立加载。

h5 字段（官方预处理格式子集）：
  frame_index (N,1) int32 | cam_index (N,1) int32 | face_patch (N,224,224,3) uint8
  face_mat_norm (N,3,3) | facial_landmarks_2d (N,106,2) | face_gaze (N,2) (theta,phi) 弧度

R 由恒等式 hR_norm = R @ hR 反解: R = Rodrigues(hr_norm) @ Rodrigues(rvec).T。
"""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger
from utils.normalization import estimateHeadPose, normalizeData_face, vector_to_angles

log = get_logger('preprocess.xgaze')


class XGazePreprocessor:
    """ETH-XGaze 数据预处理器（Zhang2015 归一化 + insightface 关键点）"""

    # 3D face model 中 PnP 使用的点：双眼四角 + 两鼻角
    # （对应 68 点标注的 36,39,42,45,48,54）
    LANDMARK_USE = [20, 23, 26, 29, 15, 19]
    # insightface 106 点中与 LANDMARK_USE 对应的索引
    IDX6 = [35, 39, 89, 93, 78, 84]
    # 原始数据中倒置的相机(0-based), 处理前需 180° 旋转
    FLIP_CAMERAS = [3, 6, 13]
    # 排除不参与处理的相机(0-based), 直接跳过, 不检测
    EXCLUDE_CAMERAS = []
    NUM_CAMS = 18

    def __init__(self, config):
        self.config = config
        self.cams = [c for c in range(self.NUM_CAMS)
                     if c not in self.EXCLUDE_CAMERAS]
        # 独立加载本管线目录下的 3D face model ((3,6) x 50 行文本 -> 50 个三维点)
        face_model = np.loadtxt(Path(__file__).parent / 'face_model_xgaze.txt')
        self.face_model_use = face_model.reshape(50, 1, 3)[self.LANDMARK_USE, :]
        # 预载全部相机标定
        self.calibs = {c: self._load_camera_calibration(c)
                       for c in range(self.NUM_CAMS)}

    def _load_camera_calibration(self, cam_index):
        """读取 ETH-XGaze 相机标定 xml -> (cameraMatrix, distortion)"""
        xml_path = Path(self.config.calib_dir) / f"cam{cam_index:02d}.xml"
        fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("Camera_Matrix").mat()
        distortion = fs.getNode("Distortion_Coefficients").mat()
        fs.release()
        return camera_matrix, distortion

    # ------------------------------------------------------------- 单被试处理
    def load_annotations(self, subject_index):
        """一次性解析标注 CSV -> {(frame, cam): gaze_point}"""
        csv_path = Path(self.config.annotation_dir) / f"subject{subject_index:04d}.csv"
        gt = {}
        with open(csv_path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 7 or not p[0].startswith("frame"):
                    continue
                fr = int(p[0][5:])
                cm = int(p[1][3:5])
                gt[(fr, cm)] = [float(p[4]), float(p[5]), float(p[6])]
        return gt

    def process_subject(self, subject_index, app, recorder: FailureRecorder):
        """单个受试者完整归一化 -> output_dir/subjectNNNN.h5, 返回写入样本数."""
        subject_dir = Path(self.config.raw_data_dir) / self.config.sub_folder \
            / f"subject{subject_index:04d}"
        gaze_pts = self.load_annotations(subject_index)
        frame_dirs = sorted(d for d in subject_dir.iterdir()
                            if d.name.startswith("frame"))
        if getattr(self.config, 'max_frames', 0):
            frame_dirs = frame_dirs[:self.config.max_frames]  # 调试：只处理前 N 帧
        log.info(f"subject{subject_index:04d}: {len(frame_dirs)} 帧 "
                 f"x {len(self.cams)} 相机, 标注 {len(gaze_pts)} 条")

        out_path = Path(self.config.output_dir) / f"subject{subject_index:04d}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_max = len(frame_dirs) * self.NUM_CAMS
        h5 = h5py.File(out_path, "w")
        dsets = self._create_datasets(h5, n_max)

        n = 0
        t_sub = time.time()

        # 读盘-处理流水线: 后台线程读"下一帧"(读与解码并行),
        # 主线程同时处理"当前帧"(检测/PnP/归一化/写 h5), 队列深度 2
        q = queue.Queue(maxsize=2)

        def read_frames():
            try:
                with ThreadPoolExecutor(
                        max_workers=self.config.num_read_workers) as ex:
                    for fd in frame_dirs:
                        imgs = dict(zip(self.cams, ex.map(
                            lambda c: cv2.imread(str(fd / f"cam{c:02d}.JPG")),
                            self.cams)))
                        q.put((fd, imgs))
                q.put(None)
            except Exception as e:  # 读线程异常传回主线程, 避免主线程卡在 q.get()
                q.put(e)

        threading.Thread(target=read_frames, daemon=True).start()

        pbar = tqdm(total=len(frame_dirs), desc=f"subject{subject_index:04d}",
                    unit="帧", ncols=100, leave=False)
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                fd, imgs = item
                frame_index = int(fd.name[5:])
                n_ok = 0
                for cam in self.cams:
                    img = imgs.get(cam)
                    if img is None:
                        recorder.add(subject_index,
                                     f"frame{frame_index:04d}/cam{cam:02d}",
                                     'imread_failed')
                        continue
                    if cam in self.FLIP_CAMERAS:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    faces = app.get(img)
                    if len(faces) == 0:
                        recorder.add(subject_index,
                                     f"frame{frame_index:04d}/cam{cam:02d}",
                                     'no_face_detected')
                        continue
                    lm106 = faces[0].landmark_2d_106
                    gaze_point = gaze_pts.get((frame_index, cam))
                    if gaze_point is None:
                        recorder.add(subject_index,
                                     f"frame{frame_index:04d}/cam{cam:02d}",
                                     'no_annotation')
                        continue
                    camera_matrix, distortion = self.calibs[cam]
                    landmarks2d = lm106[self.IDX6, :].reshape(6, 1, 2).astype(float)
                    rvec, tvec = estimateHeadPose(
                        landmarks2d, self.face_model_use, camera_matrix, distortion)
                    img_warped, hr_norm, gc_normalized = normalizeData_face(
                        img, self.face_model_use, rvec, tvec, gaze_point,
                        camera_matrix)[:3]
                    # hR_norm = R @ hR  =>  R = hR_norm @ hR^T (旋转矩阵逆=转置)
                    R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
                    g_theta, g_phi = vector_to_angles(gc_normalized.flatten())

                    dsets["frame_index"][n] = frame_index
                    dsets["cam_index"][n] = cam
                    dsets["face_patch"][n] = img_warped
                    dsets["face_mat_norm"][n] = R
                    dsets["facial_landmarks_2d"][n] = lm106
                    dsets["face_gaze"][n] = (g_theta, g_phi)
                    n += 1
                    n_ok += 1
                pbar.update(1)
                pbar.set_postfix({"帧": f"{frame_index:04d}",
                                  "入库": f"{n_ok}/{len(self.cams)}", "样本": n})
        finally:
            pbar.close()
            for d in dsets.values():
                d.resize((n,) + d.shape[1:])
            h5.close()

        log.info(f"subject{subject_index:04d} 完成: {n} 样本 -> {out_path.name} "
                 f"({(time.time() - t_sub) / 60:.1f} min)")
        return n

    @staticmethod
    def _create_datasets(h5, n_max):
        return {
            "frame_index": h5.create_dataset("frame_index", (n_max, 1), np.int32,
                                             chunks=(1, 1), maxshape=(None, 1)),
            "cam_index": h5.create_dataset("cam_index", (n_max, 1), np.int32,
                                           chunks=(1, 1), maxshape=(None, 1)),
            "face_patch": h5.create_dataset("face_patch", (n_max, 224, 224, 3),
                                            np.uint8, chunks=(1, 224, 224, 3),
                                            maxshape=(None, 224, 224, 3),
                                            compression="lzf"),
            "face_mat_norm": h5.create_dataset("face_mat_norm", (n_max, 3, 3),
                                               float, chunks=(1, 3, 3),
                                               maxshape=(None, 3, 3)),
            "facial_landmarks_2d": h5.create_dataset(
                "facial_landmarks_2d", (n_max, 106, 2), np.float32,
                chunks=(1, 106, 2), maxshape=(None, 106, 2)),
            "face_gaze": h5.create_dataset("face_gaze", (n_max, 2), float,
                                           chunks=(1, 2), maxshape=(None, 2)),
        }

    # ------------------------------------------------------------------ 入口
    def run(self, recorder: FailureRecorder):
        """按配置处理全部（或指定）受试者，返回总样本数"""
        from insightface.app import FaceAnalysis
        app = FaceAnalysis("buffalo_l",
                           allowed_modules=["detection", "landmark_2d_106"],
                           providers=self.config.providers)
        app.prepare(ctx_id=0)

        dataset_path = Path(self.config.raw_data_dir) / self.config.sub_folder
        subjects = self.config.subjects if self.config.subjects else sorted(
            int(d.name[7:]) for d in dataset_path.iterdir()
            if d.name.startswith("subject"))

        log.info(f"{self.config.sub_folder} 共 {len(subjects)} 个受试者: {subjects}")
        log.info(f"输出目录: {self.config.output_dir}")
        total_n = 0
        t_all = time.time()
        overall = tqdm(subjects, desc="xgaze 全集", unit="人", ncols=100,
                       leave=True)
        for subject_index in overall:
            try:
                total_n += self.process_subject(subject_index, app, recorder)
            except Exception as e:  # 单人失败不中断整体，但记录在案
                log.error(f"[错误] subject{subject_index:04d}: "
                          f"{type(e).__name__}: {e}")
                recorder.add(subject_index, '<subject>',
                             f'error:{type(e).__name__}:{e}')
            overall.set_postfix({"累计样本": total_n})
        overall.close()
        log.info(f"全部完成: {len(subjects)} 人, 共 {total_n} 样本, "
                 f"总耗时 {(time.time() - t_all) / 3600:.2f} 小时")
        return total_n


def run(config, recorder):
    """preprocess.py 入口适配：构建预处理器并执行"""
    return XGazePreprocessor(config).run(recorder)
