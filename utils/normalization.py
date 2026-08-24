"""视线数据归一化管线纯函数（Zhang2015 虚拟相机归一化）

数据集无关的三个数学函数；face model、关键点索引、相机表等管线常量
由各预处理器独立配置（见 preprocess/zhang2015-insightface/）。
"""
import cv2
import numpy as np


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    """solvePnP 头姿估计（EPNP 初始化 + 迭代精化）"""
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion,
                                   flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion,
                                       rvec, tvec, True)
    return rvec, tvec


def normalizeData_face(img, face_model, rvec, tvec, gaze_point, camera_matrix, landmarks=None):
    """完整图像归一化流程: 把人脸对齐到虚拟相机, 输出归一化图像/头部姿态/视线方向.

    返回 [img_warped, hr_norm, gc_normalized, (landmarks_warped)].
    """
    focal_norm = 960          # 虚拟相机焦距
    distance_norm = 600       # 归一化后人脸中心到相机的距离
    roiSize = (224, 224)      # 归一化图像尺寸

    ht = tvec.reshape(3, 1)
    hR = cv2.Rodrigues(rvec)[0]          # 旋转矩阵
    gc = np.array(gaze_point).reshape(3, 1)
    face = face_model.reshape(6, 3).T    # (3,6), 每列一个三维点, 与官方 face_model.T 一致

    # 3D 人脸模型变换到相机坐标系, 求人脸中心
    Fc = np.dot(hR, face) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape(3, 1)
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape(3, 1)
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape(3, 1)

    distance = np.linalg.norm(face_center)          # 人脸中心到原相机的距离
    z_scale = distance_norm / distance
    cam_norm = np.array([
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    # 构建归一化旋转: forward 指向人脸中心, down 由 hRx 确定
    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))

    img_warped = cv2.warpPerspective(img, W, roiSize)

    # 归一化后的头部姿态
    hR_norm = np.dot(R, hR)
    hr_norm = cv2.Rodrigues(hR_norm)[0]

    # 归一化后的视线方向
    gc_normalized = np.dot(R, gc - face_center)
    gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

    data = [img_warped, hr_norm, gc_normalized]

    if landmarks is not None:
        pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 1, 2)
        landmarks_warped = cv2.perspectiveTransform(pts, W).reshape(-1, 2)
        data.append(landmarks_warped)

    return data


def vector_to_angles(v):
    """方向向量 -> (theta, phi) 弧度, 与官方 h5 的 face_gaze 约定一致."""
    theta = np.arcsin((-1) * v[1])
    phi = np.arctan2((-1) * v[0], (-1) * v[2])
    return theta, phi
