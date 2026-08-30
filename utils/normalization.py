"""视线数据归一化管线纯函数（Zhang2015 虚拟相机归一化 + 人脸模型标准系）

数据集无关的数学函数；face model、关键点索引、相机表等管线常量
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


def normalizeData_face(img, face_model, rvec, tvec, gaze_point, camera_matrix,
                       landmarks=None, fixed_forward=False):
    """完整图像预处理流程: 把人脸对齐到虚拟相机, 输出归一化图像/头部姿态/视线方向.

    fixed_forward=False: forward 指向人脸中心(官方归一化);
    fixed_forward=True : forward 固定为 [0,0,1](虚拟相机光轴=原相机 z 轴),
                         并平移主点加偏移量, 使人脸中心始终落在归一化图像中心.
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

    # 构建归一化旋转: down 由 hRx 确定
    hRx = hR[:, 0]
    if fixed_forward:
        forward = np.array([0.0, 0.0, 1.0])   # 虚拟相机光轴 = 原相机 z 轴
    else:
        forward = (face_center / distance).reshape(3)  # 指向人脸中心
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    if fixed_forward:
        # 人脸中心不再位于光轴上, 平移主点(加偏移量)使其投影落在图像中心
        fc_n = np.dot(R, face_center).ravel()
        u_c = focal_norm * fc_n[0] / (z_scale * fc_n[2]) + roiSize[0] / 2
        v_c = focal_norm * fc_n[1] / (z_scale * fc_n[2]) + roiSize[1] / 2
        cam_norm[0, 2] += roiSize[0] / 2 - u_c
        cam_norm[1, 2] += roiSize[1] / 2 - v_c

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


def canonicalize_face_model(P6):
    """6 点人脸模型 → 解剖轴标准坐标系（唯一实现；CLAUDE.md 约定 9）。

    零位标准（解剖轴定义，2026-08-29 定稿）：
      roll  = 0 ⇔ 两眼中心连线 ∥ x 轴
      yaw   = 0 ⇔ 眼心—鼻心连线 ⊥ x 轴（无 x 分量）
      pitch = 0 ⇔ 眼心—鼻心连线 ∥ y 轴

    构造：x̂ = eye_R − eye_L（眼心连线）归一；
    ŷ = (nose_c − eye_c) 去除 x̂ 分量后归一；ẑ = x̂ × ŷ（右手系，指向头内，
    与 OpenCV 相机系一致）；原点 = 眼心 eye_c。

    P6 点序（insightface IDX6 / 6 点人脸模型通用序）：
      [eye_out_L, eye_in_L, eye_in_R, eye_out_R, nose_L, nose_R]

    模型构建三步管线（所有个性化/通用模型交付前必经，防坐标系漂移）：
      ① 粗对齐：Kabsch 刚体对齐到 gen6（消头运动/初始朝向，无缩放）；
      ② 欧拉角归零：本函数，三个欧拉角归到解剖零位；
      ③ 中心化：原点平移到眼心（②③合并于本函数：P_std = R @ (P − eye_c)）。

    返回 (P6_std, R, origin)：标准系 6 点 (6,3)；R 行向量 = 标准系三轴在
    原坐标系中的方向；origin = 眼心。
    """
    P6 = np.asarray(P6, dtype=float).reshape(6, 3)
    eye_L = (P6[0] + P6[1]) / 2
    eye_R = (P6[2] + P6[3]) / 2
    eye_c = (eye_L + eye_R) / 2
    nose_c = (P6[4] + P6[5]) / 2
    x = eye_R - eye_L
    x = x / np.linalg.norm(x)
    d = nose_c - eye_c
    y = d - (d @ x) * x          # 去 x 分量：yaw=0 约束
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)           # 右手系：pitch/roll 零位由 x̂/ŷ 唯一确定
    R = np.stack([x, y, z])
    return (R @ (P6 - eye_c).T).T, R, eye_c
