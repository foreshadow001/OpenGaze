# zhang2015-specific-face-model（v2）vs zhang2015-insightface（v1）关键区别

一句话：v1 四数据集统一「通用 gen6 + 单目 PnP」；v2 按有无多相机分档——xgaze/EVE
用「逐人真实模型 true6_canonical + DLT 三角化」直接解 3D 几何，GC/MPII 保留但
降档为「通用 gen_xe6 + 单目 PnP」。

## 1. 人脸模型

| | v1 | v2 |
|---|---|---|
| 逐人模型 | 无（所有人同一副） | **true6_canonical**：全帧 × 多相机 × 官方外参 DLT 三角化 → 逐帧 Kabsch 消头运动取中位 → 标准化（交付 `sfm/<ds>_specific_face_model/face_models/<id>/true6_canonical.txt`） |
| 通用回退 | gen6（`face_model_xgaze.txt` 6 点子集） | **gen_xe6**（xgaze 80 + EVE 44 的 true6 均值再标准化，[get_face_model/gen_xe6_canonical.txt](get_face_model/gen_xe6_canonical.txt)） |
| 依据 | — | DLT 真值实证 gen6 鼻宽 +4.6mm 等几何缺陷；换 gen_xe6 后 GC/MPII 的 HCS yaw 中位由 −6.3° 归零 |

## 2. 头部姿态

| | v1 | v2 |
|---|---|---|
| 方法 | 每帧每相机 6 点 PnP（`estimateHeadPose`） | xgaze/eve：多相机 DLT 3D 点 → Kabsch 拟合逐人模型（**不走 PnP**） |
| 精度 | PnP 3D 位置误差 ~53mm（xgaze 极端相机更大） | DLT 3D 误差 ~5mm；跨相机 HCS 一致性 xgaze **0.01°**、EVE **~2°**（webcam 时钟同步残差，官方明言仅 basler 可靠） |

## 3. 坐标系约定

- v1：gen6 原生系（头姿零位 = 官方模型自带坐标轴）
- v2：**标准系**（解剖轴，CLAUDE.md 约定 9；pitch=0 ⇔ 眼→鼻连线空间竖直，非自然平视——自然平视鼻线前倾 ~13.2°，故常态读 +1~+16°）
- 两系相差 13.46°：跨管线比较头姿读数差 ~13° 属坐标系修正而非 bug；**gaze 与归一化图像不受影响**（HCS = hRᵀ·gc 与归一化旋转严格无关）

## 4. 范围与产物

- v1：四数据集全量 → ylx 盘 `*_insightface_224`
- v2：**四数据集全部保留** → sfm 盘 `*_specific_224`，分两档：
  - xgaze / EVE：逐人 true6_canonical + DLT 头姿（上表）
  - GC / MPII：无多相机，逐 session 个性化建模已证不可行（单目 6 点 PnP 平面
    二义、GC 缺官方内参、MediaPipe 系统性低估 pitch，探索已否决并清理），改用
    **通用 gen_xe6 + 单目 6 点 PnP**——gen6→gen_xe6 后 HCS yaw 中位由 −6.3° 归零，
    精度可接受

## 5. 两版一致的部分

- 视线标签链路：各官方标注源（xgaze 注释 3D 点 / MPII pXX.txt 列 24-26 / GC dot 链 / EVE PoG 屏幕 px）+ `normalizeData_face(fixed_forward=False)` 标准 Zhang 归一化
- landmarks 索引遍历、FailureRecorder、h5 格式（BGR、224、face_gaze (pitch,yaw) 弧度）等平台约定

> 备注：BA 时代的 `normalize_*.py` 端口已全部删除（2026-08-30），v2 正式归一化
> 实现按本文件重建。
