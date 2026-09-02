# ours-without-roll（v3）归一化协议（2026-09-02 定稿）

一句话：v3 = 最后一版预处理。与 v2 唯一的核心区别是归一化函数取
`normalizeData_face(fixed_forward=True)`——虚拟相机光轴固定为原相机 z 轴，
**只消除 roll，头部 pitch/yaw 归一化前后严格不变**；训练目标改为 HCS 视线。

## 1. 动机与定位

| | v2（fixed_forward=False） | v3（fixed_forward=True） |
|---|---|---|
| 虚拟相机光轴 | 指向人脸中心 | **= 原相机 z 轴**（固定） |
| patch 中头姿 | pitch/yaw 被归一化消掉（虚拟相机随头转） | **pitch/yaw 保留在图像里，仅双眼水平化（roll=0）** |
| 主点 | 人脸中心天然在光轴上 | 平移主点使人脸中心投影在图像中心 |
| gaze 标签语义 | 相机系视线（含头姿，头已被消掉故等价于头相对相机的量） | CCS=roll 修正后相机系（含头姿）；**训练用 HCS**（eye-in-head） |
| 训练表述 | 网络从"头姿≈0"的 patch 直接回归相机系视线 | 网络从保留头姿的 patch 预测**头架系视线**（头姿-视线解耦的另一半） |

`face_gaze_hcs` 参与训练（loader/method 侧启用，见 §7）；本协议只负责产物正确。

## 2. 数学定义（fixed_forward=True；实现已是 utils/normalization.py 现有代码，无需改动）

构造（与 `normalizeData_face` 完全一致）：

```
forward = (0,0,1)                      # 原相机 z 轴
down    = forward × hRx → 归一          # hRx = hR 第一列（眼线方向，相机系）
right   = down × forward → 归一
R_v3    = [right; down; forward]（行向量）
hR_norm = R_v3 · hR
gc_norm = R_v3 · (gp − face_center) → 归一
```

**pitch/yaw 不变、roll=0 的证明**（已数值验证到机器精度 1e-16）：
设 hR 的欧拉分解（extrinsic xyz）hR = Rz(γ)·Ry(β)·Rx(α)，则第一列
hRx = (cosγcosβ, sinγcosβ, −sinβ)，于是
down ∝ (−sinγ, cosγ, 0)、right = (cosγ, sinγ, 0)，即
**R_v3 = Rz(−γ)**，故

```
hR_norm = Rz(−γ)·Rz(γ)·Ry(β)·Rx(α) = Ry(β)·Rx(α)
→ pitch=α、yaw=β 与归一化前逐位一致，roll≡0
```

**HCS 不变**（数学恒等式，同样验证到 1e-16）：
`hR_norm^T·gc_norm = hR^T·(gp−face_center)`——与归一化旋转无关，
**v3 的 face_gaze_hcs 与 v2 数值恒等**。

**退化条件**：|forward×hRx| = cosβ → β=±90°（眼线与光轴重合，如 xgaze 侧视
相机 cam06/07/15 看正脸被试）时 down 病态、roll 不稳。v1/v2 的 down 构造
同式（其 forward=人脸中心方向 ≈ z_cam），**非 v3 新增问题**；极端侧视样本
的 roll 本就由被剧烈缩短的眼线投影决定，任何 roll 消除方案皆然。

## 3. 加速：从 v2 产物精确恢复几何（无 DLT / 无 PnP / 无 insightface）

每样本的全部归一化输入均可由 v2 h5 **精确**恢复（已核对四数据集 v2 产物
字段齐全），只有读图需要 I/O：

| v3 输入 | v2 来源 | 恢复方法 | 精度 |
|---|---|---|---|
| hR, t（模型→相机） | `face_landmarks_3d` | Kabsch(model, X_cam)；X_cam = hR·model+t 的原值，6 点非共面刚体反解 | ~1e-12（浮点级） |
| gp 方向（相机系，由 face_center 指向注视点） | `face_gaze` + `face_mat_norm` | v = face_mat_norm^T · unit(θ,φ)；face_mat_norm = hR_norm·hR^T = R_v2 恰是 v2 的归一化旋转 | 精确（单位向量↔角度可逆） |
| face_center | `face_landmarks_3d` | 与 normalizeData_face 同式的 6 点加权中心 | 精确 |
| K（原始内参） | 静态标定（不入 h5） | xgaze：cam xml；EVE：step h5 camera_matrix；GC：orientation 查表 xml；MPII：Calibration/Camera.mat | 官方值 |
| 原始图像 | v2 索引字段 | xgaze：frame/cam → train/…/camXX.JPG；EVE：frame/cam/step → mp4（step 分组顺序读，同 v2）；GC：frame+orientation（ori 2/4 旋转同 v2）；MPII：day_index+image_name | — |

调 `normalizeData_face` 时 gp 传 `face_center + v`（函数内部只用
gp−face_center 的方向，尺度无关）。

**逐图独立**：样本间无跨相机/跨帧依赖 → 可按图/按被试/session 多进程并行，
I/O 与计算完全重叠。预期瓶颈只剩读图（xgaze HDD、EVE mp4 解码）。

**索引同构**：v3 与 v2 的 split/文件名/样本顺序逐一相同（索引字段原样复制），
v2↔v3 可逐样本对齐对照。

## 4. h5 产物字段

| 字段 | 形状 | 类型 | v3 语义 |
|---|---|---|---|
| face_patch | (N,224,224,3) | uint8 | fixed_forward=True 的 roll-only warp（BGR+lzf，虚拟相机 960 焦距/600mm 距离） |
| face_gaze | (N,2) | float | v3 归一化相机系视线 (θ,φ) 弧度（roll 已修正、**头姿保留**） |
| **face_gaze_hcs** | (N,2) | float | 头架系视线 (θ,φ) 弧度——**训练目标**；与 v2 数值恒等（兼作跨版一致性校验字段） |
| **face_head_elev_azim** | (N,2) | float | **替代 v2 的 face_head_pose**：模型 −z 轴在世界系的 (elev, azim) 度（定义见下） |
| face_landmarks_3d | (N,6,3) | float | 相机系 3D（与归一化无关）——**原值复制自 v2** |
| face_mat_norm | (N,3,3) | float | R_v3（纯 roll 旋转 Rz(−γ)） |
| facial_landmarks_2d | (N,106,2) | float32 | 原始帧像素坐标——原值复制自 v2 |
| frame/cam/step_index、orientation、day_index、image_name | — | — | 索引字段按各数据集 v2 原样复制 |

attrs 同 v2（face_model、face_model_type、EVE 另有 steps/cameras），
`pipeline = "ours-without-roll v3"`。

### face_head_elev_azim 定义（世界系，替代 face_head_pose 的理由）

```
v_w  = R_head_world · [0,0,−1]        # 模型 −z 轴（下颌方向）在世界系
elev = degrees(arcsin(clip(−v_w[1]))) − 30     # 标准系零位（HEAD_PITCH_OFFSET）
azim = degrees(atan2(−v_w[0], −v_w[2]))
```

R_head_world 的逐数据集来源（均为已知量，无需 DLT；多相机数据集同帧各相机
反解结果一致）：

| 数据集 | R_head_world | 说明 |
|---|---|---|
| xgaze | ROT_c^T · hR | 官方穹顶系（= cam00 系，其外参恰为 I/0） |
| EVE | R_basler · Rs_c^T · hR | **basler 相机系**（cam 0，2026-09-02 定；basler 样本自身退化为单位变换） |
| GC / MPII | hR | 单目，世界=唯一相机 |

替换理由：
1. v3 归一化不再改变 pitch/yaw——若仍存"归一化系头姿"，它 ≡ 原相机系 2 角，
   cam13 类极点拖尾问题（[viz/distribution/exception/pitch60_tail](../zhang2015-specific-face-model/viz/distribution/exception/pitch60_tail/README.md)）复现；
2. 世界系坐标卡对多相机唯一、极点在竖直方向，四数据集头姿远离极点；
3. 无信息损失——任何 per-camera 角度都可由 face_landmarks_3d（→hR）+ 标定外参
   随时重导。

## 5. 与 v2 一致的部分

人脸模型（xgaze/EVE 逐人 true6_canonical、GC/MPII 通用 gen_xe6）、视线标注
链路（各官方源）、landmarks 索引遍历、FailureRecorder、BGR+lzf、图像读取
方式（含 EVE step 分组 mp4 顺序读、GC ori 2/4 的 180° 旋转）全部不变；
`face_landmarks_3d` / `facial_landmarks_2d` / `face_gaze_hcs` 与 v2 数值相同
（复制 / 恒等）。

## 6. 冒烟验收（每数据集小样本）

1. **欧拉不变性**：hR_norm 的欧拉 (α,β) == hR 的 (α,β)（≤1e-9°），γ≡0；
2. **HCS 恒等**：v3 face_gaze_hcs == v2 face_gaze_hcs（≤1e-6 rad）；
3. face_landmarks_3d / facial_landmarks_2d / 索引字段与 v2 逐位一致；
4. **世界系一致**：xgaze/EVE 同帧多相机反解的 (elev, azim) 相同（≤1e-6°）；
5. patch 目视：头姿俯仰/偏航与原图一致、双眼水平（roll=0）。

## 7. 工程约定

- 脚本：`preprocess/ours-without-roll/normalize_<ds>.py`（暴露 `run(config, recorder)`，
  共享恢复链 `v3_core.py`），入口 `python preprocess.py --dataset <ds> --method ours-without-roll`；
- 配置：`configs/preprocess/ours-without-roll/<ds>.yaml`（数据源 = v2 产物路径
  `v2_dir` + 原始数据路径 + 静态标定路径）；
- 输出：sfm 盘 `<ds>_noroll_224`（与 v2 的 `*_specific_224` 并列）；
- 训练侧（另行开发）：datasets loader 提供 face_gaze_hcs 作为标签源
  （method 配置选择 gaze 目标 = CCS / HCS），训练/评测协议沿用平台约定。

> 冒烟验收（2026-09-02，四数据集各 1 被试/session 小样本）：§6 全部通过——
> 索引/复制字段与 v2 逐位一致，HCS 与 v2 差 ≤4e-15 rad，欧拉 (α,β) norm==raw
> ≤2.4e-13°、norm roll ≤6.4e-14°，xgaze 世界系同帧跨相机一致 2e-13°，
> CCS 与 v2 差 0.09–0.45 rad（roll 修正所致，预期非零）。

> 备注：v3 不重跑任何 v2 几何（模型、DLT、PnP、门控全部继承 v2 产物）；
> v2 数据若重跑（如门控/模型变更），v3 需级联重跑。
