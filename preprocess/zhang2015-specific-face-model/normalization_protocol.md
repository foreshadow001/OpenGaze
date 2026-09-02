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
- landmarks 索引遍历、FailureRecorder、FailureRecorder 记录、BGR + lzf 等平台约定

## 6. h5 产物字段

所有数据集统一（BGR + lzf），逐字段说明：

| 字段 | 形状 | 类型 | 说明 |
|---|---|---|---|
| **face_patch** | (N,224,224,3) | uint8 | 归一化人脸图像（BGR，虚拟相机 960mm 焦距/600mm 距离） |
| **face_gaze** | (N,2) | float | 归一化相机系视线 (θ,φ) 弧度（CCS gaze，训练标签） |
| **face_gaze_hcs** | (N,2) | float | 头架系视线 (θ,φ) 弧度（HCS gaze，**与归一化无关**，跨相机不变量） |
| **face_head_pose** | (N,2) | float | 归一化系头姿 (pitch,yaw) **度数**（标准系约定：解缠绕 + −30° 零位，CLAUDE.md 约定 9） |
| **face_landmarks_3d** | (N,6,3) | float | **归一化前** 6 点在原始相机系中的 3D 坐标（mm），= hR·model + t |
| **face_mat_norm** | (N,3,3) | float | 归一化旋转矩阵 R_norm = hR_norm·hR^T |
| **facial_landmarks_2d** | (N,106,2) | float32 | 原始帧 106 点（归一化前的像素坐标） |
| frame_index | (N,) | int32 | 原始帧号 |
| cam_index | (N,) | int32 | 相机号（xgaze/eve 有） |
| step_index | (N,) | int32 | 刺激步序号（eve 有） |
| orientation | (N,) | int8 | 设备朝向 1-4（gc 有） |
| day_index | (N,) | int32 | 天序号（mpii 有） |
| image_name | (N,) | vlen str | 原始文件名（mpii 有） |

### h5 attrs（每文件一份）

| attr | 类型 | 说明 |
|---|---|---|
| face_model | (6,3) float | 归一化所用模型（true6_canonical 逐人 / gen_xe6 通用） |
| face_model_type | str | "true6_canonical" 或 "gen_xe6" |
| pipeline | str | "zhang2015-specific-face-model v2" |
| source | str | 原始数据路径前缀 |
| steps / cameras | str(json) | EVE 特有 |

### 设计考量

**为何存 face_landmarks_3d（归一化前 3D）**：最上游几何量——从它可推出
头姿（Kabsch(model, X_cam)）、人脸距离、归一化后 3D 位置
（S·R_norm·X_cam），但反向不可推。每样本 18 float（≈144B，可忽略）。

**为何存 face_gaze_hcs**：HCS = hR^T·gc 与归一化旋转严格无关（数学恒等式），
是跨相机不变量——跨数据集分析、头姿-视线解耦训练的核心量。推导需知晓
当前约定（is_true6 开关、解缠绕、−30° 零位），直接存储防约定漂移。

**为何存 face_head_pose（度数而非弧度）**：标准系约定是本平台特有定义，
直接存储避免训练侧引入 utils.normalization 依赖；度数在预处理端一次转换。

**不存归一化前 CCS 视线方向**：可由已有字段精确推出
v_cam = face_mat_norm^T · v_norm（纯旋转，一行代码，零信息损失）。

**face_model 存 attrs 而非逐样本**：模型是被试/session 级元数据（不随帧变），
每文件一份 (6,3) 零冗余；使 h5 完全自包含（分析无需读 sfm 盘）。

**为何存 face_head_pose（度数而非弧度）**：与 face_gaze（弧度）不同——头姿的
标准系约定（后向解缠绕 + −30° 零位）是本平台特有定义，直接存储计算结果
（度数）使训练侧无需引入 utils.normalization 依赖。弧度→度数的转换在预处理
端一次性完成。

**为何存 facial_landmarks_2d（原始坐标）**：归一化后的 landmarks 可由
face_mat_norm 反变换得到（原始 = 归一化 × inv(W)），但反变换需要知道原始
内参 K——而 K 不在 h5 中。存储原始坐标保留完整回溯能力（结合
face_mat_norm 可重建归一化后坐标），也用于下游特征点级 QC。

## 7. h5 attrs（回溯元信息）

| attr | 类型 | 说明 |
|---|---|---|
| steps | str(json) | EVE 刺激步名列表 |
| cameras | str(json) | EVE 相机名列表 |
| source | str | 原始数据路径前缀 |
| pipeline | str | "zhang2015-specific-face-model v2" |
| face_model | str | 使用的模型类型（true6_canonical / gen_xe6） |

> 备注：BA 时代的 `normalize_*.py` 端口已全部删除（2026-08-30），v2 正式归一化
> 实现按本文件重建（2026-08-31，`normalize_{xgaze,eve,gazecapture,mpiifacegaze}.py`）。
