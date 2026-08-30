# EVE 数据集探索报告与预处理方案

> 探索日期：2026-08-25。数据：`/media/hitsz/ylx/EVE_dataset`（180G），官方代码：`/home/hitsz/EVE`。
> 本文所有数字均为实测（脚本抽样验证，非文档转述）；标注"已验证"的结论有对拍实验支撑。

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 可用数据 | train01–39 + val01–05（44 人，有 Tobii GT）；**test01–06 无 GT 字段**，不可用于评测 |
| 相机 | 4 相机全为 1920×1080（basler 60fps，三 webcam 30fps），mp4 **已去畸变** |
| 标签几何 | PoG 屏幕像素 → 相机系 3D 点的转换链路**已验证到 1.2e-07 rad**（§4） |
| 角度约定 | 官方 `face_g_tobii` 与本平台 `vector_to_angles` **完全一致**（已对拍） |
| 预处理方案 | **4 相机全用**（basler + 3 webcam）× **5Hz**（basler 每 12 帧、webcam 每 6 帧）、insightface + Zhang2015 归一化（与 GazeCapture 同构）；**已实现并冒烟验证** |
| 规模估算 | ~107.5 万候选 → ~100 万样本；冒烟实测 ~60 帧/s → 全量 **~5 h**；~148 GB（face_patch 已启用 lzf 压缩，实际应更低） |
| 阻塞（已解除） | `/media/hitsz/ylx` 已满 → 输出改存 **`/home/hitsz/dataset/eve_insightface_224/`**（nvme 659G 空闲；ylx 保留原始数据只读） |

## 1. 数据集概况

EVE（ETH Zurich，2020）：54 名被试观看 25 英寸屏幕上的 image / video / wikipedia 刺激，
Tobii Pro Spectrum 眼动仪提供视线 GT，4 个相机同步录制（1 个 Basler 高速相机 + 3 个 Logitech webcam）。
官方为序列模型（RNN + 分割 + 潜变量）设计；本平台只取**静态 appearance baseline（ResNet）**，
训练策略与 xgaze 一致，不涉及官方复杂模型。

官方划分（`src/datasources/common.py`）：
- train01–39 / val01–05 / test01–10（etc01–02 存在于代码但未发布）

## 2. 磁盘实况盘点

- **50 / 54 被试**：train×39 + val×5 + **test×6（test07–10 未下载）**
- 全数据集 4259 个 `step*` 目录；**每人 step 数不同**（train01=51，train02=98，val01=86，test01=93），
  step 编号有空洞（train01 缺 027/028/038/039/041–044/065/066）——遍历以目录列表为准，勿按编号枚举
- 刺激类型三种：`stepNNN_image_*` / `stepNNN_video_*` / `stepNNN_wikipedia_*`；
  **无 9 点标定（points）目录**——官方 loader 亦排除
- basler 帧数：train **2,884,980** / val **339,720** / test 462,480（合计 3,687,180）；
  webcam_c 恰为其一半（30fps vs 60fps，时长一致）
- 根目录的 `eye_validity_sample_paths_{train,val,test}` 是 **pickle**（帧路径列表，供瞳孔有效性评估），与本任务无关

## 3. 每个 step 目录的文件与 h5 字段

每个刺激目录含 4 相机 ×（`{cam}.mp4` 全帧 1920×1080 **已去畸变**、`{cam}_face.mp4`/`_eyes.mp4`
官方归一化补丁、`{cam}.timestamps.txt`、`{cam}.h5`）+ screen 系列（本任务不用）。

`{cam}.h5` 实测字段（basler 为例，N=帧数）：

| 字段 | 形状 | 说明 |
|---|---|---|
| `camera_matrix` | (3,3) | basler 实测 fx≈1780.60 fy≈1779.85 cx≈959.33 cy≈579.31（每个 h5 都带，按文件读取，勿硬编码） |
| `camera_transformation` | (4,4) | **屏幕坐标系 → 该相机坐标系**（mm），含 `inv_` 逆 |
| `millimeters_per_pixel` | (2,) | ≈0.288（1920×1080 → 553×311 mm，对角 634mm = 25"，自洽） |
| `facial_landmarks/{data,validity}` | (N,68,2) / (N,) | FAN 68 点，全帧坐标（我们不用，走 insightface） |
| `head_rvec` / `head_tvec` | (N,3,1) | 官方 PnP 头姿（我们不用） |
| `{face,left,right}_PoG_tobii` | (N,2) | **注视点屏幕像素坐标**（GT 源头） |
| `{face,left,right}_g_tobii` | (N,2) | 官方归一化后视线 (θ,φ)，与平台约定一致（§4） |
| `{face,left,right}_R/W/h/o` | — | 官方归一化中间量（R=旋转校正，W=透视变换，o=注视 origin） |
| `left_p` / `right_p` | (N,) | 瞳孔直径 mm |

关键事实：
1. **帧对齐**：mp4 帧数 == h5 N == timestamps 行数（实测 basler 180/180、webcam 90/90）→
   按帧索引直接对齐，无需时间戳插值。时间戳单位 µs（仅跨相机对齐需要，本任务不用）。
2. **test 无 GT**：test01–06 的 h5 缺 `*_PoG_tobii`、`*_g_tobii`、`left_p/right_p`
   （保留 landmarks/head/R/W/h/o）——官方为 Codalab 评测保留隐藏。
3. DATASET.md 把 head_rvec 写作 `(N, 180, 3, 1)` 是**文档笔误**（180 恰为示例文件帧数），实际 (N,3,1)。
4. **官方代码约定陷阱**：训练代码 `core/gaze.py` 的 `vector_to_pitchyaw`
   （θ=−arcsin(y), φ=arctan2(x,z)）与数据集标签的实际约定是**镜像的**；
   标签实际约定 = θ=arcsin(−y), φ=arctan2(−x,−z)，即**本平台 `utils/normalization.vector_to_angles` 原样适用**（§4 对拍证实）。
5. 有效性：`face_PoG_tobii/validity` 为标签门控位。抽样有效率：image ≈100%（173–180/180），
   video 84–92%（1518/1800、3306/3600），wikipedia ≈98%（5227/5340）。

## 4. 标签几何链路（已验证，预处理的核心公式）

GT 源头是 `face_PoG_tobii`（屏幕像素）。屏幕坐标系约定经对拍实验锁定：

```
p_screen_mm = [ x_px · mmpp_x ,  y_px · mmpp_y ,  0 ,  1 ]ᵀ
               # 原点 = 屏幕左上角，+x 向右，+y 向下，z=0 为屏幕物理表面，单位 mm
p_cam_mm    = camera_transformation @ p_screen_mm      # → 该相机坐标系（mm）
```

随后走本平台统一管线（与 GazeCapture 同构）：

```
insightface 106 点 → estimateHeadPose(K=h5 的 camera_matrix, dist=0)   # mp4 已去畸变
→ normalizeData_face(img_bgr→rgb, face_model, rvec, tvec, p_cam_mm, K)
→ face_patch 224×224 + gc_normalized → vector_to_angles → (theta, phi)
```

**验证实验**（本报告的核心证据）：用官方中间量重构对拍——
`p_cam` − `face_o` 得原始注视向量 → 官方 `face_R` 旋转（即官方归一化）→ 平台角度公式，
与官方标签 `face_g_tobii` 比较：

| 文件（覆盖 4 相机、image/video/wikipedia、train/val、不同被试） | max\|Δ\| (rad) |
|---|---|
| train01 / step007_image / **basler** | 1.19e-07 |
| train01 / step007_image / **webcam_l** | 8.94e-08 |
| train01 / step007_image / **webcam_c** | 8.94e-08 |
| train01 / step007_image / **webcam_r** | 1.79e-07 |
| train01 / step029_video / **webcam_l** | 1.79e-07 |
| train01 / step029_video / **webcam_r** | 2.09e-07 |
| train01 / step029_video / basler | 1.79e-07 |
| train01 / step040_wikipedia / basler | 1.79e-07 |
| val01 / step008_image / basler | 1.19e-07 |
| train17 / step030_video / basler | 1.79e-07 |

**四相机均通过**：每相机 h5 各自带独立的 `camera_matrix`（basler fx≈1781，webcam fx≈1414/1435/1483）
与 `camera_transformation`——逐 h5 读取即可，§4 公式对所有相机通用，无需分支。

即机器精度吻合。被排除的候选：屏幕中心原点（±y 朝向均错 ~0.2/2.7 rad）。
含义：(a) 上述屏幕系约定正确且唯一；(b) 实现完成后可用 `face_g_tobii` 对拍我们管线的标签做质检，
预期差异仅来自关键点/头姿来源不同（insightface+PnP vs 官方 FAN+PnP）。

### 4.1 `face_g_tobii` 的真实语义与使用陷阱（2026-08-30 补充，已对拍）

`face_g_tobii` **不是相机系原始视线方向**，而是官方 `face_R` 归一化后（头架）的
视线 (θ,φ)——四相机同一物理时刻的该值理论上相同（官方帧号对齐为近似）。若直接把
`gv(θ,φ)` 当相机系方向使用，会产生 ~40° 量级的系统性错误（实测跨相机"一致性"p50
41.7°，即此陷阱的特征信号）。

**反旋转回相机系**（与 PoG 重构对拍，3 相机 dot = ±1.000000）：

```
d_cam = − face_Rᵀ @ gv(θ, φ)      # face_R 逐相机逐帧读自同一 h5；负号是官方约定的 z 翻转
gaze_point = 头位置 + d_cam·600     # 再走 normalizeData_face（v1 管线 PoG 路径等价）
```

**跨相机 HCS 一致性实测**（严格三角化头姿 + PoG 直算链，固定参考 cam00，
44 被试 × 120 组，`get_face_model/eve/metrics/frame_consistency/pos_hcs_consistency.py`）：

| 臂 | HCS 一致性 (vs cam00) |
|---|---|
| gen6 逐相机 PnP（v1 形态） | 10.08° |
| true6 四台一组 DLT（v2 形态） | **0.04°（机器精度）** |

**结论修正（2026-08-30 晚）**：各相机 h5 的 PoG/g_tobii 是同一路 tobii 流按
同步帧号分发的——PoG 直算链下跨相机一致性 0.04°，**不存在此前认为的 ~2°
webcam 时钟同步残差**（早期用 face_g_tobii+face_R 反旋转链测得的 1~2° 残差
来自官方 face_R 逐相机归一化噪声 + 600mm 合成注视点近似，非数据本身）。
basler //2 帧号映射由此得到机器精度级验证。

## 5. 划分方案（决策点 D3）

- 官方 test（10 人）不可用：无 GT（且本地仅 6/10）→ **弃用 test01–06**（46.2 万帧，无标注价值）
- 本平台不用验证集（GazeCapture 先例：官方 val 并入 train）
- **建议**：`train01–39` = 平台 train，`val01–05` = 平台 test
  - 直接尊重官方被试级边界（val 被试从未参与任何训练），比重新混划更干净
  - 测试规模：339,720 帧 → 10Hz 后 ~5.7 万帧，充足
  - 与 xgaze 的自划分（75+5）、MPII 的 LOO 并列为本平台第三种划分模式，写入 `configs/datasets/eve.yaml`

## 6. 预处理流水线方案（与 gazecapture 同构）

| 环节 | 方案 | 依据 |
|---|---|---|
| 相机 | **basler + webcam_l/c/r 全部 4 个** | 视角更丰富（同一时刻 4 个观测）；webcam 时间戳同步为系统级"尽力而为"（官方明言仅 basler 与眼动仪可靠同步），该噪声已接受并由标签有效性部分兜底 |
| 采样 | **5Hz（已定）**：basler 每 12 帧、webcam 每 6 帧（`round(fps/5)`，四相机起始帧对齐，同一时刻四视角） | 2Hz 曾定量对比后弃用（分布保真等价、冗余更低，但样本量减 60%）；对比数据保留于 §7 |
| 解码 | cv2.VideoCapture 顺序读，BGR→RGB 存储 | 实测 0.7 ms/帧（1080p H.264） |
| 检测 | insightface buffalo_l（det 640 + 2d106），1080p 原生 | 实测稳态 11.6 ms/帧（含 warmup 首帧 ~170ms 是 cudnn 算法搜索，勿被误导） |
| 门控 | `face_PoG_tobii/validity` 且检测成功 | 无效 PoG / 检测失败均记 FailureRecorder（`no_annotation` / `no_face_detected`） |
| 归一化 | `utils/normalization.py` 纯函数，face_model_xgaze.txt，dist=0，K 逐 h5 读取 | §4 |
| 产出 | 每被试一个 h5：`face_patch(N,224,224,3) uint8(lzf) + face_gaze(N,2) + face_mat_norm + facial_landmarks_2d + frame_index/cam_index/step_index`；**attrs 自描述回溯**——`steps`/`cameras` 有序列表，原始帧 = `<source>/<steps[step_index]>/<cameras[cam_index]>.mp4` 第 `frame_index` 帧（单独 frame_index 无法定位，故三索引齐存）；**BGR 存储**（cv2 原生顺序，与其他数据集一致，loader 统一翻转） | 平台统一格式（见 CLAUDE.md 约定 5）；实现 `preprocess/zhang2015-insightface/eve/`，配置 `configs/preprocess/eve.yaml` |
| 代码/配置 | `preprocess/zhang2015-insightface/eve/`（包形式，preprocessor.py + 薄 re-export）；`configs/preprocess/eve.yaml`；入口 `preprocess.py --dataset eve --method zhang2015-insightface` | gazecapture 同款结构 |

## 7. 采样率定量分析与规模估算（基于 §6 实测，4 相机）

### 7.1 帧间去相关时间尺度（train01 三类刺激，basler 全帧标签实测）

同一刺激内相隔 Δt 的两帧注视方向平均夹角，除以随机配对基线（100% = 完全独立）：

| 刺激 | 独立基线 | 0.1s | **0.2s（5Hz）** | **0.5s（2Hz）** | 1s | 2s |
|---|---|---|---|---|---|---|
| image | 5.39° | 12% | **22%** | **57%** | 118% | — |
| video | 12.70° | 13% | **25%** | **51%** | 69% | 88% |
| wikipedia | 12.09° | 17% | **32%** | **62%** | 85% | 84% |

- 5Hz 的相邻样本对仅 22–32% 去相关（多数还在同一注视内）；2Hz 过半；1s 间隔基本完全独立
- 头姿去相关更慢（video 1s 才 54%），但 4 相机的外观多样性不受采样率影响，补足外观维度

### 7.2 5Hz vs 2Hz 采样模拟（4 被试 train01/02/17 + val01 全部 355 步 basler 标签；按预处理同款规则 `i % 12` / `i % 30` 模拟）

**分布保真度**（相对该类型全部有效帧；W = Wasserstein 距离，PoG 用像素、gaze 用度；
覆盖比 = 屏幕 64×36 格覆盖数 ÷ 同规模随机抽样基准，>100% 表示系统采样比随机更均匀）：

| 类型 | 全集 N | 5Hz N | W_x px | W_y px | W_gaze ° | 覆盖比 | 2Hz N | W_x px | W_y px | W_gaze ° | 覆盖比 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| image | 31,570 | 2,615 | 6.9 | 5.6 | 0.17 | 109% | 1,029 | 18.2 | 13.0 | 0.47 | 109% |
| video | 141,633 | 11,814 | 1.9 | 0.7 | 0.04 | 106% | 4,723 | 3.2 | 1.3 | 0.10 | 106% |
| wikipedia | 51,912 | 4,331 | 1.8 | 1.1 | 0.05 | 106% | 1,731 | 5.9 | 1.5 | 0.18 | 109% |

1920×1080 屏上 18px ≈ 屏宽 1%——**两档的边缘分布都无实质损失**。

**相邻采样帧的视线角变化**（冗余度直接度量；<0.5° 为 Tobii 噪声级近重复）：

| 类型 | 档 | 对数 | 中位 | 均值 | p90 | <0.5° | 0.5–2° | 2–5° | >5° |
|---|---|---|---|---|---|---|---|---|---|
| image | 5Hz | 2,403 | 0.34° | 1.92° | 6.0° | **54.3%** | 16.4% | 16.1% | 13.2% |
| image | 2Hz | 827 | 1.55° | 3.29° | 9.0° | **37.4%** | 15.1% | 20.8% | 26.7% |
| video | 5Hz | 11,721 | 0.92° | 2.92° | 8.6° | **40.3%** | 22.6% | 17.0% | 20.1% |
| video | 2Hz | 4,630 | 3.64° | 5.64° | 13.9° | **15.1%** | 20.2% | 23.0% | 41.7% |
| wikipedia | 5Hz | 4,321 | 2.63° | 4.19° | 9.7° | **17.8%** | 23.0% | 35.7% | 23.5% |
| wikipedia | 2Hz | 1,721 | 6.34° | 8.33° | 17.8° | **4.2%** | 10.6% | 24.6% | 60.5% |

**对比结论**：
1. **分布层面两档等价**——PoG/注视方向的边缘分布、屏幕覆盖均无损失（2Hz 最差也只有 18px/0.47°）
2. **差异全部在冗余度**——5Hz 相邻对 18–54% 是 <0.5° 近重复；2Hz 降到 4–37%（video/wiki 降幅最显著）
3. **唯一代价**：image 刺激（3s）每步样本 15 → 6（个别全无效步为 0）；聚合分布不受影响（W 0.47°）

### 7.3 规模与耗时（两档；解码 806 万流帧 ×0.7ms 由读帧线程重叠隐藏，检测/归一化 ~12.6ms/候选帧为瓶颈）

| 档 | 候选帧 | train 样本 | test 样本 | 存储 | 耗时 |
|---|---|---|---|---|---|
| **5Hz（已定，每 12/6 帧）** | 1,074,900 | ~90 万 | ~10 万 | ~148 GB | **~5 h**（冒烟实测 60 帧/s 外推） |
| ~~2Hz~~（弃用） | 429,960 | ~36 万 | ~4.3 万 | ~60 GB | ~1.6 h |

**冒烟验证**（train01 × 2 步 × 4 相机 = 120 候选）：120/120 入库、0 失败；
我们的标签 vs 官方 `face_g_tobii` 中位差 **1.53°/1.61°**（insightface vs 官方 FAN 关键点/头姿差异的预期量级，
证明几何链路端到端正确）；face_patch 肉眼检查通过（居中正脸、无通道错乱、无裁切）。

（参考量级：xgaze train 61.6 万；MPII train 3.5 万 → LOO 4.98°。webcam 检测成功率可能略低，实际值运行后由失败记录给出。）

磁盘：`/home`（nvme）现有 659G 空闲，两档均无压力（GazeCapture 预处理输出已定不落系统盘）。

## 8. 阻塞与待审核决策

**磁盘（已解除）**：`/media/hitsz/ylx` 已满（98%），不可作输出。按新约定输出放
**`/home/hitsz/dataset/eve_insightface_224/`**（nvme 系统盘，659G 空闲，~75G 无压力）；
ylx 上的 EVE 原始数据保持只读。另注：无 GT 的 test01–06 占 ~26 G，弃用后可释放（是否删除由用户定）。

| # | 决策 | 状态 |
|---|---|---|
| D1 | 相机 = basler + webcam_l/c/r **全部 4 个** | ✅ 已定（视角丰富；webcam 同步噪声已接受，见 §6） |
| D2 | 采样率 = **5Hz**（basler 每 12 帧、webcam 每 6 帧取 1） | ✅ 已定（2Hz/10Hz 均经定量对比后弃用，对比数据保留 §7） |
| D3 | 划分：train01–39 训练 / val01–05 → 平台 test | ✅ 已定（§5） |
| D4 | 输出 `/home/hitsz/dataset/eve_insightface_224/{train,test}/{trainNN\|valNN}.h5`（保留官方被试名溯源） | ✅ 已定 |
| D5 | insightface 用原始 1080p 帧检测，不做任何缩放 | ✅ 已定 |
