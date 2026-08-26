# GazeCapture 逐 session 个性化人脸模型（已实现）

参照 `../xgaze/`（方案文档 `personalized_face_modeling.md`）实现：`personalized_face_model.py`，
几何与优化走共享核心 `../face_model_core.py`。

与 xgaze 的差异：session 级身份（无跨 session 人 ID，逐 session 独立建模）、单前置相机、
内参为按设备分组的外部考证值（`../../zhang2015-insightface/gazecapture/` 的
`generate_calibration.py` 生成 `calibration/<slug>_<w>x<h>.xml`、`front_cameras.yaml`、
`obtain_camera_intrinsics.md`）而非官方逐相机标定。

「相机组」= 帧朝向（Orientation 1~4，同一 session 内混布；竖屏 1/2 用 480x640、
横屏 3/4 用 640x480 内参，与预处理逐帧查表同规则），各组独立 BA。

**两段式选择准则 + 质量门槛（2026-08-26 全量定稿）**：
1. 严格段（默认）：组中位姿态角 <40° 且训练视图 ≥15（xgaze 准则），质量门槛全程生效 → 1204 session
2. 补跑段：`--max-angle 65` 对严格段淘汰的极端姿态 session 放宽组选择，配**组级质量门槛**
   （留出 test RMS ≤2px 且 IOD 80~100mm，不过即弃组）→ 再补 216 session（test 中位 1.11px、
   2.0x 改善，可用）
3. 最终覆盖 **1420/1471 session（96.5%，帧级 79.4%）**，指标：train 中位 0.48px（p90 0.78）、
   test 中位 0.97px（p90 1.55）、改善 2.2x、IOD 87.1~98.3mm

**范围（2026-08-26 用户定稿）**：模型已按 FAZE 名单裁剪——train <500 / test <1000 的 session
不参与人脸建模与预处理，现存 **1177 个模型目录**（FAZE 1192 中 15 个本就无模型）。

**建模覆盖统计（全量 1471 session：严格段 1204 + 宽松段 216 + 失败/无模型 51）与
训练/测试协议（FAZE 筛选）的交集**（2026-08-27 统计；帧数为该批 session 的全部帧，
帧级朝向覆盖另见下，两者口径不同）：

| 协议池 | 严格段 | 宽松段 | 建模合计 | 失败/无模型 |
|---|---|---|---|---|
| train ≥500（1069 session / 1,942,187 帧） | 904 | 151 | **1055（98.7%，1,921,350 帧）** | 14 |
| test ≥1000（123 session / 259,776 帧） | 102 | 20 | **122（99.2%，256,895 帧）** | 1 |

即训练/测试真正用到的 session 里 98.7%/99.2% 有专属模型（session 级）；
帧级（扣除有模型 session 内未过准则/门槛的朝向组后实际能用专属模型的帧）为 **79.4%**，
其余回退通用模型。

**严格 vs 宽松段精度对比**（中位 [p10, p90]）：

| 指标 | 严格段（<40°，n=1204） | 宽松段（40~65°，n=216） |
|---|---|---|
| test RMS px | 0.95 [0.49, 1.53] | 1.11 [0.57, 1.58] |
| 通用基线 px | 2.16 [1.48, 3.05] | 2.08 [1.48, 3.09] |
| 改善 | 2.3× | 2.0× |

宽松段仅差 0.16px（+16%）且 p90 与严格段持平（门槛掐住了尾部），通用基线两组几乎
相同——40~65° 姿态下专属模型仍有稳定 2× 收益，放宽收编未牺牲模型质量。

**下游归一化回退约定**（zhang2015-specific-face-model 归一化管线遵循；2026-08-27 用户定稿）：
**预处理范围 = FAZE 筛选 ∩ 建模成功（即现存 1177 个模型目录的 session）——建模失败的
session（含协议池内 14 train + 1 test）一律不预处理、不参与训练与测试**；
有模型 session 内查表按帧朝向取 `ori{o}_model6.txt`，该朝向组缺失时回退通用
`face_model_xgaze.txt`（帧级 79.4% 用专属模型）。
（注：insightface 管线的训练协议不受此影响，仍为 FAZE 筛选全量 1069/123。）

- 观测：`/media/yanglinxuan/zyx/GazeCapture/landmarks/<split>/<session>.h5`（facial_landmarks_2d + orientation）
- 设备名：`<raw>/<session>/info.json` 的 DeviceName
- 输出：`/media/yanglinxuan/ylx/gazecapture_specific_face_model/face_models/<session>/`，
  `ori{o}_model6/28.txt`（o=1..4）
- 指标留档：`metrics/`（`summary_all.csv` 已含全部 1177 session；重跑某段后可用
  per-session txt 重建）

```bash
# 全量两段式（先默认跑，再补跑；已完成的 session 自动跳过）
.../personalized_face_model.py                                   # 严格段
.../personalized_face_model.py --max-angle 65                    # 补跑段
# 分片：-sb/-se 按排序后 session 列表切片；--split train|test 单跑一个 split
```
