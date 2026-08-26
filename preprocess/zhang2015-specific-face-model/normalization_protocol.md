# zhang2015-specific-face-model 归一化管线协议（完整预处理流程）

目标：用逐人/session 刚体人脸模型（`get_face_model/` 已全量生成，覆盖见各 README）
替换通用 `face_model_xgaze.txt` 重跑四数据集归一化，产出第二套统一 h5 训练数据，
与现行 `*_insightface_224`（通用模型版）构成「预处理管线 ×」对照轴。

前置成果：四套模型 `ylx/<ds>_specific_face_model/face_models/<id>/{组名}_model6.txt`
（xgaze/eve=cam{cc}、gazecapture=ori{o}、mpii=cam00）+ 指标 `get_face_model/*/metrics/`。

## 1. 接入方式（零入口改动）

```bash
python preprocess.py --dataset xgaze --method zhang2015-specific-face-model
```

- 配置：`configs/preprocess/zhang2015-specific-face-model/<ds>.yaml`（与 insightface 版同构，
  新增 `face_model_root`、`pnp_points`、`compress` 字段）
- 模块：`preprocess/zhang2015-specific-face-model/normalize_<ds>.py` ×4（文件形式，
  暴露 `run(config, recorder)`；复用 utils/normalization 纯函数与 zhang2015-insightface
  各预处理器的遍历骨架/线程池——从 landmarks h5 索引遍历，**不做 insightface 检测**）
- 训练侧：`configs/datasets/zhang2015-specific-face-model/<ds>.yaml`（新管线数据目录），
  scripts 同构第三条管线目录——对照实验天然成立

## 2. 数据流（每帧）

```
landmarks h5 行（帧索引 + facial_landmarks_2d）
  → 相机组定位（xgaze: cam_index / gc: orientation / eve: cam_index / mpii: 唯一）
  → 模型查表：<face_model_root>/<id>/{组名}_model6.txt
      组模型缺失（GC 39 session、未过准则的相机/朝向）→ 回退通用模型，
      记 recorder 类别 fallback_generic（含 id+组名，可统计但不计失败）
  → estimateHeadPose：PnP（点集 = pnp_points 配置，6 点 IDX6 或 28 点 RIGID）
  → normalizeData_face（旋转中心用 model6 的 6 行，同现行几何约定）
  → 统一 h5：face_patch uint8 BGR + face_gaze (θ,φ) + face_mat_norm + 索引字段
      （字段与 insightface 版逐 dataset 完全一致，loader 零改动）
```

视线标签链路不变（xgaze CSV / mpii pXX.txt / gc dotInfo 映射 / eve PoG_tobii），
仅头姿估计与归一化几何随模型替换改变。

## 3. 逐数据集参数

| | 帧源（landmarks） | 模型查表 | 内参/畸变 | 特殊处理 |
|---|---|---|---|---|
| xgaze | Expansion/xgaze_raw/data/landmarks | cam{cc:02d}（未选中相机→通用） | 18 相机 xml | FLIP_CAMERAS=[3,6,13] 旋转 |
| mpiifacegaze | zyx/MPIIFaceGaze/landmarks | cam00（缺失→通用） | 逐人 Camera.mat | — |
| gazecapture | zyx/GazeCapture/landmarks | ori{o}（组缺失→通用） | 设备×分辨率 xml | 朝向→分辨率查表 |
| eve | zyx/EVE_dataset/…/landmarks | cam{c:02d}（未过准则相机→通用） | 逐相机 h5，零畸变 | mp4 顺序解码 + lzf |

## 4. 资源估算（全量）

| | 帧数 | 预计时长* | 预计空间（lzf） |
|---|---|---|---|
| xgaze | 756k | 2~3h | ~89G |
| mpiifacegaze | 37.7k | ~10min | ~4.5G |
| gazecapture | 1.49M | 4~7h | ~176G |
| eve | 997k | 4~6h（解码+lzf） | ~114G |
| 合计 | | **~11~16h（串行，CPU）** | **~385G** |

\* 无 insightface 检测（原 ~11.6ms/帧 GPU 省去），瓶颈为原图 IO + warp + lzf 压缩。
ylx 现剩 **431G**：全 lzf 勉强放下（余 ~46G），见决策 D2。

## 5. 验证方案

1. **小样本先行**：每数据集 1 被试/session 全流程跑通 + `viz.py` 可视化对齐 sanity
2. **几何统计**：新旧两套逐帧头姿角差分布（期望：多数 <2°，模型修正集中在原通用模型
   误差大的视角）、注视角差分布、fallback 比例统计
3. **训练对照**（最终验收）：xgaze within-dataset 两套数据各训一次（bs200 默认，2.4h/次），
   test error 下降即验证收益；再决定其余数据集

## 6. 决策记录（2026-08-26 用户确认）

| # | 决策 | 结论 |
|---|---|---|
| D1 | PnP 点集 | **A：6 点**（model6+IDX6 默认）。小样本 A/B 实测：6 vs 28 点的头姿差中位 0°、p90 0.15°——**差异可忽略**，6 点定稿 |
| D2 | 输出空间 | **新 2TB SATA 固态**（Samsung 870，/dev/sdo，挂载到 `/media/yanglinxuan/sfm`），输出 `/media/yanglinxuan/sfm/<ds>_specific_224`；ylx 盘不动 |
| D3 | 启动时机 | **小样本验证已完成**（见 §8），是否全量预处理待用户决定 |
| D4 | 验证顺序 | **A：xgaze 先行**（全量预处理 + 对照训练） |
| 范围 | GazeCapture | **预处理范围 = FAZE 筛选 ∩ 建模成功（1177 session，名单已固化为 `configs/splits/gazecapture_sfm.yaml`：train 1055 / test 122）**：train <500 / test <1000 的不参与（已裁剪模型）；**建模失败的 session（协议池内 14 train + 1 test）也不预处理、不参与训练测试**（2026-08-27 定稿）；有模型 session 内缺失朝向组回退通用 |

## 8. 小样本验证结果（2026-08-26，subject0000 前 50 帧 × 18 相机）

- **管线正确性** ✅：fallback 相机（10/18）输出与现行通用版产物逐字节一致（像素差 0.02/255）；
  specific 相机（8/18）正常注入模型；failures.json 记录 fallback_generic×10
- **几何变化量**：specific 相机上，头姿差 vs 通用版 中位 ~0° / p90 0.16° / max 0.88°，
  视线角差同量级；归一化 patch 平均像素差 2.0/255（max 6.2）
- **⚠️ 关键解读**：通用模型 25px 的重投影残差**大部分被 PnP 拟合吸收，未传导为姿态/patch 差异**——
  个性化模型对 xgaze 归一化的实际改变远小于建模指标（4.2×）所暗示的幅度。
  全量对照训练的预期收益相应下调（可能 <0.1°）；是否继续全量，由用户基于此判断

## 7. 实施步骤（按决策结果填充）

1. [ ] D1~D4 确认，本文档定稿
2. [ ] 四份 normalize_<ds>.py + configs（复用骨架，预计每份 200~400 行）
3. [ ] 小样本验证（每数据集 1 被试）+ 几何统计报告
4. [ ] 按 D4 顺序全量
5. [ ] datasets/scripts 第三管线目录 + 对照训练
