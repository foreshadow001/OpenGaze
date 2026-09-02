# EVE 逐被试真实人脸模型（严格三角化，v2）

与 `../xgaze/true_face_model.py` 同法（2026-08-29 定稿），差异在相机与帧同步：

- **帧同步**：basler 与 webcam 帧率 2:1 → 同步帧号 = `basler_raw // 2`
  （landmarks h5 按行存 (frame, cam, step)）
- **组门控（2026-08-30 定稿）**：四相机齐全 + 各自 PoG validity 全真 +
  **PoG 跨相机离散 ≤5px**，缺一即弃——实证：四相机标注为同一 tobii 流按各自
  时间轴插值分发，webcam 时钟漂移使扫视瞬间各相机读到不同物理时刻
  （PoG 离散 ↔ HCS 误差 Pearson r=0.987，~19px↔1°）；全量 252,164 组中
  弃 2.4%（相机不齐）+ 19.5%（离散>5px），保留 78.2%。阈值决策留档
  `metrics/frame_consistency/pog_spread_gate_stats.{csv,png}`；
  离群案例归档 `metrics/frame_consistency/exception/`（四案：坏有效性 /
  PoG 离散 376px / 20px 边界 19px / 5px 底线 7px，通用分析工具
  `analyze_outlier.py`）。5px 门控后 true6 HCS 一致性 p95=0.12° / max=0.27°
  （时间错位机制底线）
- **DLT**：逐组全部相机——官方 `camera_transformation` 外参 + 逐相机
  `camera_matrix`（mp4 已去畸变，畸变取零），insightface 6 点 → 世界系 3D
- **成模**：逐帧 Kabsch 对齐 gen6（消头运动）→ 帧间中位 = true6 →
  `canonicalize`（标准系，CLAUDE.md 约定 9）

**输出**（`/media/yanglinxuan/sfm/eve_specific_face_model/face_models/<被试>/`）：
- `true6.txt`（gen6 对齐系）/ `true6_canonical.txt`（标准系），44 被试

**配套**（结构与 ../xgaze/ 一致：一个功能目录 = 脚本 + 产物）：
- [metrics/eye_nose_features/](metrics/eye_nose_features/)：模型几何特征——
  `gen6_vs_dlt.py`（gen6 vs DLT 真值 6 指标 → 本目录 `gen6_vs_dlt.png`，
  汇总进日志；实证 gen6 鼻宽 21.9 vs 真值 27.3 缺陷）
- [metrics/frame_consistency/](metrics/frame_consistency/)：跨相机一致性——
  `frame_consistency.py`（2+2 相机互验）及产物
- `multi_vs_single_camera.md`：早期多/单相机 BA 对比实验（历史记录）

```bash
/ssd/conda/envs/yanglinxuan/opengaze/bin/python preprocess/zhang2015-specific-face-model/get_face_model/eve/true_face_model.py
```
