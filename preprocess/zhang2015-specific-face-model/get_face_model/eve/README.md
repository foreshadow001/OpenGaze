# EVE 逐被试真实人脸模型（严格三角化，v2）

与 `../xgaze/true_face_model.py` 同法（2026-08-29 定稿），差异在相机与帧同步：

- **帧同步**：basler 与 webcam 帧率 2:1 → 同步帧号 = `basler_raw // 2`
  （landmarks h5 按行存 (frame, cam, step)），仅保留 ≥3 相机齐全的同步组
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
