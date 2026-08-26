# EVE 逐被试个性化人脸模型（已实现）

参照 `../xgaze/`（方案文档 `personalized_face_modeling.md`）实现：`personalized_face_model.py`，
几何与优化走共享核心 `../face_model_core.py`。

与 xgaze 的差异：每刺激步 4 相机（basler 高质量 + 3 webcam）——「相机组」= cam_index
（landmarks h5 attrs['cameras'] 顺序：basler/webcam_l/webcam_c/webcam_r），逐相机独立 BA，
组选择沿用 xgaze 准则（中位姿态角 <40° 且训练视图 ≥15）。

- 观测：`/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks/<split>/<被试>.h5`
  （facial_landmarks_2d + cam_index，5Hz 采样）
- 内参：`<raw>/<被试>/<任一 step>/<cameras[c]>.h5` 的 camera_matrix（实测同一被试同一相机
  跨 step 恒定）；mp4 官方已去畸变 → 畸变系数取零
- 输出：`/media/yanglinxuan/ylx/eve_specific_face_model/face_models/<被试>/`，
  `cam{c:02d}_model6/28.txt`（c 与预处理 h5 的 cam_index 对齐）
- 指标留档：`metrics/`（train01 实测：4 相机全过准则，train 中位 0.29 px / test 0.55 px /
  通用基线 4.15 px，改善 7.6x，IOD ~90 mm）

```bash
# 全量 train39+val05 共 44 被试（CPU，约 1 分钟）
/ssd/conda/envs/yanglinxuan/opengaze/bin/python preprocess/zhang2015-specific-face-model/get_face_model/eve/personalized_face_model.py
```
