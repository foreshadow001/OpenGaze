# MPIIFaceGaze 逐人个性化人脸模型（已实现）

参照 `../xgaze/`（方案文档 `personalized_face_modeling.md`）实现：`personalized_face_model.py`，
几何与优化走共享核心 `../face_model_core.py`（自 xgaze 实现提炼，fx 参数化）。

与 xgaze 的差异：单相机（无逐相机拆分问题）、实验室固定机位、无官方逐相机标定 xml——
内参逐人取原始数据 `pXX/Calibration/Camera.mat`（与 zhang2015-insightface 预处理同源）。

- 观测：`/media/yanglinxuan/zyx/MPIIFaceGaze/landmarks/pXX.h5`（facial_landmarks_2d）
- 输出：`/media/yanglinxuan/ylx/mpiifacegaze_specific_face_model/face_models/pXX/`，
  单相机组命名 `cam00_model6/28.txt`（沿用 xgaze 的 `{组}_` 约定，下游查找代码跨数据集通用）
- 指标留档：`metrics/`（p00 实测：train 0.46 px / test 0.83 px / 通用基线 2.28 px，改善 2.8x，IOD 90.1 mm）

```bash
# 全量 15 人（仓库根目录运行，CPU 即可）
/ssd/conda/envs/yanglinxuan/opengaze/bin/python preprocess/zhang2015-specific-face-model/get_face_model/mpiifacegaze/personalized_face_model.py
```
