# MPIIFaceGaze 逐人个性化人脸模型（待实现）

参照 `../xgaze/`（方案文档 `personalized_face_modeling.md`、实现 `personalized_face_model.py`）。

与 xgaze 的差异：单相机（无逐相机拆分问题）、实验室固定机位、无官方逐相机标定 xml——
内参沿用预处理管线的标定来源（见 `../../zhang2015-insightface/normalize_mpiifacegaze.py`），
需要针对单人序列重新设计建模流程。
