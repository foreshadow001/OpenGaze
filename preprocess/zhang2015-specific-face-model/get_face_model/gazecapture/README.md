# GazeCapture 逐人个性化人脸模型（待实现）

参照 `../xgaze/`（方案文档 `personalized_face_modeling.md`、实现 `personalized_face_model.py`）。

与 xgaze 的差异：session 级身份（无跨 session 人 ID，逐 session 独立建模）、单前置相机、
内参为按设备分组的外部考证值（见 `../../zhang2015-insightface/gazecapture/front_cameras.yaml`
与 `obtain_camera_intrinsics.md`）而非官方逐相机标定。
