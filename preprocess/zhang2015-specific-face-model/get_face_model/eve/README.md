# EVE 逐人个性化人脸模型（待实现）

参照 `../xgaze/`（方案文档 `personalized_face_modeling.md`、实现 `personalized_face_model.py`）。

与 xgaze 的差异：每刺激步 4 相机（basler 高质量 + 3 webcam，内参逐相机 h5 自带、mp4 已去畸变），
标签端 `face_PoG_tobii` + `camera_transformation` 链路已验证（见
`../../zhang2015-insightface/eve/dataset_report.md` §4）；
建模观测可取自预处理 h5 的 `facial_landmarks_2d` + `cam_index`（预处理已按 5Hz 采样存档）。
