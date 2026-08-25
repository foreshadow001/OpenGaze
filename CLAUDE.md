# CLAUDE.md

通用视线估计训练/测试平台（多数据集：ETH-XGaze、MPIIFaceGaze 已接入，GazeCapture、EVE 计划中）。基于 ETH-XGaze 官方代码演化而来。

## 环境

- conda 环境名 `opengaze`：`/home/hitsz/anaconda3/envs/opengaze/bin/python`
- Python 3.12 + torch 2.13.0 (cu126)，RTX 4060 Ti 16G
- 运行示例：`/home/hitsz/anaconda3/envs/opengaze/bin/python main.py --dataset xgaze_smoke --method resnet50 --set method.train.epochs=2`

## 运行方式

```bash
python main.py --dataset xgaze --method resnet50            # 训练（自动建 exp/expNN）
python main.py --dataset xgaze --method resnet50 --exp exp01  # 指定编号（已存在则拒绝）
python main.py --resume exp01                                # 断点续训（仅此一参，配置取快照）
python main.py --dataset xgaze --method resnet50 --test --exp exp00  # 测试（最新 ckpt）
python main.py --dataset mpiifacegaze --method resnet50 --test --exp exp00  # 跨数据集评测
python main.py ... --set method.train.epochs=2              # 点路径覆盖配置
python main.py --resume exp01 --run fold_03                 # 子运行级续训（LOO 各折）
bash scripts/resnet50/within-dataset/xgaze.sh                       # within：训练+测试一条龙（新开实验）
REUSE_EXP=exp00 bash scripts/resnet50/cross-dataset/xgaze_mpiifacegaze.sh  # cross：评测（必须指定实验）
XGAZE_EXP=exp00 bash scripts/resnet50/cross-dataset/all.sh          # 全 cross-dataset 矩阵（按源指定）
```

## 关键约定（改代码前必读）

1. **日志规范**：所有输出必须经 `utils.logger.get_logger(__name__)`，禁止 `print` / `tqdm.write`。格式含时间+级别+文件:行号，console 与 `exp/expNN/logs/run.log` 双写。tqdm 进度条本身除外。

2. **Checkpoint 格式标准（opengaze-ckpt v1）**：`save_checkpoint` 写出的字典字段为
   `format_version`(1) / `epoch` / `train_iter` / `model_state` / `optim_state` / `scheduler_state`，
   文件名 `epoch_{N}_ckpt.pth`（N 从 0）。详见 README「Checkpoint 格式标准」。
   仅支持本标准，不兼容其他格式（含 ETH-XGaze 官方旧 ckpt），缺字段直接报错。
   **新增字段必须递增 format_version**。保存频率由 method yaml 的
   `train.ckpt_save_interval` 控制（默认 5，最后一个 epoch 始终保存，不影响 resume）。

3. **配置两轴**：`configs/datasets/*.yaml`（数据位置、split 划分、dataloader 参数）× `configs/methods/*.yaml`（模型+训练策略），运行时合并。快照存 `exp/expNN/config.yaml`，测试/续训以快照为准。

4. **实验目录自包含**：`exp/expNN/` = config.yaml + method.yaml 副本 + ckpt/ + logs/（tensorboard + run.log）+ test_result_*.json。训练自动递增创建，`--exp` 指定已存在目录会拒绝。带子运行的实验（MPII LOO）用 `--run NAME`（fold_00~fold_14、all）把多次运行放进同一实验目录；**快照两级**——每个 run 目录有自己的 config.yaml（含该折 fold 等实际配置，resume/test 以它为准），顶层实验快照在首建时写（require_exp 用）；子运行目录已存在会拒绝。within 的 mpiifacegaze.sh 可安全重跑（已完成折跳过、中断折自动 resume）。`exp/` 整体在 .gitignore 中。

5. **统一数据格式**：所有数据集预处理为 h5：`face_patch (N,224,224,3) uint8`、`face_gaze (N,2) (pitch,yaw) 弧度`，目录 `<data_dir>/<sub_folder>/<file>`。**所有 h5（含官方 xgaze 与本平台预处理产物）均为 BGR 存储**（cv2 原生顺序，预处理不做通道转换），各 loader 统一传 `bgr_to_rgb=True` 翻转。

6. **数据位置**：原始数据在 `/media/hitsz/ylx/`（xgaze_224、mpiifacegaze_insightface_224、MPIIFaceGaze、GazeCapture、EVE_dataset、Gaze360；该盘已近满，**只读不写**）。**预处理输出（2026-08 起）统一放 `/home/hitsz/dataset/<name>_insightface_224/`**（nvme 系统盘，空间充足；已有 xgaze_insightface_224）。
   - ETH-XGaze：训练用 `xgaze_224/train/`（官方预处理）下 80 被试自划分 75 train + 5 test（subject0106~0111），划分列表在 `configs/datasets/xgaze.yaml`；`test/` 目录是官方无标注集，不要使用。**不重新预处理 xgaze**（`xgaze_insightface_224/` 是 insightface 管线的历史产物，仅作参考，勿覆盖勿重跑）
   - GazeCapture：原始数据 `/media/hitsz/ylx/GazeCapture`（1474 个五位数 session 目录：`frames/` + 帧级 json，官方代码 /home/hitsz/GazeCapture）；**官方划分为 session 级**（iTracker reference_metadata.mat 提取，已与 info.json 的 Dataset 字段互证），保存在 `configs/splits/gazecapture_sessions.yaml`：train 1321（官方 val 50 已并入，本平台不用验证集）/ test 150 session，excluded 3 个（01185/01730/02065，预处理跳过）；**不使用官方 appleFace/eye bbox 与 IsValid**——人脸定位完全走 insightface 管线，帧有效性由管线成败决定；视线标签在 dotInfo.json 的 XCam/YCam（相机系平面坐标）
   - MPIIFaceGaze：`mpiifacegaze_insightface_224/` 15 个 pXX.h5 直接在根目录（全量 37,667 样本），BGR 存储；单一配置 `configs/datasets/mpiifacegaze.yaml`，`split.mode` 切协议——`leave_one_out`（折 i 测 p{i:02d}，`--set dataset.split.fold=i`）与 `all_subjects`（15 人全量）；within 脚本把 15 折 + 全量放进**同一实验目录**的子运行（fold_00~fold_14、all）并汇总；cross 的 mpiifacegaze 源取 `--run all` 的 ckpt、目标测试加 `--set dataset.split.mode=all_subjects`（cross 脚本已内置）；训练策略与 XGaze 同一份 method yaml

## 结构

- `main.py` 入口（train / test / resume 三模式）
- `preprocess.py` 预处理入口（`--dataset xgaze|mpiifacegaze --method zhang2015-insightface`；配置在 `configs/preprocess/`；管线脚本在 `preprocess/<method>/normalize_<dataset>.py` 按路径 importlib 加载——目录名含连字符非合法包名）
- `preprocess/` 预处理：`common.py`（FailureRecorder 失败样本记录 + 运行目录）；管线目录 `zhang2015-insightface/`（Zhang2015 归一化 + insightface 关键点）：`normalize_xgaze.py`（XGazePreprocessor 类）、`normalize_mpiifacegaze.py`、`face_model_xgaze.txt`、`gazecapture/`（**已实现已验证**：GazeCapturePreprocessor + 方向映射 docstring；`generate_calibration.py` 生成内参 xml → `/media/hitsz/ylx/GazeCapture/calibration/`（15 设备×双分辨率，cam00.xml 同构）；`calibrate_fx.py` 为 fx 自标定负结果实验；`front_cameras.yaml` 设备分组与 fx；`dot_transfer.md` 四朝向 dot→CCS 定稿公式；`obtain_camera_intrinsics.md` 内参获取全记录）；预处理器两种形式：`normalize_<ds>.py` 文件或 `<ds>/__init__.py` 目录包，均暴露 `run(config, recorder)`；每次运行留档 `preprocess/log/<dataset>_<时间戳>/{run.log, failures.json}`；失败/跳过帧逐条记录（imread_failed / no_face_detected / no_annotation / invalid_dot / error:*）
- `utils/normalization.py` 归一化管线纯函数（estimateHeadPose / normalizeData_face / vector_to_angles），无常量、无数据集逻辑
- `scripts/` 运行脚本：`resnet18|resnet50 × within-dataset|cross-dataset`。within-dataset = 训练+测试一条龙（每次新开实验，无需参数，断点续训直接用 main.py --resume）；cross-dataset = 只做评测，`scripts/common.sh` 的 `require_exp` 强制 `REUSE_EXP=expNN` 显式指定实验，不指定/不存在/快照不符直接报错；`all.sh` 按源数据集用 `<大写名>_EXP` 指定，未指定的源跳过；测试结果存 `exp/expNN/test_result_<dataset>.json`
- `trainers/trainer.py` 训练与评测循环
- `datasets/` 各数据集 loader + `base.py` 统一 h5 基类 + `__init__.py` 工厂（新增数据集注册到 DATASETS）
- `models/` GazeNet（backbone + FC），BACKBONES 注册表
- `utils/` config（yaml 加载合并、--set 覆盖）、logger（实验目录 + 分级日志）、metrics（angular_error）
- 设计文档：STRUCTURE.md（含各模块设计细节与数据集接入计划）

## 预处理注意

- insightface **0.7.3** + onnxruntime-gpu **1.22.0**（CUDA 12 系；1.23+ 需 CUDA 13 不能用）
- onnxruntime 找 pip 版 cudnn 需 `LD_LIBRARY_PATH` 含 `site-packages/nvidia/{cudnn,cublas,cuda_runtime}/lib`——conda activation 脚本已配置（`envs/opengaze/etc/conda/activate.d/cuda_libs.sh`），**必须 `conda activate opengaze` 后跑预处理**（直接用绝对路径 python 时需手动前缀 LD_LIBRARY_PATH，否则静默 fallback 到 CPU，速度差 ~10 倍）
- 预处理会**覆盖** output_dir 下的同名 h5；调试用 `--set output_dir=... subjects=... max_days/max_frames=...`
