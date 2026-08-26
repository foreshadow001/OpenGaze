# CLAUDE.md

通用视线估计训练/测试平台（四数据集全部接入训练：ETH-XGaze、MPIIFaceGaze、EVE、GazeCapture）。基于 ETH-XGaze 官方代码演化而来。

## 环境

- conda 环境名 `opengaze`：`/ssd/conda/envs/yanglinxuan/opengaze/bin/python`（本机 /ssd 盘）
- Python 3.12.13 + torch 2.13.0+cu132，**4× RTX 4090 24G**（训练走 DDP，见约定 7）
- 运行示例：`/ssd/conda/envs/yanglinxuan/opengaze/bin/python main.py --dataset zhang2015-insightface/xgaze_smoke --method resnet50 --set method.train.epochs=2`

## 运行方式

```bash
# --dataset 为「预处理管线/数据集」子路径（配置按管线分文件夹，见约定 3）
python main.py --dataset zhang2015-insightface/xgaze --method resnet50          # 训练（自动建 exp/expNN）
python main.py --dataset zhang2015-insightface/xgaze --method resnet50 --exp exp01  # 指定编号（已存在则拒绝）
python main.py --resume exp01                                # 断点续训（仅此一参，配置取快照）
python main.py --dataset zhang2015-insightface/xgaze --method resnet50 --test --exp exp00      # 测试（最新 ckpt）
python main.py --dataset zhang2015-insightface/mpiifacegaze --method resnet50 --test --exp exp00  # 跨数据集评测
python main.py ... --set method.train.epochs=2              # 点路径覆盖配置
python main.py --resume exp01 --run fold_03                 # 子运行级续训（LOO 各折）
python -m torch.distributed.run --nproc_per_node=4 main.py --dataset zhang2015-insightface/xgaze --method resnet50  # 4 卡 DDP 训练
# 用哪些卡：改 configs/common.yaml 的 gpus（默认 [0,1,2,3]）；临时覆盖用环境变量 GPUS（下同）
bash scripts/zhang2015-insightface/resnet50/within-dataset/xgaze.sh            # within：训练+测试一条龙（训练自动多卡）
GPUS=0 bash scripts/zhang2015-insightface/resnet50/within-dataset/xgaze.sh     # 单卡训练（只用 0 号卡）
REUSE_EXP=exp00 bash scripts/zhang2015-insightface/resnet50/cross-dataset/xgaze_mpiifacegaze.sh  # cross：评测（必须指定实验）
XGAZE_EXP=exp00 bash scripts/zhang2015-insightface/resnet50/cross-dataset/all.sh  # 全 cross-dataset 矩阵（按源指定）
```

## 关键约定（改代码前必读）

1. **日志规范**：所有输出必须经 `utils.logger.get_logger(__name__)`，禁止 `print` / `tqdm.write`。格式含时间+级别+文件:行号，console 与 `exp/expNN/logs/run.log` 双写。tqdm 进度条本身除外。

2. **Checkpoint 格式标准（opengaze-ckpt v1）**：`save_checkpoint` 写出的字典字段为
   `format_version`(1) / `epoch` / `train_iter` / `model_state` / `optim_state` / `scheduler_state`，
   文件名 `epoch_{N}_ckpt.pth`（N 从 0）。详见 README「Checkpoint 格式标准」。
   仅支持本标准，不兼容其他格式（含 ETH-XGaze 官方旧 ckpt），缺字段直接报错。
   **新增字段必须递增 format_version**。保存频率由 method yaml 的
   `train.ckpt_save_interval` 控制（默认 5，最后一个 epoch 始终保存，不影响 resume）。

3. **配置三轴 + 平台公共配置**：`configs/datasets/<预处理管线>/<数据集>.yaml`（数据位置、split 划分、dataloader 参数）× `configs/methods/*.yaml`（模型+训练策略）× `configs/preprocess/<预处理管线>/<数据集>.yaml`（预处理），数据集/预处理配置按**预处理管线**分文件夹（目前唯一管线 `zhang2015-insightface`；`zhang2015-specific-face-model` 管线暂无 configs）；另有 `configs/common.yaml` 平台公共配置（目前仅 `gpus` 用卡列表，见约定 7）。`--dataset` 参数为「管线/数据集」子路径（如 `zhang2015-insightface/xgaze`），配置内 `name:` 字段（loader 注册名）不含前缀。快照存 `exp/expNN/config.yaml`，测试/续训以快照为准。

4. **实验目录自包含**：`exp/expNN/` = config.yaml + method.yaml 副本 + ckpt/ + logs/（tensorboard + run.log）+ test_result_*.json。训练自动递增创建，`--exp` 指定已存在目录会拒绝。带子运行的实验（MPII LOO）用 `--run NAME`（fold_00~fold_14、all）把多次运行放进同一实验目录；**快照两级**——每个 run 目录有自己的 config.yaml（含该折 fold 等实际配置，resume/test 以它为准），顶层实验快照在首建时写（require_exp 用）；子运行目录已存在会拒绝。within 的 mpiifacegaze.sh 可安全重跑（已完成折跳过、中断折自动 resume）。`exp/` 整体在 .gitignore 中。

5. **统一数据格式**：所有数据集预处理为 h5：`face_patch (N,224,224,3) uint8`、`face_gaze (N,2) (pitch,yaw) 弧度`，目录 `<data_dir>/<sub_folder>/<file>`。**所有 h5（含官方 xgaze 与本平台预处理产物）均为 BGR 存储**（cv2 原生顺序，预处理不做通道转换），各 loader 统一传 `bgr_to_rgb=True` 翻转。

6. **数据位置**（2026-08-25 迁至本机，三块外置盘；**已配置 fstab 开机自动挂载**到 `/media/yanglinxuan/<卷标>/`——`nofail` 盘不在则跳过、`uid/gid=1000` 用户可写；若 fstab 尚未生效可手动 `sudo mount -a`）：原始数据在 `/media/yanglinxuan/zyx/`（1TB 三星 T7：GazeCapture、EVE_dataset、MPIIFaceGaze、Gaze360、xgaze_224 官方预处理版）与 `/media/yanglinxuan/Expansion/`（10TB 希捷：xgaze_raw 数据/标注/标定）；**预处理产物统一在 `/media/yanglinxuan/ylx/`**（另一块三星 T7：xgaze / mpiifacegaze / gazecapture / eve 四套 `*_insightface_224`）；**特征点索引在各原始数据集目录的 `landmarks/`**（Expansion/xgaze_raw/data、zyx 各数据集根下），预处理配置的 `landmarks_dir` 已默认填好——**landmarks 模式默认开启**（从特征点 h5 索引遍历，跳过 insightface 检测）。
   - ETH-XGaze：**训练/评测用自预处理版 `/media/yanglinxuan/ylx/xgaze_insightface_224/`**（h5 在根目录，80 被试自划分 75 train + 5 test（subject0106~0111），划分列表在 `configs/datasets/zhang2015-insightface/xgaze.yaml`）；官方预处理版 `/media/yanglinxuan/zyx/xgaze_224/`（train/ 下 80 被试）**仅作参考对照**，其 `test/` 是官方无标注集不要使用。**不重跑 xgaze 预处理**（勿覆盖 ylx 正式产物）
   - GazeCapture：原始数据 `/media/yanglinxuan/zyx/GazeCapture`（1474 个五位数 session 目录：`frames/` + 帧级 json；内参 xml 在 `calibration/`（15 设备×双分辨率），已随盘迁移）；**官方划分为 session 级**（iTracker reference_metadata.mat 提取，已与 info.json 的 Dataset 字段互证；官方 GazeCapture 代码已不在本机，划分已固化），保存在 `configs/splits/gazecapture_sessions.yaml`：train 1321（官方 val 50 已并入，本平台不用验证集）/ test 150 session，excluded 3 个（01185/01730/02065，预处理跳过）；**不使用官方 appleFace/eye bbox 与 IsValid**——人脸定位完全走 insightface 管线，帧有效性由管线成败决定；视线标签在 dotInfo.json 的 XCam/YCam（相机系平面坐标）。**训练已接入**：处理产物 `/media/yanglinxuan/ylx/gazecapture_insightface_224/{train,test}/<session>.h5`，loader `datasets/gazecapture.py`（无 LOO）；**训练/测试协议参考 FAZE（ICCV19）的被试级筛选（无 few-shot、用全帧；2026-08-26 定稿 train ≥500）**——train 官方 1321 session 中样本 ≥500 → **1069 session / 1,942,187 帧**，test 官方 150 session 中样本 ≥1000 → **123 session / 259,776 帧**（FAZE 只取每人最后 500 帧，本平台用全帧），筛选列表 `configs/splits/gazecapture_faze.yaml`（按预处理产物帧数生成，重跑预处理后需重新生成）；官方全量划分 `gazecapture_sessions.yaml` 仅供预处理使用
   - MPIIFaceGaze：`/media/yanglinxuan/ylx/mpiifacegaze_insightface_224/` 15 个 pXX.h5 直接在根目录（全量 37,667 样本），BGR 存储；单一配置 `configs/datasets/zhang2015-insightface/mpiifacegaze.yaml`，`split.mode` 切协议——`leave_one_out`（折 i 测 p{i:02d}，`--set dataset.split.fold=i`）与 `all_subjects`（15 人全量）；within 脚本把 15 折 + 全量放进**同一实验目录**的子运行（fold_00~fold_14、all）并汇总；cross 的 mpiifacegaze 源取 `--run all` 的 ckpt、目标测试加 `--set dataset.split.mode=all_subjects`（cross 脚本已内置）；训练策略与 XGaze 同一份 method yaml
   - EVE：`/media/yanglinxuan/ylx/eve_insightface_224/{train,test}/<trainNN|valNN>.h5`（自预处理 zhang2015-insightface：4 相机 5Hz，train 889,296 / test 108,095 样本，BGR + lzf）；官方 test01–06 无 Tobii 标注弃用 → **平台 test = 官方 val01–05**；h5 三索引（frame/cam/step_index）+ attrs 自描述回溯原始 mp4 帧；单一配置 `configs/datasets/zhang2015-insightface/eve.yaml`（与 xgaze 同构，无 LOO）；勘探、几何验证与决策记录在 `preprocess/zhang2015-insightface/eve/dataset_report.md`（其中路径为旧机记录）

7. **多卡训练（DDP）**：用哪些卡由 **`configs/common.yaml` 的 `gpus` 列表**指定（默认 `[0,1,2,3]`；运行时 `GPUS=0,1 bash xxx.sh` 环境变量覆盖，优先级更高；直接 `python main.py` 也会读取该文件设 CUDA_VISIBLE_DEVICES——环境已有该变量时跳过）。`scripts/common.sh` 的 `py()` 启动器据此 export CUDA_VISIBLE_DEVICES：列表 1 张卡直接 python，多张卡自动 `torchrun --nproc_per_node=<张数>` DDP；**测试始终单卡**（列表中第一张）。`method.train.batch_size` 语义为**全局批大小**，多卡时按卡数均分到各 rank（全局有效批 = floor(batch_size/world)×world，单卡行为不变）。实验目录/快照/run.log/ckpt **仅主进程（rank 0）写**；epoch 指标经 all_reduce 汇总，与单卡同口径。**ckpt 仍为 opengaze-ckpt v1 裸权重**（保存/加载统一 unwrap DDP，单卡↔多卡 checkpoint 互通，resume 可跨方式续训）。
   **吞吐与 batch 实测（2026-08-26，4×4090，eve 真实数据，ResNet50 fp32）**：单卡计算饱和 ~770 img/s（≥25 样本/卡）；全局 bs50（12/卡）4 卡 DDP 仅 ~800 img/s（单卡 bs50 即 693，DDP 几乎无增益）；bs200（50/卡）~2080 img/s（2.6×，bs400 无进一步增益；num_workers 5/卡即够）。**泛化代价**：bs200 在短程代理上比 bs50 差 ~0.25-0.35°（eve 10 万样本×5ep：2.95→3.18°，bs50 种子方差仅 ±0.03°；MPII fold0 5ep：2.48→2.82°；提 lr 无改善）——**用户已接受该代价，method yaml 默认 batch_size=200**；需要与旧实验严格可比时 `--set method.train.batch_size=50`（单卡 bs50 693 img/s，或 4 卡并行 4 个独立 run 等效吞吐 ~2770 img/s）。单卡跑 bs200 峰值显存 ~16.5G（4090 24G 可跑）。

## 结构

- `main.py` 入口（train / test / resume 三模式；torchrun 启动自动 DDP，见约定 7）
- `preprocess.py` 预处理入口（`--dataset <ds> --method zhang2015-insightface`；配置在 `configs/preprocess/<method>/<ds>.yaml`；管线脚本在 `preprocess/<method>/normalize_<dataset>.py` 按路径 importlib 加载——目录名含连字符非合法包名）
- `preprocess/` 预处理：`common.py`（FailureRecorder 失败样本记录 + 运行目录）；管线目录 `zhang2015-insightface/`（Zhang2015 归一化 + insightface 关键点）：`normalize_xgaze.py`（XGazePreprocessor 类）、`normalize_mpiifacegaze.py`、`face_model_xgaze.txt`、`gazecapture/`（**已实现已验证**：GazeCapturePreprocessor + 方向映射 docstring；`generate_calibration.py` 生成内参 xml → `/media/yanglinxuan/zyx/GazeCapture/calibration/`（15 设备×双分辨率，cam00.xml 同构）；`calibrate_fx.py` 为 fx 自标定负结果实验；`front_cameras.yaml` 设备分组与 fx；`dot_transfer.md` 四朝向 dot→CCS 定稿公式；`obtain_camera_intrinsics.md` 内参获取全记录）、`eve/`（**已实现已验证**：EVEPreprocessor，4 相机 5Hz，几何链路与公式验证见 `eve/dataset_report.md`，D1–D5 决策已定）、`extract_landmarks.py`（从四套预处理产物抽取 `facial_landmarks_2d`+索引字段 → **各原始数据集目录** `<raw_data_dir>/landmarks/`，每人/session 一个轻量 h5，eve 含 steps/cameras attrs）；四个预处理器均支持 **landmarks 模式**——配置 `landmarks_dir` 非空即从 landmarks h5 索引遍历帧集、跳过 insightface 检测、特征点直读，与原始模式输出逐字节一致（gazecapture/eve 已验证；xgaze/mpii 原数据出自旧脚本，重跑差异仅末位浮点噪声 ≤0.035°）；另一管线 `zhang2015-specific-face-model/get_face_model/`（逐数据集的逐人/session 刚体人脸模型生成，**四数据集全部实现**）：共享核心 `face_model_core.py`（自 xgaze 实现提炼的单相机多帧联合 BA，fx 参数化）+ 各数据集端口 `personalized_face_model.py`——xgaze（方案文档 `personalized_face_modeling.md`，逐相机组；脚本自包含未走 core）、mpiifacegaze（单人单相机组 cam00，内参逐人 Camera.mat）、gazecapture（逐 session×朝向组 ori1~4，内参按设备×分辨率考证 xml；**两段式选择**——默认 <40° 严格段 + `--max-angle 65` 补跑段配组级质量门槛（test ≤2px、IOD 80~100mm），覆盖 1420/1471 session（帧级 79.4%），test 中位 0.97px/改善 2.2x）、eve（逐相机组 cam00~03，内参逐相机 h5、已去畸变）；观测一律取自各原始数据集的 landmarks 索引 h5；**输出统一在 ylx 盘** `/media/yanglinxuan/ylx/<ds>_specific_face_model/face_models/<id>/{组名}_model6.txt|_model28.txt + canonical + summary`（组名：xgaze/eve=cam{cc}、gazecapture=ori{o}、mpii=cam00；下游按「组名+模型」查找可跨数据集通用）；指标留档各端口目录 `metrics/`（含通用模型基线对比）。**下游归一化回退约定**（未来 specific-face-model 归一化管线遵循）：查 `<id>/summary.txt` 存在则用其 `{组名}_model6.txt`，组缺失回退通用 `face_model_xgaze.txt`；**GazeCapture 例外（2026-08-27 定稿）：建模失败的 session（协议池内 14 train + 1 test）不预处理、不参与训练测试**——预处理范围 = FAZE 筛选 ∩ 建模成功（1177 session，名单 `configs/splits/gazecapture_sfm.yaml`）；预处理器两种形式：`normalize_<ds>.py` 文件或 `<ds>/__init__.py` 目录包，均暴露 `run(config, recorder)`；每次运行留档 `preprocess/<method>/log/<dataset>_<时间戳>/{run.log, failures.json}`（日志归属各管线目录）；失败/跳过帧逐条记录（imread_failed / no_face_detected / no_annotation / invalid_dot / error:*）
- `utils/normalization.py` 归一化管线纯函数（estimateHeadPose / normalizeData_face / vector_to_angles），无常量、无数据集逻辑
- `scripts/` 运行脚本：**`<预处理管线>/<method>/<mode>/`** 三级（如 `scripts/zhang2015-insightface/resnet50/within-dataset/xgaze.sh`），`resnet18|resnet50 × within-dataset|cross-dataset`。within-dataset = 训练+测试一条龙（每次新开实验，训练自动多卡 `py` 启动、测试单卡；断点续训直接用 main.py --resume）；cross-dataset = 只做评测，`scripts/common.sh` 的 `require_exp` 强制 `REUSE_EXP=expNN` 显式指定实验，不指定/不存在/快照不符直接报错（dataset_config 现为「管线/数据集」形式）；`all.sh` 按源数据集用 `<大写名>_EXP` 指定，未指定的源跳过；测试结果存 `exp/expNN/test_result_<dataset>.json`
- `trainers/trainer.py` 训练与评测循环（DDP 封装/unwrap、主进程日志与 ckpt、指标 all_reduce）
- `datasets/` 四数据集 loader（xgaze / mpiifacegaze / eve / gazecapture，均已接入训练）+ `base.py` 统一 h5 基类与 `make_train_loader`（DDP 时挂 DistributedSampler）+ `__init__.py` 工厂（新增数据集注册到 DATASETS；gazecapture 的 session 列表从 `configs/splits/` 读）
- `models/` GazeNet（backbone + FC），BACKBONES 注册表
- `utils/` config（yaml 加载合并、--set 覆盖）、logger（实验目录 + 分级日志）、metrics（angular_error）
- 设计文档：STRUCTURE.md（含各模块设计细节与数据集接入计划）

## 预处理注意

- 本机环境：insightface **1.0.1** + onnxruntime **1.29.0**（⚠️ 当前装的是 **CPU 版** `onnxruntime`，providers 无 CUDAExecutionProvider——跑预处理前需 `pip uninstall onnxruntime && pip install onnxruntime-gpu`（CUDA 13 系配 1.23+），否则 insightface 静默跑 CPU）
- onnxruntime-gpu 需 `LD_LIBRARY_PATH` 含 pip 版 nvidia 库：本机布局为 `site-packages/nvidia/{cu13,cudnn}/lib`（cu13/lib 内含 cudart/cublas）；本环境**未配 conda activation 脚本**，直接用绝对路径 python 跑预处理时手动前缀 `LD_LIBRARY_PATH=<env>/lib/python3.12/site-packages/nvidia/cu13/lib:<env>/lib/python3.12/site-packages/nvidia/cudnn/lib`
- 预处理会**覆盖** output_dir 下的同名 h5；调试用 `--set output_dir=... subjects=... max_days/max_frames=...`
