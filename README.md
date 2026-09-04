# OpenGaze: 通用视线估计训练与测试平台

基于 [ETH-XGaze](https://github.com/ETH-VISLAB/ETH-XGaze) 官方代码库演化的多数据集视线估计（Gaze Estimation）训练与评测平台。四数据集全部接入，支持单卡 / 多卡（DDP）训练。

## 支持的数据集

| 数据集 | 状态 | 划分协议 |
| --- | --- | --- |
| ETH-XGaze | ✅ 已接入 | 80 被试自划分 75 train / 5 test（官方 test 无公开标注） |
| MPIIFaceGaze | ✅ 已接入 | leave-one-out 15 折 + 15 人全量（`--set dataset.split.mode` 切换） |
| EVE | ✅ 已接入 | train01–39 训练 / 官方 val01–05 为平台 test（官方 test 无 Tobii 标注） |
| GazeCapture | ✅ 已接入 | FAZE 式被试筛选（全帧、无 few-shot）：train 样本 ≥500 → 1069 session / 1.94M 帧；test 样本 ≥1000 → 123 session / 260k 帧 |

### 预处理管线（三版，`--dataset` 传「管线/数据集」）

| 管线 | 归一化 | 产物 | 协议文档 |
| --- | --- | --- | --- |
| `zhang2015-insightface`（v1） | Zhang2015 虚拟相机（消头姿）+ insightface PnP | `<ds>_insightface_224` | — |
| `zhang2015-specific-face-model`（v2） | 同 v1 归一化；xgaze/EVE 逐人 true6_canonical + DLT 头姿，GC/MPII gen_xe6 + PnP | `<ds>_specific_224` | [normalization_protocol.md](preprocess/zhang2015-specific-face-model/normalization_protocol.md) |
| `ours-without-roll`（v3） | **fixed_forward=True roll-only**（头姿 pitch/yaw 保留在 patch、roll=0）；几何从 v2 产物精确恢复，无 DLT/PnP | `<ds>_noroll_224` | [normalization_protocol.md](preprocess/ours-without-roll/normalization_protocol.md) |

所有数据集预处理为统一 h5：`face_patch (N,224,224,3) uint8` + `face_gaze (N,2) (pitch,yaw) 弧度`（v2/v3 另有 `face_gaze_hcs` 头架系标签），BGR 存储、加载时统一翻转，face_patch 逐样本 chunk。**训练读取统一走 `/data/others_preprocessed_datasets/`（NVMe，十二套 v1/v2/v3 副本）；ylx/sfm 盘为预处理写出与正本**。

## 项目结构

```
OpenGaze/
├── configs/
│   ├── common.yaml                    # 平台公共配置（gpus 用卡列表）
│   ├── datasets/                      # 数据集配置（按预处理管线分文件夹：zhang2015-insightface /
│   │                                  #   zhang2015-specific-face-model / ours-without-roll，含 label 字段）
│   ├── methods/                       # 方法配置：resnet18 / resnet50
│   ├── preprocess/                    # 预处理配置（同按管线分文件夹）
│   └── splits/                        # 官方/筛选划分（GazeCapture 两份：全量预处理用 + FAZE 筛选训练用）
├── scripts/
│   ├── common.sh                      # python 路径、latest_exp / require_exp 校验、py() 多卡启动器
│   ├── zhang2015-insightface|zhang2015-specific-face-model/   # 一级 = 预处理管线，二级 = 方法
│   │   └── resnet18|resnet50/{within-dataset,cross-dataset}/
│   └── ours-without-roll-hcs|ccs/     # v3 按标签分两套（内置 --label），结构同上
├── datasets/                          # 四数据集 loader + 统一 h5 基类（label_field_of 选 ccs/hcs）+ 工厂
├── preprocess/                        # 预处理管线（zhang2015-insightface / zhang2015-specific-face-model /
│                                      #   ours-without-roll + repack_chunks.py 存量修复工具）
├── models/                            # GazeNet（backbone + FC，ResNet 骨干）
├── trainers/                          # 训练器（DDP 封装、训练/评测循环、checkpoint 管理）
├── utils/                             # yaml 配置、实验目录 logger、指标
├── main.py                            # 入口（train / test / resume）
├── preprocess.py                      # 预处理入口
└── exp/                               # 实验输出（expNN 自增，每次自包含，gitignore）
```

设计与实施细节见 [STRUCTURE.md](STRUCTURE.md)，开发约定见 [CLAUDE.md](CLAUDE.md)。

## 快速开始

`--dataset` 为「预处理管线/数据集」子路径：

```bash
# 训练：自动创建 exp/expNN（含配置快照、ckpt、log）；4 卡 DDP 见下节
python main.py --dataset zhang2015-insightface/xgaze --method resnet50

# 断点续训：只指定实验目录，配置以快照为准，从最新 ckpt 完整恢复
python main.py --resume exp01

# 测试：加载 exp00 中最新 epoch 的 checkpoint
python main.py --dataset zhang2015-insightface/xgaze --method resnet50 --test --exp exp00

# 跨数据集评测：ckpt 取自 exp00，测试集按 --dataset 现场构建
python main.py --dataset zhang2015-insightface/mpiifacegaze --method resnet50 --test --exp exp00

# 临时覆盖配置项（点路径）
python main.py --dataset zhang2015-insightface/xgaze --method resnet50 --set method.train.epochs=2

# 冒烟测试（1 被试 10 帧，验证管线）
python main.py --dataset zhang2015-insightface/xgaze_smoke --method resnet50 --set method.train.epochs=2

# v3 管线 + 头架系标签训练（--label ccs|hcs，默认 ccs；快照留档、测试/跨数据集自动沿用）
python main.py --dataset ours-without-roll/mpiifacegaze --method resnet18 --label hcs
```

也可以用 `scripts/` 下的脚本：

```bash
# within-dataset：训练 + 测试一条龙（每次运行新开一个实验，训练自动多卡）
bash scripts/zhang2015-insightface/resnet50/within-dataset/xgaze.sh

# cross-dataset：只做评测，训练须先完成，必须 REUSE_EXP 指定用哪次实验
python main.py --dataset zhang2015-insightface/xgaze --method resnet50   # 先训练
REUSE_EXP=exp00 bash scripts/zhang2015-insightface/resnet50/cross-dataset/xgaze_mpiifacegaze.sh

# 全矩阵：每个源数据集分别指定实验，未指定的源自动跳过其组合
XGAZE_EXP=exp00 MPIIFACEGAZE_EXP=exp03 bash scripts/zhang2015-insightface/resnet50/cross-dataset/all.sh

# v3：按标签分两套脚本（内置 --label hcs/ccs，LOO 复用按「数据集×方法×label」匹配）
bash scripts/ours-without-roll-hcs/resnet50/within-dataset/xgaze.sh
```

- within-dataset 每次新开实验；MPII 的 15 折 LOO + 全量在同一实验目录的子运行（fold_00~14、all），脚本可安全重跑（已完成折跳过、中断折自动续训）。断点续训不属脚本职责，直接 `python main.py --resume expNN [--run fold_XX]`
- cross-dataset **必须显式指定实验**：不指定、实验不存在、或快照与「数据集 × 方法」不符时直接报错，不做自动选择
- 测试结果按数据集存为 `exp/expNN/test_result_<dataset>.json`，互不覆盖

## 多卡训练（DDP）

- 用哪些卡：改 **`configs/common.yaml` 的 `gpus` 列表**（默认 `[0,1,2,3]`）；临时覆盖用环境变量 `GPUS=0,1 bash xxx.sh`（优先级更高）
- `scripts/common.sh` 的 `py()` 启动器据此 export `CUDA_VISIBLE_DEVICES`：列表 1 张卡直接 python，多张卡自动 `torchrun --nproc_per_node=<张数>`；**测试始终单卡**（列表第一张）
- `method.train.batch_size` 语义为**全局批大小**，多卡时按卡数均分到各 rank，单卡行为不变。默认 200（4×4090 吞吐 2083 vs 794 img/s；较 bs50 泛化差 ~0.3°，已接受的权衡）——需要严格 bs50 语义时 `--set method.train.batch_size=50`
- 实验目录 / 快照 / run.log / ckpt 仅主进程（rank 0）写；epoch 指标经 all_reduce 汇总，与单卡同口径
- **checkpoint 仍是 opengaze-ckpt v1 裸权重**（保存/加载统一 unwrap DDP）：单卡 ↔ 多卡互通，resume 可跨方式续训

## 数据预处理

各数据集归一化为统一 h5 格式（Zhang2015 虚拟相机归一化 + insightface 106 点 PnP 头姿），入口 `preprocess.py`，配置在 `configs/preprocess/<管线>/<数据集>.yaml`：

```bash
python preprocess.py --dataset mpiifacegaze --method zhang2015-insightface            # v1 全量 15 人
python preprocess.py --dataset mpiifacegaze --method zhang2015-insightface \
    --set 'subjects=["p00"]' max_days=1                                               # 调试小样本
python preprocess.py --dataset xgaze --method zhang2015-specific-face-model           # v2
python preprocess.py --dataset mpiifacegaze --method ours-without-roll                # v3（roll-only）
```

- v2/v3 的字段、模型与门控差异见各自 `normalization_protocol.md`；v3 以 v2 产物为索引与几何真源（无 DLT/PnP、逐图独立）
- v2/v3 重跑后需把新产物同步到 `/data`（rsync），并确认 face_patch 为逐样本 chunk（存量修复：`preprocess/ours-without-roll/repack_chunks.py`）

- 每次运行在 `preprocess/<method>/log/<dataset>_<时间戳>/` 留档 `run.log` 与 `failures.json`（**失败/跳过帧逐条记录**，含按原因汇总）
- **landmarks 模式默认开启**（各配置 `landmarks_dir` 已指向各原始数据集目录的 `landmarks/` 索引）：从特征点 h5 遍历帧集、跳过 insightface 检测，与原始模式输出一致
- 预处理会**覆盖**输出目录同名 h5；调试务必 `--set output_dir=...` 指向临时目录
- onnxruntime 需 GPU 版（`onnxruntime-gpu`；勿装 CPU 版 `onnxruntime`，否则 insightface 静默跑 CPU）。若报缺 cudnn/cublas，前缀 `LD_LIBRARY_PATH=<env>/lib/python3.12/site-packages/nvidia/{cu13,cudnn}/lib`

### 人脸模型标准坐标系（解剖轴定义）

specific-face-model 管线的人脸模型统一在此标准系下交付（唯一实现
`utils/normalization.py: canonicalize_face_model()`）：

- **零位标准**：roll = 0 ⇔ 两眼中心连线 ∥ x 轴；yaw = 0 ⇔ 眼心—鼻心连线 ⊥ x 轴；pitch = 0 ⇔ 眼心—鼻心连线 ∥ y 轴
- **构造**：x̂ = 眼心连线方向；ŷ = 眼→鼻连线去 x̂ 分量后归一；ẑ = x̂ × ŷ（右手系，指向头内）；原点 = 眼心
- **模型标准化三步管线**（防坐标系漂移，所有模型交付前必经）：① Kabsch 刚体粗对齐到 gen6（无缩放，仅消头运动/初始朝向）→ ② `canonicalize_face_model()` 欧拉角归零 → ③ 中心化到眼心（②③合并于该函数输出）
- **头姿解算与归一化在此标准系下进行**（头姿 pitch/yaw/roll 为解剖角；与 v1 insightface 管线的 gen6 原生系相差 13.46°，跨管线比较头姿读数差 ~13° 属坐标系修正而非 bug）
- **pitch 零位 = 眼→鼻连线空间竖直**（非自然平视）：自然平视时鼻线前倾 ~13.2°（解剖常数），故自然/微低头姿态读约 +1~+16°、正 pitch = 抬头；该前倾是深度倾斜，归一化 patch 中鼻线只显缩短不显倾斜，勿以 2D 目测校验
- 细节与参考数字见 [get_face_model/xgaze/README.md](preprocess/zhang2015-specific-face-model/get_face_model/xgaze/README.md)；本平台标准模型 `canonical_mean6.txt`（IOD 86.1 / 鼻宽 26.7 mm）也在此系下

## 实验目录结构

每次训练自包含，可复现：

```
exp/exp00/                        # 普通实验（如 xgaze）
├── config.yaml                   # 合并后的完整配置快照（仅实验首次创建时写）
├── method.yaml                   # 所用 configs/methods/*.yaml 的原样副本
├── ckpt/                         # epoch_N_ckpt.pth …
├── logs/                         # tensorboard + run.log
└── test_result_<dataset>.json    # 测试结果留档（按测试数据集命名）

exp/exp01/                        # 带子运行的实验（MPII 的 LOO：一次完整评测 = 一个目录）
├── config.yaml / method.yaml     # 顶层快照（首折创建时写）
├── fold_00/ … fold_14/           # 各折：config.yaml（该折实际配置）+ ckpt/ + logs/ + test_result_*.json
└── all/                          # LOO 后自动追加的 15 人全量训练（cross-dataset 用）
```

## Checkpoint 格式标准（opengaze-ckpt v1）

所有 checkpoint 由 `trainers/trainer.py::save_checkpoint` 统一写出，`torch.save` 序列化的字典：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `format_version` | int | 格式版本号，当前为 `1` |
| `epoch` | int | 已完成的 epoch 数（恢复训练时从该索引继续） |
| `train_iter` | int | 已完成的全局迭代数（tensorboard 曲线续接用） |
| `model_state` | OrderedDict | 模型权重 `state_dict`（DDP 下亦为 unwrap 后的裸权重） |
| `optim_state` | dict | 优化器（Adam）状态 |
| `scheduler_state` | dict | lr 调度器状态 |

- 文件命名：`epoch_{N}_ckpt.pth`，`N` 从 0 计，位于 `exp/expNN/[run/]ckpt/`
- 保存频率由 method yaml 的 `train.ckpt_save_interval` 控制（默认 5：每 5 个 epoch 保存一次，节约空间）；**最后一个 epoch 始终保存**，保证训练完成必有最终 ckpt，且 resume / `--test` 取最新 ckpt 的语义不受影响
- **断点续训**依赖 `epoch`、`train_iter`、`optim_state`、`scheduler_state` 全部恢复，因此任何写入新字段的改动都应递增 `format_version`
- 仅支持本标准：不兼容其他项目的 checkpoint 格式（含 ETH-XGaze 官方旧格式），加载时缺字段直接报错

## 环境

```bash
conda create -n opengaze python=3.12 -y
conda activate opengaze
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
pip install -e .
```

## 致谢

- [ETH-XGaze](https://github.com/ETH-VISLAB/ETH-XGaze): Zhang et al., "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation", ECCV 2020.
- GazeCapture 训练协议的被试筛选参考 FAZE: Park et al., "Few-Shot Adaptive Gaze Estimation", ICCV 2019（本平台无 few-shot、用全帧）。
