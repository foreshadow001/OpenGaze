# 项目结构与实施方案

> 本文档描述通用视线估计训练/测试平台的完整设计，确认后按此实施。
> 已确认的三项决策：**分包结构** / **预处理成统一 h5 格式** / **224×224 输入 + pitch/yaw 标签**。

## 1. 项目结构

```
Gaze/
├── configs/
│   ├── datasets/                     # 数据集配置：数据位置、划分方式、加载参数
│   │   ├── xgaze.yaml
│   │   ├── mpiifacegaze.yaml
│   │   ├── gazecapture.yaml
│   │   └── eve.yaml
│   ├── methods/                      # 方法配置：模型结构 + 训练策略
│   │   ├── resnet18.yaml
│   │   └── resnet50.yaml
│   └── preprocess/                   # 预处理配置（输入输出路径、被试列表、线程数）
│       ├── xgaze.yaml                #   xgaze 特有：annotation_dir / calib_dir / sub_folder
│       ├── mpiifacegaze.yaml
│       └── gazecapture.yaml          #   引用 splits/ 下的官方 session 级划分
│   └── splits/                       # 数据集官方划分（独立共享：预处理与训练配置都引用）
│       └── gazecapture_sessions.yaml #   GazeCapture：train 1271 / val 50 / test 150 session + excluded 3
│
├── datasets/                         # 数据集加载（统一 h5 格式）
│   ├── __init__.py                   #   build_train_loader / build_test_loader 工厂
│   ├── base.py                       #   GazeH5Dataset：统一 h5 读取基类（懒加载 + swmr）
│   ├── xgaze.py                      #   ETH-XGaze：按配置 split 段读取（官方数据已是 h5，直接兼容）
│   ├── mpiifacegaze.py               #   leave_one_out / all_subjects 两种 split 模式
│   ├── gazecapture.py                #   读预处理后的 h5（官方 train/val/test 划分，待实现）
│   └── eve.py                        #   读预处理后的 h5（官方 train/val/test 划分，待实现）
│
├── models/
│   ├── __init__.py                   #   build_model() 工厂：按 config.model.name 分发
│   ├── gaze_net.py                   #   gaze_network：backbone + FC(→2)
│   └── resnet.py                     #   ResNet 骨干（自 modules/resnet.py 迁移）
│
├── utils/
│   ├── __init__.py
│   ├── config.py                     #   yaml 加载 → namespace 对象（支持递归访问 config.dataset.name）
│   ├── logger.py                     #   实验目录管理（exp/expNN 自动递增）、tensorboard、文本日志、配置快照落盘
│   └── metrics.py                    #   AverageMeter / angular_error / pitchyaw↔vector（自 utils.py 迁移）
│
├── preprocess/                       # 预处理（管线脚本目录名含连字符，由 preprocess.py 按路径加载）
│   ├── common.py                     #   FailureRecorder 失败样本记录 + 运行目录/日志
│   ├── zhang2015-insightface/        #   管线：Zhang2015 归一化 + insightface 106 点 PnP
│   │   ├── normalize_xgaze.py        #     XGazePreprocessor 类（FLIP/EXCLUDE 相机表、face model 独立配置加载）
│   │   ├── normalize_mpiifacegaze.py #     MPIIFaceGaze（同管线，独立常量）
│   │   ├── gazecapture/              #     GazeCapture 预处理（目录包形式，骨架待实现）
│   │   └── face_model_xgaze.txt      #     3D face model 数据
│   └── log/                          #   每次预处理一个目录（run.log + failures.json，gitignore）
│
├── trainers/
│   ├── __init__.py
│   └── trainer.py                    # 训练器（合并原 main.py 中的 tqdm 补丁；输出统一写入 exp/expNN）
│
├── scripts/                          # 运行脚本
│   ├── common.sh                     #   公共函数：python 路径、latest_exp、require_exp 校验
│   ├── resnet18/                     #   结构与方法一一对应
│   │   ├── within-dataset/           #     每数据集一个：训练+测试一条龙（每次新开实验）
│   │   └── cross-dataset/            #     n(n-1) 个 A→B 评测（REUSE_EXP 指定）；all.sh 按源 <大写名>_EXP
│   └── resnet50/                     #   同上
│
├── main.py                           # 训练/测试入口：python main.py --dataset xgaze --method resnet50
├── preprocess.py                     # 预处理入口：python preprocess.py --dataset mpiifacegaze
├── README.md
├── STRUCTURE.md
├── pyproject.toml                    # 项目与依赖声明（pip install -e .）
├── .gitignore
│
└── exp/                              # 实验输出根目录（gitignore）
    └── exp00/                        #   每次训练自动递增创建，自包含
        ├── config.yaml               #   合并后的完整配置快照（复现依据）
        ├── method.yaml               #   所用 configs/methods/*.yaml 的副本
        ├── ckpt/                     #   checkpoint（epoch_N_ckpt.pth）
        └── logs/                     #   tensorboard
```

### 原文件迁移对照

| 原文件（已复制） | 去向 |
| --- | --- |
| `main.py` | 重写为 yaml 驱动入口；tqdm 训练循环并入 `trainers/trainer.py` |
| `trainer.py` | 迁至 `trainers/trainer.py` 并重构：合并 tqdm 版 `train_one_epoch`、输出经 `utils/logger.py` 写入 `exp/expNN`、增加 test 集逐样本误差统计 |
| `data_loader.py` | 拆分：通用 h5 读取 → `datasets/base.py`；XGaze 划分逻辑 → `datasets/xgaze.py` |
| `model.py` + `modules/resnet.py` | `models/gaze_net.py` + `models/resnet.py` |
| `utils.py` | `utils/metrics.py` |
| `train_test_split.json` | 并入 `configs/datasets/xgaze.yaml` 的 split 段 |
| `config.py`（未复制） | 由 `utils/config.py` + yaml 取代 |

## 2. 统一数据格式规范

所有数据集预处理为同一目录结构，训练代码只面对一种格式：

```
<processed_root>/
├── train/
│   ├── p000.h5          # 文件名按数据集自然分片（被试/shard）
│   │     ├─ face_patch      (N, 224, 224, 3)  uint8   RGB
│   │     ├─ face_gaze       (N, 2)            float32 (pitch, yaw) 弧度
│   │     └─ face_head_pose  (N, 2)            float32 (pitch, yaw) 弧度（可选，头姿）
│   └── ...
└── test/
    └── ...
```

约定（与 ETH-XGaze 官方训练管线一致）：

- 输入：224×224 RGB 人脸 patch，`uint8` 存储，加载时 ToTensor + ImageNet mean/std 归一化
- 标签：2 维 `(pitch, yaw)` 弧度；`pitch` = 纵向角（sin 分量为 y），`yaw` = 横向角
- 损失：L1；指标：角度误差（度）
- ETH-XGaze 官方 h5 内部为 BGR 存储，`datasets/xgaze.py` 加载时做 BGR→RGB 翻转（保持官方行为）；其余数据集预处理时直接存 RGB

> 数据在盘：`/media/hitsz/ylx/` 下已有 `xgaze_224`、`MPIIFaceGaze`、`GazeCapture`、`EVE_dataset`（及 Gaze360，可作为后续扩展）。
> 另有已预处理的 `MPIIFaceGaze_normalized`、`mpiifacegaze_insightface_224`、`xgaze_insightface_224` 和 `~/data-preprocessing-gaze` 脚本，预处理阶段可参考复用。

## 3. yaml 配置设计

配置沿**数据集 × 方法**两个正交轴组织，运行时合并为一份完整配置；实验标识为自增的 `exp/expNN` 目录（见 3.3），`{dataset}_{method}`（leave-one-out 时含折号）作为实验描述记录在配置快照中。

### 3.1 数据集配置（configs/datasets/*.yaml）

存数据位置、训练/评测协议（固定划分或 leave-one-out）、划分信息、加载参数。

`configs/datasets/xgaze.yaml`（官方 test 集无公开标注，从 train/ 目录 80 个被试中自划分 75 train + 5 test，划分列表直接写入配置）：

```yaml
name: xgaze
data_dir: /media/hitsz/ylx/xgaze_224

split:
  train:
    sub_folder: train
    subjects:                       # 75 个被试（subject0000.h5 …）
      - subject0000.h5
      - subject0003.h5
      - …
  test:
    sub_folder: train               # 与 train 同目录，被试 subject0106~0111
    subjects:
      - subject0106.h5
      - …

dataloader:
  num_workers: 5
  train_sample_size: 0              # 0 = 全量训练
  test_sample_size: 0
```

`configs/datasets/mpiifacegaze.yaml`（单一配置，`split.mode` 切换两种协议）：

```yaml
name: mpiifacegaze
data_dir: /media/hitsz/ylx/mpiifacegaze_insightface_224

split:
  mode: leave_one_out               # leave_one_out / all_subjects（见下）
  fold: 0                           # 0~14；--set dataset.split.fold=i 切换折
  sub_folder: ''                    # h5 直接在 data_dir 下（无子目录）
  subjects:                         # p00.h5 ~ p14.h5 共 15 个被试
    - p00.h5
    - …

dataloader:
  num_workers: 5
  train_sample_size: 0
  test_sample_size: 0
```

- `leave_one_out`：折 i 以 p{i:02d} 为测试被试，其余 14 人训练（within-dataset 协议）
- `all_subjects`：15 人全量（train = test = 全部），`--set dataset.split.mode=all_subjects` 切换；LOO 之后自动追加一次全量训练，也是 cross-dataset 的源训练 / 目标测试协议（cross 脚本已内置该覆盖）

within-dataset 协议：15 折 leave-one-out + 全量训练放在**同一个实验目录**的子运行中（`fold_00~fold_14`、`all`，`--run` 指定）；每个子运行保存自己的 **run 级快照**（含该折的 fold 划分），resume / test 都以 run 级快照为准，顶层另有一份实验级快照供 require_exp 匹配；`scripts/*/within-dataset/mpiifacegaze.sh` 自动循环全部折并汇总 mean±std，随后自动追加全量训练，**脚本可安全重跑**（已完成折自动跳过、中断折从最新 ckpt 续训）；训练策略与 XGaze 完全一致（同一份 method yaml）。

### 3.2 方法配置（configs/methods/*.yaml）

存算法结构 + 训练策略，与数据集无关。

`configs/methods/resnet50.yaml`：

```yaml
name: resnet50

model:
  backbone: resnet50                # 对应 models/ 工厂注册名
  pretrained: true
  num_out: 2                        # pitch, yaw

train:
  batch_size: 50
  epochs: 25
  loss: l1
  optimizer: adam
  init_lr: 1.0e-4
  lr_scheduler: step                # step / cosine
  lr_patience: 10                   # StepLR step_size
  lr_decay_factor: 0.1
  ckpt_save_interval: 5             # 每 N 个 epoch 保存一次 ckpt（最后一个 epoch 始终保存）

output:
  print_freq: 1000
  use_gpu: true
  seed: 0
```

`resnet18.yaml` 同构，仅 backbone 与超参不同（如 `init_lr: 3.0e-4`）。

### 3.3 实验目录管理与运行

不再使用固定的 `ckpt/`、`logs/` 目录。每次训练在 `exp/` 下自动创建下一个编号目录（扫描已有 `expNN` 取最大值 +1），由 `utils/logger.py` 统一管理。带子运行的实验（如 MPII 的 LOO）用 `--run NAME` 把多次运行放进同一实验目录：

```
exp/exp00/                        # 无子运行（如 xgaze）
├── config.yaml                   # 合并后的完整配置快照（仅实验首次创建时写）
├── method.yaml                   # 所用 configs/methods/*.yaml 的原样副本
├── ckpt/                         # epoch_0_ckpt.pth、epoch_1_ckpt.pth …
└── logs/                         # tensorboard 事件文件 + 文本运行日志

exp/exp01/                        # 有子运行（MPII 的 LOO：一次完整评测 = 一个实验目录）
├── config.yaml / method.yaml     # 顶层快照（首折创建时写，供 require_exp 匹配）
├── fold_00/                      # 折 0：config.yaml（run 级快照，含该折 fold 值）
│   └── ckpt/ + logs/ + test_result_mpiifacegaze.json
├── fold_01/ … fold_14/           # 每折同构；resume/test 读取各自的 run 级快照
└── all/                          # LOO 后自动追加的 15 人全量训练（cross-dataset 用）
```

```bash
# 训练：自动创建 exp/expNN（编号递增），配置快照自动落盘
python main.py --dataset xgaze --method resnet50

# 训练：显式指定编号（目录已存在则报错拒绝覆盖）
python main.py --dataset xgaze --method resnet50 --exp exp01

# 断点续训：只指定实验目录，配置以 exp01 快照为准，从最新 ckpt 完整恢复
# （model/optim/scheduler/epoch/train_iter），--set 可延长 epochs（快照同步更新）
python main.py --resume exp01 --set method.train.epochs=40

# 测试：指定实验目录，自动加载其中最新 epoch 的 ckpt
python main.py --dataset xgaze --method resnet50 --test --exp exp00

# 跨数据集评测：ckpt 取自 exp00，测试集按 --dataset 构建（阶段 5 的评测矩阵即由此实现）
python main.py --dataset mpiifacegaze --method resnet50 --test --exp exp00
```

要点：

- **日志规范**：所有输出经 `utils.logger.get_logger(__name__)`，格式 `时间 | 级别 | 文件:行号 | 消息`，console 与 `exp/expNN/logs/run.log` 双写；训练、续训、测试的日志按时间序追加在同一 run.log 中，全程可溯源（tqdm 进度条除外）
- **checkpoint 标准**：opengaze-ckpt v1（`format_version`/`epoch`/`train_iter`/`model_state`/`optim_state`/`scheduler_state`），详见 README「Checkpoint 格式标准」；加载时先校验 `format_version`，其他项目格式（含 ETH-XGaze 官方旧 ckpt）一律拒绝
- 测试时以 `--exp` 定位实验目录；ckpt 默认取该目录 `ckpt/` 下编号最大的 epoch，可用 `--ckpt` 指定具体文件
- 训练时的随机种子、CUDA 设置等在快照 `config.yaml` 中留档，保证可复现
- `exp/` 整体进 `.gitignore`，不进版本库

## 4. 环境（conda 环境名：opengaze，由你创建）

```bash
conda create -n opengaze python=3.12 -y
conda activate opengaze

# 第一步：PyTorch 2.13.0 + CUDA 12.6（RTX 4060 Ti；系统 CUDA 驱动 12.8 向下兼容 cu126 运行时；
# cu126 轮子在官方专用索引上，需先单独安装）
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# 第二步：安装本项目及其余依赖（依赖声明见 pyproject.toml）
pip install -e .
```

依赖清单（声明于 `pyproject.toml`，`pip install -e .` 一并安装）：

| 包 | 用途 |
| --- | --- |
| python 3.12 | 与 torch 2.13 cp312 轮子配套 |
| torch 2.13.0 / torchvision 0.28.0 (cu126) | 训练框架、transforms、ResNet 预训练权重；轮子自带 CUDA 运行时与 cuDNN，不依赖系统 CUDA Toolkit |
| numpy | 数值计算、角度误差 |
| h5py | 统一 h5 数据格式读写（swmr 懒加载） |
| pyyaml | yaml 配置解析 |
| tensorboard | 训练曲线 |
| tqdm | 进度条 |
| opencv-python | 预处理：图像读取、人脸检测/对齐、归一化变换 |
| scipy | 读取 .mat 标注（MPIIFaceGaze / EVE） |
| matplotlib | 预处理与调试可视化（可选） |

## 5. 实施阶段

1. **框架搭建**：分包迁移、yaml 配置系统、`utils/logger.py` 实验目录与分级日志管理、trainer 重构、ckpt v1 标准与断点续训；已用 ETH-XGaze 冒烟验证（1 被试 10 帧：训练→续训→测试）✅
2. **MPIIFaceGaze**：数据已预处理为统一 h5（`mpiifacegaze_insightface_224`，见 `~/data-preprocessing-gaze/normalize_mpiifacegaze_h5.py`）；LOO 15 折 + 全量配置与脚本已接入，冒烟验证通过 ✅（全量训练待跑）
3. **GazeCapture**：预处理脚本（jpg + json → h5，官方 train/val/test 划分）→ 训练 + 测试
4. **EVE**：预处理脚本（视频帧抽帧 + 标注 → h5）→ 训练 + 测试
5. **跨数据集评测**：train on A / test on B 评测矩阵，验证平台通用性
