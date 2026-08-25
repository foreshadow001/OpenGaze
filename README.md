# Gaze: 通用视线估计训练与测试平台

基于 [ETH-XGaze](https://github.com/ETH-VISLAB/ETH-XGaze) 官方代码库构建的多数据集视线估计（Gaze Estimation）通用训练与测试平台。

## 支持的数据集

| 数据集 | 状态 | 说明 |
| --- | --- | --- |
| ETH-XGaze | ✅ 已接入 | 75 train / 5 test 自划分（官方 test 无公开标注） |
| MPIIFaceGaze | ✅ 已接入 | leave-one-out 15 折 + 15 人全量（cross 用），`mpiifacegaze_insightface_224` |
| EVE | 🚧 计划中 | 事件相机视线数据集 |
| GazeCapture | 🚧 计划中 | 移动设备 crowdsourcing 视线数据集 |

## 项目结构

```
Gaze/
├── configs/
│   ├── datasets/        # 数据集配置：数据位置、划分（已写入 yaml）、加载参数
│   └── methods/         # 方法配置：模型结构 + 训练策略（resnet18 / resnet50）
├── scripts/             # 运行脚本
│   ├── common.sh        #   公共函数：python 路径、latest_exp、require_exp 校验
│   └── resnet18|resnet50/
│       ├── within-dataset/    # 每数据集一个：训练+测试一条龙（新开实验）
│       └── cross-dataset/     # n(n-1) 个 A→B 评测（REUSE_EXP 指定）+ all.sh
├── datasets/            # 数据集加载（统一 h5 读取基类 + 各数据集划分逻辑 + 工厂）
├── preprocess/          # 数据预处理（管线目录 zhang2015-insightface/；入口 preprocess.py）
├── models/              # 模型（GazeNet：backbone + FC，ResNet 骨干）
├── trainers/            # 训练器（tqdm 训练循环、测试评测、checkpoint 管理）
├── utils/               # yaml 配置加载合并、实验目录管理 logger、指标
├── main.py              # 入口
├── pyproject.toml       # 项目与依赖声明
└── exp/                 # 实验输出（exp00、exp01…自增；每次实验自包含，gitignore）
```

设计与实施细节见 [STRUCTURE.md](STRUCTURE.md)。

## 快速开始

```bash
# 训练：自动创建 exp/expNN（含配置快照、ckpt、log）
python main.py --dataset xgaze --method resnet50

# 断点续训：只指定实验目录，配置以快照为准，从最新 ckpt 完整恢复
python main.py --resume exp01

# 测试：加载 exp00 中最新 epoch 的 checkpoint
python main.py --dataset xgaze --method resnet50 --test --exp exp00

# 跨数据集评测：ckpt 取自 exp00，测试集按 --dataset 现场构建
python main.py --dataset mpiifacegaze --method resnet50 --test --exp exp00

# 临时覆盖配置项（点路径）
python main.py --dataset xgaze --method resnet50 --set method.train.epochs=2

# 冒烟测试（1 被试 10 帧，验证管线）
python main.py --dataset xgaze_smoke --method resnet50 --set method.train.epochs=2
```

也可以用 `scripts/` 下的脚本：

```bash
# within-dataset：训练 + 测试一条龙（每次运行新开一个实验，无需参数）
bash scripts/resnet50/within-dataset/xgaze.sh

# cross-dataset：只做评测，训练须先完成，必须 REUSE_EXP 指定用哪次实验
python main.py --dataset xgaze --method resnet50        # 先训练
REUSE_EXP=exp00 bash scripts/resnet50/cross-dataset/xgaze_mpiifacegaze.sh

# 全矩阵：每个源数据集分别指定实验，未指定的源自动跳过其组合
XGAZE_EXP=exp00 MPIIFACEGAZE_EXP=exp03 bash scripts/resnet50/cross-dataset/all.sh
```

- within-dataset 每次新开实验，断点续训不属脚本职责，直接 `python main.py --resume expNN`
- cross-dataset **必须显式指定实验**：不指定、实验不存在、或快照与「数据集 × 方法」不符时直接报错，不做自动选择
- 测试结果按数据集存为 `exp/expNN/test_result_<dataset>.json`，互不覆盖

## 数据预处理

各数据集归一化为统一 h5 格式（insightface 106 点 PnP + 虚拟相机归一化），入口 `preprocess.py`，配置在 `configs/preprocess/`：

```bash
conda activate opengaze   # 必须：activation 脚本为 onnxruntime-gpu 配置 cudnn 库路径

python preprocess.py --dataset mpiifacegaze --method zhang2015-insightface             # 全量 15 人
python preprocess.py --dataset mpiifacegaze --method zhang2015-insightface \
    --set 'subjects=["p00"]' max_days=1                                                # 调试小样本
python preprocess.py --dataset xgaze --method zhang2015-insightface --set 'subjects=[0, 3]'
```

每次运行在 `preprocess/<method>/log/<dataset>_<时间戳>/` 留档 `run.log`（分级日志）与 `failures.json`（**失败/跳过帧逐条记录**：读图失败 / 检测不到脸 / 无标注 / 异常，含按原因汇总），供核对数据完整性。预处理脚本按管线组织在 `preprocess/zhang2015-insightface/`（Zhang2015 归一化 + insightface 关键点；xgaze 为 `XGazePreprocessor` 类，相机翻转/排除表与 face model 独立配置加载）。注意：预处理会覆盖输出目录同名 h5，调试务必用 `--set output_dir=...` 指向临时目录。

实验目录结构（每次训练自包含，可复现）：

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
| `model_state` | OrderedDict | 模型权重 `state_dict` |
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
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

## 致谢

- [ETH-XGaze](https://github.com/ETH-VISLAB/ETH-XGaze): Zhang et al., "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation", ECCV 2020.
