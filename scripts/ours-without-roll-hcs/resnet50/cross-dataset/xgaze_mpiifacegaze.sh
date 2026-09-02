#!/usr/bin/env bash
# resnet50 / ours-without-roll-hcs / cross-dataset: xgaze 训练的模型 → MPIIFaceGaze 评测（标签 = hcs）
# 只做评测；训练实验必须先完成并显式指定。标签自动取自源实验快照（hcs 训练
# 必须用 hcs 标签评测，跨数据集同样成立）。
# 用法: REUSE_EXP=expNN bash scripts/ours-without-roll-hcs/resnet50/cross-dataset/xgaze_mpiifacegaze.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

exp=$(require_exp "ours-without-roll/xgaze" "resnet50")

# MPII 目标自动取 --run all 的 ckpt 并切 all_subjects（main.py 内置）
"$PYTHON" main.py --dataset ours-without-roll/mpiifacegaze --method resnet50 \
    --test --exp "$exp"
