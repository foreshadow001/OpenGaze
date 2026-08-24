#!/usr/bin/env bash
# resnet18 / cross-dataset: 用 mpiifacegaze 上训练的模型（expNN）在 gazecapture 上测试
# 训练先完成: python main.py --dataset mpiifacegaze --set dataset.split.mode=all_subjects --method resnet18
# 用法: REUSE_EXP=expNN bash scripts/resnet18/cross-dataset/mpiifacegaze_gazecapture.sh
set -euo pipefail
source "$(dirname "$0")/../../common.sh"

exp=$(require_exp mpiifacegaze resnet18)
"$PYTHON" main.py --dataset gazecapture --method resnet18 --test --exp "$exp" --run all
