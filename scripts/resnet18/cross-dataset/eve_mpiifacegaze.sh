#!/usr/bin/env bash
# resnet18 / cross-dataset: 用 eve 上训练的模型（expNN）在 mpiifacegaze 上测试
# 训练先完成: python main.py --dataset eve --method resnet18
# 用法: REUSE_EXP=expNN bash scripts/resnet18/cross-dataset/eve_mpiifacegaze.sh
set -euo pipefail
source "$(dirname "$0")/../../common.sh"

exp=$(require_exp eve resnet18)
"$PYTHON" main.py --dataset mpiifacegaze --method resnet18 --test --exp "$exp" --set dataset.split.mode=all_subjects
