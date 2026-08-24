#!/usr/bin/env bash
# resnet50 / cross-dataset: 用 mpiifacegaze 上训练的模型（expNN）在 eve 上测试
# 训练先完成: python main.py --dataset mpiifacegaze --set dataset.split.mode=all_subjects --method resnet50
# 用法: REUSE_EXP=expNN bash scripts/resnet50/cross-dataset/mpiifacegaze_eve.sh
set -euo pipefail
source "$(dirname "$0")/../../common.sh"

exp=$(require_exp mpiifacegaze resnet50)
"$PYTHON" main.py --dataset eve --method resnet50 --test --exp "$exp" --run all
