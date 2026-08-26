#!/usr/bin/env bash
# resnet50 / cross-dataset: 用 eve 上训练的模型（expNN）在 mpiifacegaze 上测试
# 训练先完成: python main.py --dataset zhang2015-insightface/eve --method resnet50
# 用法: REUSE_EXP=expNN bash scripts/zhang2015-insightface/resnet50/cross-dataset/eve_mpiifacegaze.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

exp=$(require_exp zhang2015-insightface/eve resnet50)
"$PYTHON" main.py --dataset zhang2015-insightface/mpiifacegaze --method resnet50 --test --exp "$exp" --set dataset.split.mode=all_subjects
