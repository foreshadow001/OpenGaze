#!/usr/bin/env bash
# resnet50 / cross-dataset: 用 eve 上训练的模型（expNN）在 xgaze 上测试
# 训练先完成: python main.py --dataset eve --method resnet50
# 用法: REUSE_EXP=expNN bash scripts/resnet50/cross-dataset/eve_xgaze.sh
set -euo pipefail
source "$(dirname "$0")/../../common.sh"

exp=$(require_exp eve resnet50)
"$PYTHON" main.py --dataset xgaze --method resnet50 --test --exp "$exp"
