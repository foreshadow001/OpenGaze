#!/usr/bin/env bash
# resnet18 / cross-dataset: 用 gazecapture 上训练的模型（expNN）在 eve 上测试
# 训练先完成: python main.py --dataset gazecapture --method resnet18
# 用法: REUSE_EXP=expNN bash scripts/resnet18/cross-dataset/gazecapture_eve.sh
set -euo pipefail
source "$(dirname "$0")/../../common.sh"

exp=$(require_exp gazecapture resnet18)
"$PYTHON" main.py --dataset eve --method resnet18 --test --exp "$exp"
