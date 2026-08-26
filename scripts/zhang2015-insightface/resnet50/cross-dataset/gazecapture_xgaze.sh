#!/usr/bin/env bash
# resnet50 / cross-dataset: 用 gazecapture 上训练的模型（expNN）在 xgaze 上测试
# 训练先完成: python main.py --dataset zhang2015-insightface/gazecapture --method resnet50
# 用法: REUSE_EXP=expNN bash scripts/zhang2015-insightface/resnet50/cross-dataset/gazecapture_xgaze.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

exp=$(require_exp zhang2015-insightface/gazecapture resnet50)
"$PYTHON" main.py --dataset zhang2015-insightface/xgaze --method resnet50 --test --exp "$exp"
