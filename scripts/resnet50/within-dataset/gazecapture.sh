#!/usr/bin/env bash
# resnet50 / within-dataset: 在 gazecapture 上训练并测试（每次运行新开一个实验）
# 断点续训请直接用: python main.py --resume expNN
# 用法: bash scripts/resnet50/within-dataset/gazecapture.sh
set -euo pipefail
source "$(dirname "$0")/../../common.sh"

"$PYTHON" main.py --dataset gazecapture --method resnet50
exp=$(latest_exp)
"$PYTHON" main.py --dataset gazecapture --method resnet50 --test --exp "$exp"
