#!/usr/bin/env bash
# resnet18 / within-dataset: 在 gazecapture 上训练并测试（每次运行新开一个实验）
# 断点续训请直接用: python main.py --resume expNN
# 用法: bash scripts/resnet18/within-dataset/gazecapture.sh
set -euo pipefail
source "$(dirname "$0")/../../common.sh"

"$PYTHON" main.py --dataset gazecapture --method resnet18
exp=$(latest_exp)
"$PYTHON" main.py --dataset gazecapture --method resnet18 --test --exp "$exp"
