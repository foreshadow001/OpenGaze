#!/usr/bin/env bash
# resnet18 / within-dataset: 在 xgaze 上训练并测试（每次运行新开一个实验）
# 断点续训请直接用: python main.py --resume expNN
# 用法: bash scripts/zhang2015-insightface/resnet18/within-dataset/xgaze.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

py main.py --dataset zhang2015-insightface/xgaze --method resnet18
exp=$(latest_exp)
"$PYTHON" main.py --dataset zhang2015-insightface/xgaze --method resnet18 --test --exp "$exp"
