#!/usr/bin/env bash
# resnet50 / within-dataset: 在 eve 上训练并测试（每次运行新开一个实验）
# 断点续训请直接用: python main.py --resume expNN
# 用法: bash scripts/zhang2015-specific-face-model/resnet50/within-dataset/eve.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

py main.py --dataset zhang2015-specific-face-model/eve --method resnet50
exp=$(latest_exp)
"$PYTHON" main.py --dataset zhang2015-specific-face-model/eve --method resnet50 --test --exp "$exp"
