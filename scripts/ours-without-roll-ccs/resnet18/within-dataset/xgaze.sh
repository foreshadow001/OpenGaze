#!/usr/bin/env bash
# resnet18 / ours-without-roll-ccs / within-dataset: 在 xgaze 上训练并测试（标签 = ccs）
# v3 roll-only 归一化（头姿保留在 patch），标签 face_gaze（v3 归一化相机系，roll 修正、头姿保留）
# 断点续训请直接用: python main.py --resume expNN
# 用法: bash scripts/ours-without-roll-ccs/resnet18/within-dataset/xgaze.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

py main.py --dataset ours-without-roll/xgaze --method resnet18 --label ccs
exp=$(latest_exp)
"$PYTHON" main.py --dataset ours-without-roll/xgaze --method resnet18 --test --exp "$exp"
