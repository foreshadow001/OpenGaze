#!/usr/bin/env bash
# resnet50 / ours-without-roll-hcs / within-dataset: 在 gazecapture 上训练并测试（标签 = hcs）
# v3 roll-only 归一化（头姿保留在 patch），标签 face_gaze_hcs（头架系视线，与归一化无关）
# 断点续训请直接用: python main.py --resume expNN
# 用法: bash scripts/ours-without-roll-hcs/resnet50/within-dataset/gazecapture.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

py main.py --dataset ours-without-roll/gazecapture --method resnet50 --label hcs
exp=$(latest_exp)
"$PYTHON" main.py --dataset ours-without-roll/gazecapture --method resnet50 --test --exp "$exp"
