#!/usr/bin/env bash
# resnet18 / cross-dataset: 用 gazecapture 上训练的模型（expNN）在 mpiifacegaze 上测试
# 训练先完成: python main.py --dataset zhang2015-insightface/gazecapture --method resnet18
# 用法: REUSE_EXP=expNN bash scripts/zhang2015-insightface/resnet18/cross-dataset/gazecapture_mpiifacegaze.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

exp=$(require_exp zhang2015-insightface/gazecapture resnet18)
"$PYTHON" main.py --dataset zhang2015-insightface/mpiifacegaze --method resnet18 --test --exp "$exp" --set dataset.split.mode=all_subjects
