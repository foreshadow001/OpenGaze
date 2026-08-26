#!/usr/bin/env bash
# 运行全部 cross-dataset 评测组合（n(n-1) 个）。
# 每个源数据集需用 <大写名>_EXP=expNN 指定其训练实验，未指定的源自动跳过全部组合。
# 训练先完成，如: python main.py --dataset zhang2015-insightface/xgaze --method resnet50
# 用法: XGAZE_EXP=exp00 MPIIFACEGAZE_EXP=exp03 bash scripts/zhang2015-insightface/resnet50/cross-dataset/all.sh
set -uo pipefail
source "$(dirname "$0")/../../../common.sh"

declare -A SRC_EXP
for src in xgaze mpiifacegaze gazecapture eve; do
    var="${src^^}_EXP"
    if [ -n "${!var:-}" ]; then
        SRC_EXP[$src]="${!var}"
    else
        echo "跳过源 $src（未指定 ${var}=expNN）"
    fi
done

dir="$(dirname "$0")"
pass=0; fail=0; skip=0; failed=(); skipped=()
for script in "$dir"/*.sh; do
    name="$(basename "$script")"
    [ "$name" = "all.sh" ] && continue
    src="${name%%_*}"
    if [ -z "${SRC_EXP[$src]:-}" ]; then
        skip=$((skip + 1)); skipped+=("$name")
        continue
    fi
    echo "===== $name (REUSE_EXP=${SRC_EXP[$src]}) ====="
    if REUSE_EXP="${SRC_EXP[$src]}" bash "$script"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); failed+=("$name")
    fi
done
echo "===== 汇总: 成功 $pass / 失败 $fail / 跳过 $skip ====="
((${#failed[@]})) && printf '失败: %s\n' "${failed[@]}"
((${#skipped[@]})) && printf '跳过: %s\n' "${skipped[@]}"
exit "$fail"
