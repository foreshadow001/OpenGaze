#!/usr/bin/env bash
# resnet50 / within-dataset: MPIIFaceGaze leave-one-out（15 折，折 i 以 p{i:02d} 为测试被试）
# 全部折与最后的全量训练放在同一个实验目录下的子运行中（fold_00~fold_14、all），
# 跑完 15 折后自动追加一次 15 人全量训练（split.mode=all_subjects，供 cross-dataset 评测）。
# 脚本可安全重跑：已完成的折（test_result 存在）自动跳过，中断的折自动从最新 ckpt 续训，
# 手动续训某折: python main.py --resume expNN --run fold_XX
# 用法: bash scripts/zhang2015-specific-face-model/resnet50/within-dataset/mpiifacegaze.sh
set -euo pipefail
source "$(dirname "$0")/../../../common.sh"

# 复用未完成的本「数据集 × 方法」LOO 实验（取最新），全部完成或不存在则新开
exp=""
for d in $(ls exp/ 2>/dev/null | grep -E '^exp[0-9]+$' | sort -rV); do
    if grep -q "^dataset_config: zhang2015-specific-face-model/mpiifacegaze$" "exp/$d/config.yaml" 2>/dev/null \
       && grep -q "^method_config: resnet50$" "exp/$d/config.yaml" 2>/dev/null; then
        exp=$d
        echo "续跑已有实验 $exp（未完成的折自动续训/补跑）"
        break
    fi
done

# run_one <run_name> <extra_set...>：完成一个子运行（新训/续训 + 幂等）
run_one() {
    local run=$1; shift
    # 已完成（有测试结果；all 以 ckpt 非空近似）→ 跳过
    if [ -n "$exp" ] && [ -f "exp/$exp/$run/test_result_mpiifacegaze.json" ]; then
        echo "跳过已完成的 $run"; return 0
    fi
    if [ -z "$exp" ]; then
        py main.py --dataset zhang2015-specific-face-model/mpiifacegaze --method resnet50 --run "$run" "$@"
        exp=$(latest_exp)
    elif [ -d "exp/$exp/$run" ]; then
        if ls "exp/$exp/$run/ckpt/"*.pth >/dev/null 2>&1; then
            py main.py --resume "$exp" --run "$run"          # 中断续训
        else
            rm -rf "exp/$exp/$run"                                   # 空壳重建
            py main.py --dataset zhang2015-specific-face-model/mpiifacegaze --method resnet50 --exp "$exp" --run "$run" "$@"
        fi
    else
        py main.py --dataset zhang2015-specific-face-model/mpiifacegaze --method resnet50 --exp "$exp" --run "$run" "$@"
    fi
}

for fold in $(seq 0 14); do
    echo "===== leave-one-out fold $fold（测试被试 p$(printf '%02d' "$fold")） ====="
    run_one "$(printf 'fold_%02d' "$fold")" --set dataset.split.fold=$fold
    "$PYTHON" main.py --dataset zhang2015-specific-face-model/mpiifacegaze --method resnet50 --test --exp "$exp" --run "$(printf 'fold_%02d' "$fold")"
done

echo "===== 15 人全量训练（split.mode=all_subjects，用于 cross-dataset） ====="
run_one all --set dataset.split.mode=all_subjects

echo "===== leave-one-out 汇总 ====="
"$PYTHON" - "$exp" <<'PYEOF'
import glob, json, statistics, sys
exp = sys.argv[1]
files = sorted(glob.glob(f'exp/{exp}/fold_*/test_result_mpiifacegaze.json'))
errs = []
for fp in files:
    r = json.load(open(fp))
    errs.append(r['gaze_error_deg'])
    print(f"  {fp.split('/')[-2]}: {r['gaze_error_deg']:.2f} deg ({r['num_samples']} samples)")
if len(errs) < 15:
    print(f'警告: 仅完成 {len(errs)}/15 折')
print(f'LOO mean: {statistics.mean(errs):.2f} +/- {statistics.stdev(errs):.2f} deg')
PYEOF
