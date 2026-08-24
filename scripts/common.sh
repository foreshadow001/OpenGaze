# scripts/ 公共函数：python 路径、repo 根目录、实验指定与校验
# 各脚本 source 本文件后自动 cd 到 repo 根目录，可从任意位置调用
#
# 脚本职责：
#   within-dataset/*.sh  每次新开实验：训练 + 测试一条龙，无需任何参数
#   cross-dataset/*.sh   只做评测：训练须先完成，REUSE_EXP=expNN 指定用哪次实验
# 断点续训不属于脚本职责，直接用 main.py --resume expNN

PYTHON="${PYTHON:-/home/hitsz/anaconda3/envs/opengaze/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# exp/ 下编号最大的实验（expNN）
latest_exp() {
    ls exp/ 2>/dev/null | grep -E '^exp[0-9]+$' | sort -V | tail -n1
}

# require_exp <dataset> <method>
# 校验并输出 REUSE_EXP 指定的实验编号（expNN）。
# 必须 REUSE_EXP=expNN 显式指定；不指定、实验不存在、或快照与脚本的
# 「数据集 × 方法」不符时直接报错（不猜测、不自动选择）。
require_exp() {
    local dataset=$1 method=$2
    if [ -z "${REUSE_EXP:-}" ]; then
        echo "错误: 未指定实验。用法: REUSE_EXP=expNN bash $0" >&2
        echo "      训练实验需先用 main.py 完成，如: python main.py --dataset $dataset --method $method" >&2
        return 1
    fi
    if [ ! -f "exp/$REUSE_EXP/config.yaml" ]; then
        echo "错误: REUSE_EXP=$REUSE_EXP 不存在（exp/ 下无该目录或快照）" >&2
        return 1
    fi
    if ! grep -q "^dataset_config: $dataset$" "exp/$REUSE_EXP/config.yaml" \
       || ! grep -q "^method_config: $method$" "exp/$REUSE_EXP/config.yaml"; then
        echo "错误: REUSE_EXP=$REUSE_EXP 不是 $dataset × $method 的训练实验（快照不匹配）" >&2
        return 1
    fi
    echo "使用实验 $REUSE_EXP（$dataset × $method）" >&2
    echo "$REUSE_EXP"
}
