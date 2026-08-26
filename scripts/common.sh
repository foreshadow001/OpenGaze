# scripts/ 公共函数：python 路径、repo 根目录、实验指定与校验、多卡启动器
# 各脚本 source 本文件后自动 cd 到 repo 根目录，可从任意位置调用
#
# 脚本职责：
#   within-dataset/*.sh  每次新开实验：训练 + 测试一条龙，无需任何参数
#   cross-dataset/*.sh   只做评测：训练须先完成，REUSE_EXP=expNN 指定用哪次实验
# 断点续训不属于脚本职责，直接用 main.py --resume expNN
#
# 多卡：训练类命令用 py 启动（多卡时经 torchrun，DDP）；测试始终单卡。
# 用哪些卡由 configs/common.yaml 的 gpus 列表指定，运行时可用环境变量覆盖：
#   GPUS=0,1 bash xxx.sh    # 只用 0、1 两卡（DDP）
#   GPUS=0 bash xxx.sh      # 单卡（不起 torchrun，直接 python）
#   GPUS=2,3 bash xxx.sh    # 用 2、3 两卡
# 单卡与多卡训练的 checkpoint 互通（opengaze-ckpt v1），可互相 resume/test。

PYTHON="${PYTHON:-/ssd/conda/envs/yanglinxuan/opengaze/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 训练用卡列表：configs/common.yaml 的 gpus（读不到回退 0,1,2,3）；
# 对 source 本文件的所有命令生效，含单卡测试
_gpus_from_cfg() {
    sed -nE 's/^gpus:[[:space:]]*\[(.*)\].*/\1/p' configs/common.yaml 2>/dev/null \
        | head -1 | tr -d ' '
}
GPUS="${GPUS:-$(_gpus_from_cfg)}"
[ -n "$GPUS" ] || GPUS="0,1,2,3"
export CUDA_VISIBLE_DEVICES="$GPUS"

# 训练启动器：py main.py <args...> —— 列表 1 张卡 == "$PYTHON" main.py <args...>，
#            多张卡 == CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=<张数> main.py <args...>
py() {
    local n
    n=$(echo "$GPUS" | awk -F, '{print NF}')
    if [ "$n" -gt 1 ]; then
        "$PYTHON" -m torch.distributed.run --nproc_per_node="$n" "$@"
    else
        "$PYTHON" "$@"
    fi
}

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
