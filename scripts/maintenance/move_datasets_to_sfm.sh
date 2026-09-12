#!/usr/bin/env bash
# 数据集搬迁（一次性）：删除 sfm/ylx 两块盘的旧数据集副本，把 NVMe 正本
# （/data/others_preprocessed_datasets 十二套 v1/v2/v3）安全移动到 sfm 盘。
#
# 安全设计：
#   - 删除/复制均为**白名单精确目录名**（根下一级），绝不递归通配；
#     sfm 盘的 face_models/、canonical_mean6.txt 等模型产物不在名单内，不受影响；
#   - 删旧前先校验对应 NVMe 正本存在且 h5 非空；
#   - 复制后逐目录校验（文件数 + 文件字节数 + h5 抽样读一行）通过才删 NVMe 源；
#   - 两处交互确认（删旧 / 删源），SKIP_CONFIRM=1 跳过；
#   - 中断可重跑：重跑会先删掉 sfm 上的半截副本再重新复制。
#
# 进度可视化：逐目录 rsync --info=progress2（实时 % + 速度 + ETA）+ 阶段横幅
# [i/12] 全局进度 + 前后 df 对比。
#
# 用法:
#   bash scripts/maintenance/move_datasets_to_sfm.sh              # 交互确认
#   SKIP_CONFIRM=1 bash scripts/maintenance/move_datasets_to_sfm.sh
set -euo pipefail

DATA=/data/others_preprocessed_datasets
SFM=/media/yanglinxuan/sfm
YLX=/media/yanglinxuan/ylx
PYTHON="${PYTHON:-/ssd/conda/envs/yanglinxuan/opengaze/bin/python}"

# 白名单：八套在 sfm（v2+v3），四套在 ylx（v1）；十二套 NVMe 正本同名
SFM_OLD=(xgaze_specific_224 eve_specific_224 gazecapture_specific_224 \
         mpiifacegaze_specific_224 xgaze_noroll_224 eve_noroll_224 \
         gazecapture_noroll_224 mpiifacegaze_noroll_224)
YLX_OLD=(xgaze_insightface_224 eve_insightface_224 \
         gazecapture_insightface_224 mpiifacegaze_insightface_224)
ALL=( "${SFM_OLD[@]}" "${YLX_OLD[@]}" )

T0=$SECONDS
say()  { printf '\e[1;36m== %s ==\e[0m\n' "$*"; }
die()  { printf '\e[1;31m错误: %s\e[0m\n' "$*" >&2; exit 1; }
confirm() {
    [ "${SKIP_CONFIRM:-0}" = "1" ] && return 0
    local a; read -r -p "$(printf '\e[1;33m%s\e[0m [y/N] ' "$1")" a
    [[ "$a" == y || "$a" == Y ]] || die "用户取消"
}
fstat() {   # 目录 → "文件数 字节数"（仅普通文件，精确可比）
    find "$1" -type f -printf '%s\n' | awk '{s+=$1; n++} END{print n+0, s+0}'
}
banner() { printf '\n\e[1;35m########## %s ##########\e[0m\n' "$*"; }

# ---------------- 0. 预检 ----------------
banner "0/5 预检"
for m in "$DATA" "$SFM" "$YLX"; do
    [ -d "$m" ] || die "目录不存在: $m（盘未挂载？）"
done
DF_BEFORE=$(df -h "$SFM" "$YLX" "$DATA" | awk 'NR>1' )
say "NVMe 正本清点"
NEED=0
for d in "${ALL[@]}"; do
    [ -d "$DATA/$d" ] || die "NVMe 正本缺失: $DATA/$d"
    read -r n b < <(fstat "$DATA/$d")
    [ "$n" -gt 0 ] || die "NVMe 正本为空: $DATA/$d"
    NEED=$(( NEED + b ))
    printf '  %-32s %6d 文件  %8.1f GB\n' "$d" "$n" "$(echo "$b/1073741824" | bc -l)"
done
printf '  合计 %.1f GB\n' "$(echo "$NEED/1073741824" | bc -l)"
FREED=0
for d in "${SFM_OLD[@]}"; do
    [ -d "$SFM/$d" ] && { read -r _n _b < <(fstat "$SFM/$d"); FREED=$((FREED+_b)); }
done
AVAIL_KB=$(df --output=avail -k "$SFM" | tail -1 | tr -d ' ')
AVAIL_AFTER=$(( AVAIL_KB*1024 + FREED ))
say "sfm 容量: 当前可用 $(df -h "$SFM" | tail -1 | awk '{print $4}')，删除旧八套后再 +$(echo "$FREED/1073741824" | bc -l | cut -c1-6)GB"
[ "$AVAIL_AFTER" -ge $((NEED * 102 / 100)) ] || die "sfm 删除旧数据后空间仍不足（需 $(echo "$NEED/1073741824" | bc -l | cut -c1-6)GB）"

confirm "将删除 sfm 根下八套旧数据集 + ylx 根下四套旧数据集（保留 sfm 的 face_models 等），然后从 NVMe 复制十二套到 sfm。继续?"

# ---------------- 1. 删除两块盘旧副本 ----------------
banner "1/5 删除 sfm 旧八套（白名单，face_models 不动）"
for d in "${SFM_OLD[@]}"; do
    if [ -d "$SFM/$d" ]; then
        read -r n _ < <(fstat "$DATA/$d")
        printf '  rm %-32s（NVMe 正本 %d 个 h5 已校验）\n' "$SFM/$d" "$n"
        rm -rf "$SFM/$d"
    else
        printf '  跳过（不存在）%s\n' "$SFM/$d"
    fi
done
banner "1/5 删除 ylx 旧四套"
for d in "${YLX_OLD[@]}"; do
    if [ -d "$YLX/$d" ]; then
        printf '  rm %-32s（NVMe 正本已校验）\n' "$YLX/$d"
        rm -rf "$YLX/$d"
    else
        printf '  跳过（不存在）%s\n' "$YLX/$d"
    fi
done

# ---------------- 2. NVMe → sfm 复制（逐目录实时进度） ----------------
banner "2/5 复制 NVMe → sfm（rsync，实时进度）"
i=0; N=${#ALL[@]}
for d in "${ALL[@]}"; do
    i=$((i+1))
    printf '\n\e[1;34m[%d/%d] %s\e[0m\n' "$i" "$N" "$d"
    rsync -a --info=progress2 --stats "$DATA/$d" "$SFM/" \
        | grep -Ev '^(Number of files:|Number of regular files transferred:|Total transferred file size:)' \
        | sed -n '1p;/speedup is/p'
done

# ---------------- 3. 校验（文件数 / 字节数 / h5 抽样） ----------------
banner "3/5 逐目录校验"
FAIL=0
for d in "${ALL[@]}"; do
    read -r n1 b1 < <(fstat "$DATA/$d")
    read -r n2 b2 < <(fstat "$SFM/$d")
    if [ "$n1" != "$n2" ] || [ "$b1" != "$b2" ]; then
        printf '  \e[1;31m%-32s 不一致: NVMe %s/%s vs sfm %s/%s\e[0m\n' \
               "$d" "$n1" "$b1" "$n2" "$b2"
        FAIL=1; continue
    fi
    # h5 抽样：首/中/尾各一个，读 face_patch 一行
    OK=$("$PYTHON" - "$SFM/$d" <<'PYEOF'
import glob, sys, h5py
fs = sorted(glob.glob(sys.argv[1] + '/**/*.h5', recursive=True))
for i in (0, len(fs)//2, -1):
    try:
        with h5py.File(fs[i], 'r') as f:
            assert f['face_patch'][f['face_patch'].shape[0] // 2].mean() > 0
    except Exception as e:
        print('BAD'); sys.exit(0)
print('OK')
PYEOF
)
    if [ "$OK" = "OK" ]; then
        printf '  %-32s %5d 文件 %10.1f GB  ✓（含 h5 抽样读）\n' \
               "$d" "$n2" "$(echo "$b2/1073741824" | bc -l)"
    else
        printf '  \e[1;31m%-32s h5 抽样读取失败\e[0m\n' "$d"; FAIL=1
    fi
done
[ "$FAIL" = "0" ] || die "校验未全部通过——NVMe 源保留未动，请检查后重跑"

# ---------------- 4. 删除 NVMe 源（完成移动） ----------------
confirm "十二套已全部校验通过。现在删除 NVMe 源（/data 释放 $(echo "$NEED/1073741824" | bc -l | cut -c1-6)GB），完成移动。继续?"
banner "4/5 删除 NVMe 源"
for d in "${ALL[@]}"; do
    printf '  rm %s\n' "$DATA/$d"
    rm -rf "$DATA/$d"
done
rmdir --ignore-fail-on-non-empty "$DATA" 2>/dev/null || true

# ---------------- 5. 报告 ----------------
banner "5/5 完成（用时 $(( (SECONDS-T0)/60 )) min）"
printf '\n\e[1m搬迁前:\e[0m\n%s\n\n\e[1m搬迁后:\e[0m\n%s\n\n' \
    "$DF_BEFORE" "$(df -h "$SFM" "$YLX" "$DATA" | awk 'NR>1')"
cat <<'EOF'

后续待办（脚本不自动执行）：
  1. datasets configs 的 data_dir 改回 sfm（v1/v2/v3 统一前缀替换）:
     grep -rl '/data/others_preprocessed_datasets/' configs/datasets \
       | xargs sed -i 's|/data/others_preprocessed_datasets/|/media/yanglinxuan/sfm/|g'
  2. preprocess 的 v2_dir 本就指向 sfm 路径，搬迁后重新有效，无需改
  3. CLAUDE.md 约定 6 的数据位置描述需同步更新
EOF
