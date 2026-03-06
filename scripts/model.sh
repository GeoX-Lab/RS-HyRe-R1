#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./export_hf_slices.sh [--force]
#
# 说明：
# - 固定从这四个路径复制：
#   /g0001sr/lzy/EasyR1/checkpoints/easy_r1/1015Ours_K0_vqa/global_step_{20,40,60,80}/actor/{huggingface|hugginface}
# - 复制到：
#   /g0001sr/lzy/model/test_our_models/1015Ours_K0_vqa_step_{20,40,60,80}
# - 默认若目标已存在则跳过；加 --force 则先删后拷。

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

EXP_NAME="1015Ours_K0_vqa"
BASE_IN="/g0001sr/lzy/EasyR1/checkpoints/easy_r1/${EXP_NAME}"
BASE_OUT="/g0001sr/lzy/model/test_our_models"
STEPS=(20 40 60 80)

mkdir -p "$BASE_OUT"

for step in "${STEPS[@]}"; do
  step_dir="${BASE_IN}/global_step_${step}/actor"
  src=""
  if [[ -d "${step_dir}/huggingface" ]]; then
    src="${step_dir}/huggingface"
  elif [[ -d "${step_dir}/hugginface" ]]; then   # 兼容可能的拼写
    src="${step_dir}/hugginface"
  else
    echo "✗ 未找到 huggingface 目录：${step_dir}/{huggingface|hugginface}，跳过 step ${step}"
    continue
  fi

  dst="${BASE_OUT}/${EXP_NAME}_step_${step}"

  if [[ -e "$dst" && $FORCE -eq 0 ]]; then
    echo "⚠️  目标已存在，跳过（使用 --force 覆盖）：$dst"
    continue
  fi

  if [[ -e "$dst" && $FORCE -eq 1 ]]; then
    echo "🧹 删除已存在的目标：$dst"
    rm -rf "$dst"
  fi

  echo "➡️  复制：$src  →  $dst"
  # 使用 rsync 保留权限/时间戳，显示进度
  rsync -a --info=progress2 "$src"/ "$dst"/

  echo "✅ 完成 step ${step}"
done

echo "🎉 全部处理完成。输出目录：$BASE_OUT"
