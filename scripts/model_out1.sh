#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./merge_and_export.sh [hf_repo]
# 例子:
#   ./merge_and_export.sh
#   ./merge_and_export.sh username/my-model
#
# 可选环境变量:
#   FORCE=1                 # 目标存在时强制覆盖（默认 0 = 跳过）
#   EXPORT_BASE=/path/dir   # 导出根目录（默认 /g0001sr/lzy/model/test_our_models）

# --- 放在脚本开头的函数区 ---
copy_tree() {
  # 用法：copy_tree <src_dir> <dst_dir>
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"

  if command -v rsync >/dev/null 2>&1; then
    # rsync：快且鲁棒
    rsync -a --info=progress2 "$src"/ "$dst"/
  elif cp --help 2>&1 | grep -q '\-a'; then
    # cp -a：保留权限/时间戳；注意用 . 以包含隐藏文件
    cp -a "$src"/. "$dst"/
  else
    # 兜底：用 Python shutil
    python3 - "$src" "$dst" <<'PY'
import os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for name in os.listdir(src):
    s = os.path.join(src, name)
    d = os.path.join(dst, name)
    if os.path.isdir(s) and not os.path.islink(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
PY
  fi
}

HF_REPO=${1:-}
FORCE=${FORCE:-1}
EXPORT_BASE=${EXPORT_BASE:-/g0001sr/zgz/model/none1_models}

LOCAL_DIRS=(
  # "/g0001sr/zgz/EasyR1/checknovelty/global_step_10/actor"
  "/g0001sr/zgz/EasyR1/checknone/global_step_10/actor"
  # "/g0001sr/zgz/EasyR1/checkpointsnone/global_step_20/actor

)

mkdir -p "$EXPORT_BASE"

for LOCAL_DIR in "${LOCAL_DIRS[@]}"; do
  if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "跳过：目录不存在 -> $LOCAL_DIR"
    continue
  fi

  step_name="$(basename "$(dirname "$LOCAL_DIR")")"   # e.g. global_step_80
  step_num="${step_name##*_}"                         # 80
  exp_name="$(basename "$(dirname "$(dirname "$LOCAL_DIR")")")"  # e.g. 1015Ours_K0_vqa

  echo "==> 合并 $step_name ：$LOCAL_DIR"

  if [[ -n "$HF_REPO" ]]; then
    python3 model_merger.py \
      --local_dir "$LOCAL_DIR" \
      --hf_upload_path "${HF_REPO}:${step_name}"
  else
    python3 model_merger.py --local_dir "$LOCAL_DIR"
  fi

  # ---- 导出 huggingface 文件 ----
  src=""
  if [[ -d "${LOCAL_DIR}/huggingface" ]]; then
    src="${LOCAL_DIR}/huggingface"
  elif [[ -d "${LOCAL_DIR}/hugginface" ]]; then  # 兼容拼写
    src="${LOCAL_DIR}/hugginface"
  else
    echo "✗ 未找到 huggingface 目录：${LOCAL_DIR}/{huggingface|hugginface}，跳过导出 step ${step_num}"
    continue
  fi

  dst="${EXPORT_BASE}/${exp_name}_step_${step_num}"

  if [[ -e "$dst" && "$FORCE" != "1" ]]; then
    echo "⚠️  目标已存在，跳过导出（设定 FORCE=1 可覆盖）：$dst"
  else
    [[ -e "$dst" ]] && { echo "🧹 删除已存在的目标：$dst"; rm -rf "$dst"; }
    echo "➡️  导出：$src  →  $dst"
    copy_tree "$src" "$dst"
    echo "✅ 导出完成：$dst"
  fi
done

echo "全部完成。导出目录：$EXPORT_BASE"
