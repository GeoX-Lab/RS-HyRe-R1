#!/usr/bin/env bash
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1


LOG_DIR="/g0001sr/zgz/EasyR1/job_logs/$(date +%Y%m%d_%H%M%S)_novelty"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/full.log") 2>&1

echo "[$(date)] 🚀 JOB STARTED | Log dir: $LOG_DIR"
echo ">>> Host: $(hostname)"
echo ">>> GPUs available: $(nvidia-smi -L | wc -l) cards"

# ========== Step 1: embedding ==========
(
  echo "[$(date)] ➤ Starting embedding service in background..."
  cd /g0001sr/zgz/embedding
  source venv/bin/activate
  
  nohup bash embedding.sh > "$LOG_DIR/embedding.log" 2>&1 & EMBED_PID=$!
  
  # Waiting
  echo "Waiting for embedding service to start (PID=$EMBED_PID)..."
  sleep 180
)

# ========== Step 2: Main training ==========
RAY_BIN="/g0001sr/zgz/EasyR1_main/venv/bin/ray"
echo "[$(date)] ➤ Cleaning up old Ray..."
$RAY_BIN stop --force || true

cd /g0001sr/zgz/EasyR1
source /g0001sr/zgz/EasyR1_main/venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export VLLM_USE_V1=1 && ray start --head

MODEL_PATH=/g0001sr/zgz/model_weight/Qwen2.5_VL_3B
VRSB_JSONL_DIR=/g0001sr/zgz/mutil_task
FORMAT_PROMPT=/g0001sr/zgz/EasyR1/examples/format_prompt/multi-task.jinja
REWARD_PY=/g0001sr/zgz/EasyR1/examples/reward_function/HyRe_score.py
EXP_NAME="HyRe-R1"

echo "[$(date)] ➤ Launching training (4 GPUs)..."
python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=${VRSB_JSONL_DIR}/train.jsonl \
  data.val_files=${VRSB_JSONL_DIR}/val.jsonl \
  data.image_dir=null \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  data.format_prompt=${FORMAT_PROMPT} \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.rollout.tensor_parallel_size=1 \
  worker.reward.reward_type=batch \
  worker.reward.reward_function=${REWARD_PY}:compute_score \
  trainer.experiment_name=${EXP_NAME} \
  trainer.n_gpus_per_node=4 \
  worker.actor.global_batch_size=64 \
  worker.actor.micro_batch_size_per_device_for_experience=8 \
  data.val_batch_size=128 \
  data.rollout_batch_size=128 \
  trainer.total_epochs=40 \
  trainer.save_limit=-1 \
  trainer.val_freq=-1 \
  trainer.save_freq=10 \
  trainer.val_before_train=false

echo "[$(date)] ✅ JOB COMPLETED SUCCESSFULLY!"