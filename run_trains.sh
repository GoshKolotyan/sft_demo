#!/bin/bash
set -e

MODELS=(
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.2-1B"
    "google/gemma-2-2b"
    "Qwen/Qwen2.5-3B"
)

EXPERIMENT_NAMES=(
    # "Llama-3.2-3B"
    # "Llama-3.2-1B"
    "gemma-2-2b"
    "Qwen2.5-3B"
)

# Per-model LoRA and training configs (r, alpha, dropout, batch_size, grad_accum, 8bit)
declare -A LORA_CONFIGS
LORA_CONFIGS["google/gemma-2-2b"]="4 8 0.05 2 4 1"
LORA_CONFIGS["Qwen/Qwen2.5-3B"]="8 16 0.05 2 4 1"
# LORA_CONFIGS["meta-llama/Llama-3.2-3B"]="16 32 0.05 8 1 0"
# LORA_CONFIGS["meta-llama/Llama-3.2-1B"]="16 32 0.05 8 1 0"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/aua"
DEV_PATH="data/ambigqa.dev_4h.clarify.jsonl"
N_SAMPLES=10

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EXP="${EXPERIMENT_NAMES[$i]}"

    # Parse per-model config
    read -r LORA_R LORA_ALPHA LORA_DROPOUT BATCH_SIZE GRAD_ACCUM USE_8BIT <<< "${LORA_CONFIGS[$MODEL]}"

    LOAD_8BIT=""
    if [ "$USE_8BIT" = "1" ]; then
        LOAD_8BIT="LOAD_IN_8BIT=1"
    fi

    echo "=== Training: $EXP (r=$LORA_R, alpha=$LORA_ALPHA, bs=$BATCH_SIZE, grad_accum=$GRAD_ACCUM, 8bit=$USE_8BIT) ==="
    make sft_train_start MODEL="$MODEL" EXPERIMENT_NAME="$EXP" MODE=gen_clarify_q \
        LORA_R="$LORA_R" LORA_ALPHA="$LORA_ALPHA" LORA_DROPOUT="$LORA_DROPOUT" \
        BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM_STEPS="$GRAD_ACCUM" $LOAD_8BIT

    echo "=== Inference: $EXP ==="
    make sft_inference MODEL="$MODEL" EXPERIMENT_NAME="$EXP" MODE=gen_clarify_q INFERENCE_MODE=clarify_q

    # inference.py saves output as: {checkpoint}/{dev_stem}.clarify_q_s{N_SAMPLES}.jsonl
    CHECKPOINT_DIR="$OUTPUT_DIR/$MODEL/gen_clarify_q/$EXP/best_checkpoint"
    INFERENCE_OUTPUT="$CHECKPOINT_DIR/ambigqa.dev_4h.clarify.clarify_q_s${N_SAMPLES}.jsonl"

    echo "=== Metrics: $EXP ==="
    make sft_metrics INPUT_PATH="$INFERENCE_OUTPUT" METRICS_MODE=clarify_q

    echo "Done: $EXP"
done

echo "All experiments complete!"