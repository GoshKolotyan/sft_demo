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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/aua"
DEV_PATH="data/ambigqa.dev_4h.clarify.jsonl"
N_SAMPLES=10

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EXP="${EXPERIMENT_NAMES[$i]}"

    echo "=== Training: $EXP ==="
    make sft_train_start MODEL="$MODEL" EXPERIMENT_NAME="$EXP" MODE=gen_clarify_q

    echo "=== Inference: $EXP ==="
    make sft_inference MODEL="$MODEL" EXPERIMENT_NAME="$EXP" MODE=gen_clarify_q INFERENCE_MODE=clarify_q

    # inference.py saves output as: {checkpoint}/{dev_stem}.clarify_q_s{N_SAMPLES}.jsonl
    CHECKPOINT_DIR="$OUTPUT_DIR/$MODEL/gen_clarify_q/$EXP/best_checkpoint"
    INFERENCE_OUTPUT="$CHECKPOINT_DIR/ambigqa.dev_4h.clarify.clarify_q_s${N_SAMPLES}.jsonl"

    # echo "=== Metrics: $EXP ==="
    # make sft_metrics INPUT_PATH="$INFERENCE_OUTPUT" METRICS_MODE=clarify_q

    echo "Done: $EXP"
done

echo "All experiments complete!"