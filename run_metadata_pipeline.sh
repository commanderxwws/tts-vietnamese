#!/usr/bin/env bash
# Vietnamese pipeline (steps 1-4): CSV -> manifest -> tokenizer -> multiprocess preprocessing -> pairing
# Usage: ./run_metadata_pipeline.sh /path/to/metadata.csv /path/to/audio_dir [work_root]

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <metadata.csv> <audio_dir> [work_root]" >&2
  exit 1
fi

METADATA_CSV=$(realpath "$1")
AUDIO_DIR=$(realpath "$2")
WORK_ROOT=${3:-runs/vietnamese_from_metadata}

LANGUAGE_CODE=${LANGUAGE_CODE:-vi}
VAL_RATIO=${VAL_RATIO:-0.001}

# Optional knobs via environment variables
METADATA_EXTRA_ARGS=${METADATA_EXTRA_ARGS:-}
TOKENIZER_OUTPUT_PREFIX=${TOKENIZER_OUTPUT_PREFIX:-"$WORK_ROOT/tokenizer/${LANGUAGE_CODE}_bpe"}
VOCAB_SIZE=${VOCAB_SIZE:-12000}
TRAIN_TOKENIZER=${TRAIN_TOKENIZER:-1}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
PREPROCESS_EXTRA_ARGS=${PREPROCESS_EXTRA_ARGS:-""}
PAIRS_PER_TARGET=${PAIRS_PER_TARGET:-2}
CONFIG_PATH=${CONFIG_PATH:-checkpoints/config.yaml}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-checkpoints/gpt_old.pth}
MULTIPROC_EXTRA_ARGS=${MULTIPROC_EXTRA_ARGS:-""}
NUM_PROCESSES=${NUM_PROCESSES:-4}
WORKER_BATCH_SIZE=${WORKER_BATCH_SIZE:-4}
WORKER_THREADS=${WORKER_THREADS:-4}
MULTIPROC_DEVICE=${MULTIPROC_DEVICE:-cuda}
HF_CACHE_DIR=${HF_CACHE_DIR:-"$WORK_ROOT/hf_cache"}
PYTHON_BIN=${PYTHON_BIN:-python3}

MANIFEST_DIR="$WORK_ROOT/manifests"
MANIFEST_PATH="$MANIFEST_DIR/${LANGUAGE_CODE}_metadata.jsonl"
PROCESSED_DIR="$WORK_ROOT/processed_data"

mkdir -p "$MANIFEST_DIR" "$WORK_ROOT/tokenizer" "$PROCESSED_DIR" "$HF_CACHE_DIR"

echo "[Step 1] Converting metadata CSV -> JSONL manifest"
$PYTHON_BIN tools/metadata_to_manifest.py \
  --metadata "$METADATA_CSV" \
  --audio-root "$AUDIO_DIR" \
  --output "$MANIFEST_PATH" \
  --default-language "$LANGUAGE_CODE" \
  --delimiter '|' \
  --quotechar '"' \
  --id-column audio_path \
  --text-column text \
  --audio-column audio_path \
  --speaker-column speaker_id \
  --store-relative \
  ${METADATA_EXTRA_ARGS}

if [[ "$TRAIN_TOKENIZER" == "1" ]]; then
  echo "[Step 2] Training SentencePiece tokenizer from manifest"
  $PYTHON_BIN tools/tokenizer/train_bpe.py \
    --manifest "$MANIFEST_PATH" \
    --output-prefix "$TOKENIZER_OUTPUT_PREFIX" \
    --vocab-size "$VOCAB_SIZE" \
    --model-type bpe \
    --byte-fallback
  TOKENIZER_MODEL="${TOKENIZER_OUTPUT_PREFIX}.model"
else
  if [[ -z "$TOKENIZER_PATH" ]]; then
    echo "[Error] TOKENIZER_PATH must be set when TRAIN_TOKENIZER=0" >&2
    exit 1
  fi
  TOKENIZER_MODEL="$TOKENIZER_PATH"
fi

PREPROCESS_EXTRA_ARGS_ARRAY=()
if [[ -n "$PREPROCESS_EXTRA_ARGS" ]]; then
  read -r -a PREPROCESS_EXTRA_ARGS_ARRAY <<<"$PREPROCESS_EXTRA_ARGS"
fi

MULTIPROC_EXTRA_ARGS_ARRAY=()
if [[ -n "$MULTIPROC_EXTRA_ARGS" ]]; then
  read -r -a MULTIPROC_EXTRA_ARGS_ARRAY <<<"$MULTIPROC_EXTRA_ARGS"
fi

echo "[Step 3] Multiprocess preprocessing (val ratio $VAL_RATIO)"
PREPROCESS_CMD=(
  "$PYTHON_BIN" tools/preprocess_multiproc.py
  --manifest "$MANIFEST_PATH"
  --output-dir "$PROCESSED_DIR"
  --tokenizer "$TOKENIZER_MODEL"
  --config "$CONFIG_PATH"
  --gpt-checkpoint "$BASE_CHECKPOINT"
  --language "$LANGUAGE_CODE"
  --val-ratio "$VAL_RATIO"
  --device "$MULTIPROC_DEVICE"
  --batch-size "$WORKER_BATCH_SIZE"
  --workers "$WORKER_THREADS"
  --num-processes "$NUM_PROCESSES"
  --skip-existing
  --hf-cache-dir "$HF_CACHE_DIR"
  --audio-root "$AUDIO_DIR"
)

if [[ "${#PREPROCESS_EXTRA_ARGS_ARRAY[@]}" -gt 0 ]]; then
  PREPROCESS_CMD+=("${PREPROCESS_EXTRA_ARGS_ARRAY[@]}")
fi

if [[ "${#MULTIPROC_EXTRA_ARGS_ARRAY[@]}" -gt 0 ]]; then
  PREPROCESS_CMD+=(--extra-args)
  PREPROCESS_CMD+=("${MULTIPROC_EXTRA_ARGS_ARRAY[@]}")
fi

"${PREPROCESS_CMD[@]}"

echo "[Step 4] Building GPT prompt/target pairs"
$PYTHON_BIN tools/generate_gpt_pairs.py \
  --dataset "$PROCESSED_DIR" \
  --pairs-per-target "$PAIRS_PER_TARGET" \
  --force

echo "[Done] Steps 1-4 finished. You can now fine-tune using trainers/train_gpt_v2.py if desired."
