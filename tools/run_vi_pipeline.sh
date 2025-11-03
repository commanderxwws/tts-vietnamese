#!/usr/bin/env bash
# Run steps 1–5 of the Vietnamese training pipeline:
#   1. Convert metadata CSV -> JSONL manifest
#   2. Train SentencePiece tokenizer
#   3. Copy/update Vietnamese config
#   4. Preprocess audio/text into cached features
#   5. Build prompt/target pair manifests

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
METADATA_CSV="${METADATA_CSV:-metadata.csv}"
AUDIO_ROOT="${AUDIO_ROOT:-/path/to/audio/root}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests}"
PROCESSED_DIR="${PROCESSED_DIR:-processed_vi}"
TOKENIZER_PREFIX="${TOKENIZER_PREFIX:-checkpoints/vietnamese_bpe}"
VOCAB_SIZE="${VOCAB_SIZE:-12000}"
CONFIG_SRC="${CONFIG_SRC:-checkpoints/config.yaml}"
CONFIG_DST="${CONFIG_DST:-checkpoints/config_vi.yaml}"
VAL_RATIO="${VAL_RATIO:-0.02}"
PAIRS_PER_TARGET="${PAIRS_PER_TARGET:-2}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"

if [[ ! -d "${AUDIO_ROOT}" ]]; then
    echo "AUDIO_ROOT does not exist: ${AUDIO_ROOT}" >&2
    echo "Set AUDIO_ROOT to the directory that contains your audio files." >&2
    exit 1
fi

if [[ ! -f "${METADATA_CSV}" ]]; then
    echo "metadata.csv not found at ${METADATA_CSV}" >&2
    exit 1
fi

mkdir -p "${MANIFEST_DIR}"
mkdir -p "${PROCESSED_DIR}"

RAW_MANIFEST="${MANIFEST_DIR}/vi_raw.jsonl"
TRAIN_MANIFEST="${PROCESSED_DIR}/train_manifest.jsonl"
VAL_MANIFEST="${PROCESSED_DIR}/val_manifest.jsonl"
GPT_TRAIN_MANIFEST="${PROCESSED_DIR}/gpt_pairs_train.jsonl"
GPT_VAL_MANIFEST="${PROCESSED_DIR}/gpt_pairs_val.jsonl"

echo "[1/5] Converting metadata CSV -> JSONL manifest"
"${PYTHON_BIN}" tools/prepare_vi_metadata.py \
    --csv "${METADATA_CSV}" \
    --audio-root "${AUDIO_ROOT}" \
    --output "${RAW_MANIFEST}"

echo "[2/5] Training SentencePiece tokenizer (vocab size ${VOCAB_SIZE})"
"${PYTHON_BIN}" tools/tokenizer/train_bpe.py \
    --manifest "${RAW_MANIFEST}" \
    --output-prefix "${TOKENIZER_PREFIX}" \
    --vocab-size "${VOCAB_SIZE}" \
    --model-type bpe \
    --character-coverage 0.9995 \
    --byte-fallback

echo "[3/5] Preparing Vietnamese config at ${CONFIG_DST}"
if [[ ! -f "${CONFIG_DST}" ]]; then
    cp "${CONFIG_SRC}" "${CONFIG_DST}"
fi

"${PYTHON_BIN}" - <<PY
from pathlib import Path
from omegaconf import OmegaConf

cfg_path = Path("${CONFIG_DST}")
cfg = OmegaConf.load(cfg_path)
cfg.gpt.number_text_tokens = max(int(cfg.gpt.number_text_tokens), ${VOCAB_SIZE})
OmegaConf.save(cfg, cfg_path)
PY

TOKENIZER_MODEL="${TOKENIZER_PREFIX}.model"
if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
    echo "Tokenizer model not found: ${TOKENIZER_MODEL}" >&2
    exit 1
fi

if [[ -z "${BASE_CHECKPOINT}" ]]; then
    if [[ -f checkpoints/gpt_old.pth ]]; then
        BASE_CHECKPOINT="checkpoints/gpt_old.pth"
    elif [[ -f checkpoints/gpt.pth ]]; then
        BASE_CHECKPOINT="checkpoints/gpt.pth"
    else
        echo "No base GPT checkpoint found. Provide BASE_CHECKPOINT env or place gpt.pth in checkpoints/." >&2
        exit 1
    fi
fi

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "BASE_CHECKPOINT does not exist: ${BASE_CHECKPOINT}" >&2
    exit 1
fi

echo "[4/5] Preprocessing dataset into cached features"
"${PYTHON_BIN}" tools/preprocess_data.py \
    --manifest "${RAW_MANIFEST}" \
    --output-dir "${PROCESSED_DIR}" \
    --tokenizer "${TOKENIZER_MODEL}" \
    --config "${CONFIG_DST}" \
    --gpt-checkpoint "${BASE_CHECKPOINT}" \
    --language vi \
    --device cuda \
    --val-ratio "${VAL_RATIO}"

if [[ ! -f "${TRAIN_MANIFEST}" || ! -f "${VAL_MANIFEST}" ]]; then
    echo "Processed manifests not found after preprocessing." >&2
    exit 1
fi

echo "[5/5] Building GPT prompt/target pair manifests"
"${PYTHON_BIN}" tools/build_gpt_prompt_pairs.py \
    --manifest "${TRAIN_MANIFEST}" \
    --output "${GPT_TRAIN_MANIFEST}" \
    --pairs-per-target "${PAIRS_PER_TARGET}"

"${PYTHON_BIN}" tools/build_gpt_prompt_pairs.py \
    --manifest "${VAL_MANIFEST}" \
    --output "${GPT_VAL_MANIFEST}" \
    --pairs-per-target "${PAIRS_PER_TARGET}"

echo "Pipeline completed successfully."
echo "  Raw manifest:        ${RAW_MANIFEST}"
echo "  Tokenizer prefix:    ${TOKENIZER_PREFIX}"
echo "  Vietnamese config:   ${CONFIG_DST}"
echo "  Processed data dir:  ${PROCESSED_DIR}"
echo "  GPT train manifest:  ${GPT_TRAIN_MANIFEST}"
echo "  GPT val manifest:    ${GPT_VAL_MANIFEST}"
