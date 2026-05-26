#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-${CHECKPOINT:-./outputs/stage3_alignment/checkpoint-last}}"
MODEL_NAME="${2:-${MODEL_NAME:-skim}}"
METHOD="${3:-${METHOD:-golden_skill}}"
SKILL_MODE="${4:-${SKILL_MODE:-compress}}"
K="${5:-${K:-512}}"

SKILLRAG_ROOT="${SKILLRAG_ROOT:-./external/SkillRAG}"
CORPUS_PATH="${CORPUS_PATH:-${SKILLRAG_ROOT}/data/bench/corpus/corpus.json}"
TOOLQA_DATA_DIR="${TOOLQA_DATA_DIR:-${SKILLRAG_ROOT}/data/external_corpus}"
RESULTS_DIR="${RESULTS_DIR:-./results/toolqa}"
FP16="${FP16:-false}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-1}"

python skill/run_skill_compiler_inference.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset toolqa \
  --method "${METHOD}" \
  --skill_mode "${SKILL_MODE}" \
  --fp16 "${FP16}" \
  --k "${K}" \
  --toolqa_data_dir "${TOOLQA_DATA_DIR}" \
  --skillrag_root "${SKILLRAG_ROOT}" \
  --corpus_path "${CORPUS_PATH}" \
  --gpus_per_worker "${GPUS_PER_WORKER}" \
  --result_path "${RESULTS_DIR}/${MODEL_NAME}/${METHOD}.${SKILL_MODE}.k${K}.jsonl"
