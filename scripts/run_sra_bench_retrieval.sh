#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-${CHECKPOINT:-./outputs/stage3_alignment/checkpoint-last}}"
MODEL_NAME="${2:-${MODEL_NAME:-skim}}"
TOP_K="${3:-${TOP_K:-5}}"
SKILL_MODE="${4:-${SKILL_MODE:-compress}}"
K="${5:-${K:-512}}"

SKILLRAG_ROOT="${SKILLRAG_ROOT:-./external/SkillRAG}"
CORPUS_PATH="${CORPUS_PATH:-${SKILLRAG_ROOT}/data/bench/corpus/corpus.json}"
RETRIEVAL_ROOT="${RETRIEVAL_ROOT:-${SKILLRAG_ROOT}/retrieval}"
RESULTS_DIR="${RESULTS_DIR:-./results/sra_bench_retrieval}"
FP16="${FP16:-false}"

DATASETS="${DATASETS:-bigcodebench champ logicbench theoremqa}"

for DATASET in ${DATASETS}; do
  python skill/run_skill_compiler_inference.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --method retrieval \
    --top_k "${TOP_K}" \
    --retrieval_results "${RETRIEVAL_ROOT}/${DATASET}/bge_strength.json" \
    --skill_mode "${SKILL_MODE}" \
    --fp16 "${FP16}" \
    --k "${K}" \
    --skillrag_root "${SKILLRAG_ROOT}" \
    --corpus_path "${CORPUS_PATH}" \
    --result_path "${RESULTS_DIR}/${DATASET}/${MODEL_NAME}/retrieval.top${TOP_K}.${SKILL_MODE}.k${K}.jsonl"
done
