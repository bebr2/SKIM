# SKIM

SKIM is an adaptive multi-resolution soft-token compression framework for procedural skills. The release code is organized around the three training stages described in the paper:

- Stage 1: skill reconstruction from skill documents.
- Stage 2: procedural QA warm-up using WikiHow-style document-question-answer data.
- Stage 3: skill task alignment using generated skill-conditioned task data and a shared LoRA adapter.

## Layout

- `code/`: SKIM model, training code, inference helpers, and the vendored SkillRAG/ToolQA adapters used by the benchmark runner.
- `prepare/`: Stage 2 WikiHow conversion and Stage 3 skill task data construction scripts.
- `exam/`: offline resolution-selection exam pipeline.
- `skill/`: SRA-Bench inference runner.
- `scripts/`: parameterized training and benchmark scripts.
- `configs/`: example environment files for the three training stages.

## Training

Install dependencies:

```bash
pip install -r code/requirements.txt
```

Run the stages in order:

```bash
bash scripts/train_stage1_reconstruction.sh
bash scripts/train_stage2_warmup.sh
bash scripts/train_stage3_alignment.sh
```

Set `ENV_FILE` to point at your own environment file when using real data or different models:

```bash
ENV_FILE=configs/stage3_alignment.env bash scripts/train_stage3_alignment.sh
```

The example files in `configs/` only contain relative paths and placeholders.

## Stage 3 Data

The Stage 3 construction pipeline is:

```bash
python prepare/evaluate_skills.py --input-file data/source_skills.jsonl --output-file prepare/output/skill_analysis.jsonl
python prepare/skill_split.py --input-file prepare/output/skill_analysis.jsonl --corpus-file data/source_skills.jsonl --output-file prepare/output/skill_splits.jsonl
python prepare/generate_answer.py --analysis-file prepare/output/skill_analysis.jsonl --origin-file data/source_skills.jsonl --split-file prepare/output/skill_splits.jsonl --direct-output prepare/output/stage3_direct.jsonl --react-output prepare/output/stage3_react.jsonl
```

The generated direct and ReAct files can be referenced from `SKILL_QA_TRAIN_DATA_CONFIG` in `configs/stage3_alignment.env.example`.

## Offline Exam

The exam pipeline generates diagnostic questions, runs full-text and compressed answers, and judges compressed answers against full-text references:

```bash
python exam/generate_questions.py --config exam/config.questions.example.json
python exam/run_inference_via_skill_compiler.py --config exam/config.infer.example.json
python exam/judge_answers.py --config exam/config.judge.example.json
```

## SRA-Bench

Set `SKILLRAG_ROOT` to a local SRA-Bench/SkillRAG checkout and run:

```bash
bash scripts/run_sra_bench.sh ./outputs/stage3_alignment/checkpoint-last skim golden_skill compress 512
bash scripts/run_sra_bench_retrieval.sh ./outputs/stage3_alignment/checkpoint-last skim 5 compress 512
bash scripts/run_toolqa.sh ./outputs/stage3_alignment/checkpoint-last skim golden_skill compress 512
```

All scripts use relative paths by default and can be redirected with environment variables such as `RESULTS_DIR`, `CORPUS_PATH`, `RETRIEVAL_ROOT`, and `TOOLQA_DATA_DIR`.
