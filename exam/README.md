# Skill Exam Pipeline (3-step Split)

This folder now uses three independent scripts and three independent configs.

## Step 1: Generate Questions

Script: `exam/generate_questions.py`

Config: `exam/config.questions.example.json`

Output:
- `instances_dir/<dataset>.json` (aligned with `skill/run_skill_compiler_inference.py` input format)
- raw generated question dump json
- question manifest json

Run:

```bash
python exam/generate_questions.py --config exam/config.questions.example.json
```

## Step 2: Generate Answers (via skill compiler)

Script: `exam/run_inference_via_skill_compiler.py`

Config: `exam/config.infer.example.json`

This script calls `skill/run_skill_compiler_inference.py` directly for:
- `naive`
- `full_text`
- `compress_k*` (multiple k values)

Output:
- `answers_root/<dataset>/<mode>.jsonl`
- answer manifest json

Run:

```bash
python exam/run_inference_via_skill_compiler.py --config exam/config.infer.example.json
```

## Step 3: Judge

Script: `exam/judge_answers.py`

Config: `exam/config.judge.example.json`

Behavior:
- Use `full_text` as reference mode
- Judge candidate modes (`naive`, `compress_k*`) with LLM
- Aggregate per-skill and overall metrics

Output:
- judge summary json
- judge raw json

Run:

```bash
python exam/judge_answers.py --config exam/config.judge.example.json
```

## Config Index

`exam/config.example.json` is now an index file that points to the three split configs.
