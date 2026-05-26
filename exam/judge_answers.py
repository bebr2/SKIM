from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREPARE_ROOT = _REPO_ROOT / "prepare"
if str(_PREPARE_ROOT) not in sys.path:
    sys.path.append(str(_PREPARE_ROOT))

from llm_client import LLMClient


class QPMRateLimiter:
    def __init__(self, qpm: int) -> None:
        if qpm <= 0:
            raise ValueError("qpm must be > 0")
        self.interval = 60.0 / float(qpm)
        self._next_allowed = 0.0

    def acquire(self) -> None:
        now = time.time()
        if now >= self._next_allowed:
            self._next_allowed = now + self.interval
            return
        wait_seconds = self._next_allowed - now
        time.sleep(wait_seconds)
        self._next_allowed = time.time() + self.interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge naive/compress answers against full_text reference answers."
    )
    parser.add_argument("--config", type=str, default="./exam/config.judge.example.json")
    return parser.parse_args()


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _as_non_empty_strings(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if str(x).strip()]


def _parse_datasets(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("datasets") if "datasets" in cfg else cfg.get("dataset")
    if isinstance(raw, str):
        datasets = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, list):
        datasets = _as_non_empty_strings(raw)
    else:
        datasets = []
    if not datasets:
        raise ValueError("dataset/datasets is required")
    return datasets


def load_prompt_config(path_like: str) -> dict[str, Any]:
    path = _resolve_path(path_like)
    cfg = _load_json(path)
    required = ["system_prompt", "user_prompt_template", "output_schema"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Prompt config missing keys {missing}: {path}")
    return cfg


def _is_effective_answer(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return False
    if stripped.strip().startswith("thought 1:"):
        # ToolQA format: check for the filtered action error marker
        if "observation 20: you action is filtered due to content" in stripped:
            return False
        return True
    blocked = ["i don't know", "cannot", "can't", "unable", "unknown"]
    for token in blocked:
        if token in stripped:
            return False
    return True


def _validate_judge_output(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Judge output is not JSON object")

    is_correct = payload.get("is_correct")
    if not isinstance(is_correct, bool):
        raise ValueError("Judge output missing boolean is_correct")

    score = payload.get("score", 1.0 if is_correct else 0.0)
    if not isinstance(score, (int, float)):
        raise ValueError("Judge score must be numeric")

    reason = str(payload.get("reason", "")).strip() or "No reason provided"
    # print(reason)
    # print(score)
    return {
        "is_correct": bool(is_correct),
        "score": float(score),
        "reason": reason,
    }


def judge_answer(
    llm_client: LLMClient,
    limiter: QPMRateLimiter,
    prompt_cfg: dict[str, Any],
    judging_cfg: dict[str, Any],
    dataset: str,
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if reference_answer.strip().startswith("Thought 1:"):
        reference_answer = reference_answer.split("Answer:")[-1].strip()
    if candidate_answer.strip().startswith("Thought 1:"):
        candidate_answer = candidate_answer.split("Answer:")[-1].strip()
    user_prompt = str(prompt_cfg["user_prompt_template"]).format(
        dataset=dataset,
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
        output_schema=json.dumps(prompt_cfg["output_schema"], ensure_ascii=False, indent=2),
    )

    messages = [
        llm_client.build_text_message("system", str(prompt_cfg["system_prompt"])),
        llm_client.build_text_message("user", user_prompt),
    ]

    max_retries = int(judging_cfg.get("max_retries", 3) or 3)
    timeout = float(judging_cfg.get("request_timeout_seconds", 300) or 300)
    max_tokens = int(judging_cfg.get("max_tokens", 1024) or 1024)
    temperature = float(judging_cfg.get("temperature", 0.0) or 0.0)

    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            limiter.acquire()
            payload = llm_client.chat_json(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                timeout=timeout,
                max_retries=1,
            )
            return _validate_judge_output(payload), None
        except Exception as exc:  # pragma: no cover
            # print(f"Judge error: {exc}")
            last_error = exc

    return None, str(last_error)


def _stats_template() -> dict[str, int]:
    return {
        "total": 0,
        "judged_count": 0,
        "correct_count": 0,
        "skip_count": 0,
    }


def _finalize_stats(stats: dict[str, int]) -> dict[str, Any]:
    judged = int(stats.get("judged_count", 0))
    correct = int(stats.get("correct_count", 0))
    accuracy = (correct / judged) if judged > 0 else None
    return {
        "total": int(stats.get("total", 0)),
        "judged_count": judged,
        "correct_count": correct,
        "skip_count": int(stats.get("skip_count", 0)),
        "accuracy": accuracy,
    }


def _load_instances_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Instances file must be a JSON list: {path}")

    mapping: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id", "")).strip()
        if instance_id:
            mapping[instance_id] = row
    return mapping


def _load_answer_map(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        instance_id = str(row.get("instance_id", "")).strip()
        if instance_id:
            mapping[instance_id] = row
    return mapping


def _infer_skill_id(instance: dict[str, Any]) -> str:
    anns = instance.get("skill_annotations")
    if isinstance(anns, list) and anns:
        return str(anns[0]).strip()
    meta = instance.get("meta")
    if isinstance(meta, dict):
        return str(meta.get("skill_id", "")).strip()
    return ""


def run(cfg: dict[str, Any]) -> None:
    datasets = _parse_datasets(cfg)

    instances_dir = _resolve_path(str(cfg.get("instances_dir", "./exam/output/instances")))
    answers_root = _resolve_path(str(cfg.get("answers_root", "./exam/output/answers")))

    reference_mode = str(cfg.get("reference_mode", "full_text"))
    candidate_modes = _as_non_empty_strings(cfg.get("candidate_modes", []))

    if not candidate_modes:
        candidate_modes = ["naive"]

    judging_cfg = cfg.get("judging", {})
    prompt_cfg = load_prompt_config(str(judging_cfg.get("prompt_file", "./exam/judge_prompt.json")))

    llm_client = LLMClient.from_env(
        model=None,
        max_tokens=int(judging_cfg.get("max_tokens", 1024) or 1024),
        temperature=float(judging_cfg.get("temperature", 0.0) or 0.0),
    )
    limiter = QPMRateLimiter(int(judging_cfg.get("qpm", 60) or 60))

    all_modes = [reference_mode] + candidate_modes
    overall_stats: dict[str, dict[str, int]] = {mode: _stats_template() for mode in all_modes}

    summary_datasets: list[dict[str, Any]] = []
    raw_datasets: list[dict[str, Any]] = []

    for dataset in datasets:
        instances_path = instances_dir / f"{dataset}.json"
        if not instances_path.exists():
            raise FileNotFoundError(f"Instances file not found: {instances_path}")

        instances_map = _load_instances_map(instances_path)
        dataset_instance_ids = list(instances_map.keys())

        answer_maps: dict[str, dict[str, dict[str, Any]]] = {}
        for mode in all_modes:
            answer_path = answers_root / dataset / f"{mode}.jsonl"
            if not answer_path.exists():
                if mode == reference_mode:
                    raise FileNotFoundError(f"Reference answer file not found: {answer_path}")
                answer_maps[mode] = {}
                continue
            answer_maps[mode] = _load_answer_map(answer_path)

        dataset_stats: dict[str, dict[str, int]] = {mode: _stats_template() for mode in all_modes}
        skill_stats_map: dict[str, dict[str, dict[str, int]]] = {}
        raw_rows: list[dict[str, Any]] = []

        bar = tqdm(dataset_instance_ids, desc=f"Judge-{dataset}", dynamic_ncols=True) if tqdm else dataset_instance_ids
        for instance_id in bar:
            instance = instances_map[instance_id]
            question = str(instance.get("question", ""))
            skill_id = _infer_skill_id(instance)
            if skill_id not in skill_stats_map:
                skill_stats_map[skill_id] = {mode: _stats_template() for mode in all_modes}

            row = {
                "instance_id": instance_id,
                "dataset": dataset,
                "skill_id": skill_id,
                "question": question,
                "answers": {},
                "judgements": {},
            }

            for mode in all_modes:
                dataset_stats[mode]["total"] += 1
                overall_stats[mode]["total"] += 1
                skill_stats_map[skill_id][mode]["total"] += 1

                answer_row = answer_maps.get(mode, {}).get(instance_id, {})
                row["answers"][mode] = {
                    "raw_output": str(answer_row.get("raw_output", "")),
                    "error": answer_row.get("error"),
                }

            ref_answer = str(row["answers"][reference_mode].get("raw_output", ""))
            ref_error = row["answers"][reference_mode].get("error")
            ref_valid = _is_effective_answer(ref_answer) and (not ref_error)

            if not ref_valid:
                for mode in all_modes:
                    dataset_stats[mode]["skip_count"] += 1
                    overall_stats[mode]["skip_count"] += 1
                    skill_stats_map[skill_id][mode]["skip_count"] += 1
                    row["judgements"][mode] = {
                        "is_correct": None,
                        "score": None,
                        "reason": "Skipped because reference answer is empty/invalid.",
                        "error": None,
                    }
                raw_rows.append(row)
                continue

            dataset_stats[reference_mode]["judged_count"] += 1
            dataset_stats[reference_mode]["correct_count"] += 1
            overall_stats[reference_mode]["judged_count"] += 1
            overall_stats[reference_mode]["correct_count"] += 1
            skill_stats_map[skill_id][reference_mode]["judged_count"] += 1
            skill_stats_map[skill_id][reference_mode]["correct_count"] += 1
            row["judgements"][reference_mode] = {
                "is_correct": True,
                "score": 1.0,
                "reason": "Reference mode.",
                "error": None,
            }

            for mode in candidate_modes:
                candidate_answer = str(row["answers"][mode].get("raw_output", ""))
                candidate_error = row["answers"][mode].get("error")

                if candidate_error:
                    dataset_stats[mode]["skip_count"] += 1
                    overall_stats[mode]["skip_count"] += 1
                    skill_stats_map[skill_id][mode]["skip_count"] += 1
                    row["judgements"][mode] = {
                        "is_correct": None,
                        "score": None,
                        "reason": "Skipped because candidate generation failed.",
                        "error": str(candidate_error),
                    }
                    continue

                if not _is_effective_answer(candidate_answer):
                    dataset_stats[mode]["judged_count"] += 1
                    overall_stats[mode]["judged_count"] += 1
                    skill_stats_map[skill_id][mode]["judged_count"] += 1
                    row["judgements"][mode] = {
                        "is_correct": False,
                        "score": 0.0,
                        "reason": "Candidate answer is empty/invalid.",
                        "error": None,
                    }
                    continue

                judge_result, judge_error = judge_answer(
                    llm_client=llm_client,
                    limiter=limiter,
                    prompt_cfg=prompt_cfg,
                    judging_cfg=judging_cfg,
                    dataset=dataset,
                    question=question,
                    reference_answer=ref_answer,
                    candidate_answer=candidate_answer,
                )
                if judge_error:
                    dataset_stats[mode]["skip_count"] += 1
                    overall_stats[mode]["skip_count"] += 1
                    skill_stats_map[skill_id][mode]["skip_count"] += 1
                    row["judgements"][mode] = {
                        "is_correct": None,
                        "score": None,
                        "reason": "Skipped because judge model failed.",
                        "error": judge_error,
                    }
                    continue

                assert judge_result is not None
                dataset_stats[mode]["judged_count"] += 1
                overall_stats[mode]["judged_count"] += 1
                skill_stats_map[skill_id][mode]["judged_count"] += 1
                if bool(judge_result["is_correct"]):
                    dataset_stats[mode]["correct_count"] += 1
                    overall_stats[mode]["correct_count"] += 1
                    skill_stats_map[skill_id][mode]["correct_count"] += 1

                row["judgements"][mode] = {
                    "is_correct": bool(judge_result["is_correct"]),
                    "score": float(judge_result["score"]),
                    "reason": str(judge_result["reason"]),
                    "error": None,
                }

            raw_rows.append(row)

        per_skill = []
        for skill_id, stats_by_mode in skill_stats_map.items():
            per_skill.append(
                {
                    "skill_id": skill_id,
                    "metrics": {mode: _finalize_stats(stats) for mode, stats in stats_by_mode.items()},
                }
            )

        summary_datasets.append(
            {
                "dataset": dataset,
                "n_instances": len(dataset_instance_ids),
                "per_skill": per_skill,
                "overall": {mode: _finalize_stats(stats) for mode, stats in dataset_stats.items()},
            }
        )
        raw_datasets.append({"dataset": dataset, "rows": raw_rows})
        print(f"[judge] dataset={dataset}, instances={len(dataset_instance_ids)}")

    summary = {
        "datasets": datasets,
        "reference_mode": reference_mode,
        "candidate_modes": candidate_modes,
        "per_dataset": summary_datasets,
        "overall": {mode: _finalize_stats(stats) for mode, stats in overall_stats.items()},
    }
    raw_output = {
        "meta": {
            "datasets": datasets,
            "reference_mode": reference_mode,
            "candidate_modes": candidate_modes,
        },
        "datasets": raw_datasets,
    }

    out_cfg = cfg.get("output", {})
    summary_json = _resolve_path(str(out_cfg.get("summary_json", "./exam/output/judge_summary.json")))
    raw_json = _resolve_path(str(out_cfg.get("raw_json", "./exam/output/judge_raw.json")))
    overwrite = bool(out_cfg.get("overwrite", False))

    if not overwrite:
        collisions = [p for p in [summary_json, raw_json] if p.exists()]
        if collisions:
            raise FileExistsError(
                "Output files already exist. Set output.overwrite=true or change output paths: "
                + ", ".join([str(x) for x in collisions])
            )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    raw_json.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(raw_output, f, ensure_ascii=False, indent=2)

    print(f"[judge] wrote summary -> {summary_json}")
    print(f"[judge] wrote raw -> {raw_json}")


def main() -> None:
    args = parse_args()
    cfg = _load_json(_resolve_path(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
