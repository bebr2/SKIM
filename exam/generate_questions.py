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

ALL_DATASETS = ["theoremqa", "logicbench", "toolqa", "champ", "medcalcbench", "bigcodebench"]


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
        description=(
            "Generate benchmark questions per skill and export instance files aligned with "
            "skill/run_skill_compiler_inference.py input format."
        )
    )
    parser.add_argument("--config", type=str, default="./exam/config.questions.example.json")
    return parser.parse_args()


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_skill_items(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        has_skill_id = "skill_id" in obj
        has_compatible_id = "id" in obj and (
            "content" in obj or "description" in obj or "name" in obj
        )
        if has_skill_id or has_compatible_id:
            out.append(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _collect_skill_items(value, out)
        return

    if isinstance(obj, list):
        for item in obj:
            _collect_skill_items(item, out)


def _resolve_skill_id(row: dict[str, Any]) -> str:
    sid = str(row.get("skill_id", "")).strip()
    if sid:
        return sid
    return str(row.get("id", "")).strip()


def load_skill_corpus(path_like: str) -> dict[str, dict[str, Any]]:
    corpus_path = _resolve_path(path_like)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    mapping: dict[str, dict[str, Any]] = {}
    if corpus_path.suffix.lower() == ".jsonl":
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    continue
                sid = _resolve_skill_id(row)
                if not sid:
                    continue
                fixed = dict(row)
                fixed["skill_id"] = sid
                mapping[sid] = fixed
    else:
        payload = _load_json(corpus_path)
        items: list[dict[str, Any]] = []
        _collect_skill_items(payload, items)
        for row in items:
            sid = _resolve_skill_id(row)
            if not sid:
                continue
            fixed = dict(row)
            fixed["skill_id"] = sid
            mapping[sid] = fixed

    if not mapping:
        raise ValueError(f"No skills found in corpus: {corpus_path}")
    return mapping


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

    uniq: list[str] = []
    seen: set[str] = set()
    for name in datasets:
        if name in seen:
            continue
        seen.add(name)
        uniq.append(name)

    invalid = [x for x in uniq if x not in ALL_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}")
    return uniq


def _select_skill_ids_for_dataset(corpus: dict[str, dict[str, Any]], dataset: str, max_skills: int) -> list[str]:
    selected = sorted([sid for sid in corpus.keys() if sid.startswith(dataset)])
    if max_skills > 0:
        return selected[:max_skills]
    return selected


def load_prompt_config(path_like: str) -> dict[str, Any]:
    path = _resolve_path(path_like)
    cfg = _load_json(path)
    required = ["system_prompt", "user_prompt_template", "output_schema"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Prompt config missing keys {missing}: {path}")
    return cfg


def _validate_questions(payload: Any, n_questions: int) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("Question output is not JSON object")
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Question output missing list field 'questions'")

    cleaned = [str(x).strip() for x in questions if str(x).strip()]
    if len(cleaned) < n_questions:
        raise ValueError(f"Got {len(cleaned)} questions < required {n_questions}")
    return cleaned[:n_questions]


def generate_questions_for_skill(
    llm_client: LLMClient,
    limiter: QPMRateLimiter,
    prompt_cfg: dict[str, Any],
    q_cfg: dict[str, Any],
    dataset: str,
    skill_row: dict[str, Any],
) -> tuple[list[str], str | None]:
    n_questions = int(q_cfg.get("n_questions_per_skill", 0) or 0)
    if n_questions <= 0:
        raise ValueError("question_generation.n_questions_per_skill must be > 0")

    skill_payload = {
        "skill_id": skill_row.get("skill_id"),
        "name": skill_row.get("name"),
        "description": skill_row.get("description"),
        "content": skill_row.get("content"),
        "owner": skill_row.get("owner"),
        "repo": skill_row.get("repo"),
        "tools": skill_row.get("tools"),
    }

    user_prompt = str(prompt_cfg["user_prompt_template"]).format(
        dataset=dataset,
        n_questions=n_questions,
        output_schema=json.dumps(prompt_cfg["output_schema"], ensure_ascii=False, indent=2),
        skill_json=json.dumps(skill_payload, ensure_ascii=False, indent=2),
    )

    messages = [
        llm_client.build_text_message("system", str(prompt_cfg["system_prompt"])),
        llm_client.build_text_message("user", user_prompt),
    ]

    max_retries = int(q_cfg.get("max_retries", 3) or 3)
    timeout = float(q_cfg.get("request_timeout_seconds", 300) or 300)
    max_tokens = int(q_cfg.get("max_tokens", 2048) or 2048)
    temperature = float(q_cfg.get("temperature", 0.2) or 0.2)

    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            limiter.acquire()
            result = llm_client.chat_json(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                timeout=timeout,
                max_retries=1,
            )
            return _validate_questions(result, n_questions=n_questions), None
        except Exception as exc:  # pragma: no cover
            last_error = exc

    return [], str(last_error)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _ensure_write_paths(instances_dir: Path, questions_json: Path, manifest_json: Path, overwrite: bool) -> None:
    instances_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return

    collisions: list[Path] = []
    if questions_json.exists():
        collisions.append(questions_json)
    if manifest_json.exists():
        collisions.append(manifest_json)
    if collisions:
        raise FileExistsError(
            "Output files already exist. Set output.overwrite=true or change output paths: "
            + ", ".join([str(x) for x in collisions])
        )


def run(cfg: dict[str, Any]) -> None:
    datasets = _parse_datasets(cfg)
    corpus = load_skill_corpus(str(cfg.get("corpus_path", "")))

    q_cfg = cfg.get("question_generation", {})
    out_cfg = cfg.get("output", {})

    prompt_cfg = load_prompt_config(str(q_cfg.get("prompt_file", "./exam/question_prompt.json")))
    llm_client = LLMClient.from_env(
        model=None,
        max_tokens=int(q_cfg.get("max_tokens", 2048) or 2048),
        temperature=float(q_cfg.get("temperature", 0.2) or 0.2),
    )
    limiter = QPMRateLimiter(int(q_cfg.get("qpm", 60) or 60))

    instances_dir = _resolve_path(str(out_cfg.get("instances_dir", "./exam/output/instances")))
    questions_json = _resolve_path(str(out_cfg.get("questions_json", "./exam/output/questions_generated.json")))
    manifest_json = _resolve_path(str(out_cfg.get("manifest_json", "./exam/output/questions_manifest.json")))
    overwrite = bool(out_cfg.get("overwrite", False))
    max_skills = int(out_cfg.get("max_skills_per_dataset", 0) or 0)

    _ensure_write_paths(instances_dir, questions_json, manifest_json, overwrite=overwrite)

    all_question_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "datasets": datasets,
        "instances_dir": str(instances_dir),
        "per_dataset": [],
    }

    for dataset in datasets:
        skill_ids = _select_skill_ids_for_dataset(corpus, dataset, max_skills=max_skills)
        instances: list[dict[str, Any]] = []

        print(f"[questions] dataset={dataset}, matched_skills={len(skill_ids)}")
        bar = tqdm(skill_ids, desc=f"Questions-{dataset}", dynamic_ncols=True) if tqdm else skill_ids
        for skill_id in bar:
            skill_row = corpus[skill_id]
            questions, error = generate_questions_for_skill(
                llm_client=llm_client,
                limiter=limiter,
                prompt_cfg=prompt_cfg,
                q_cfg=q_cfg,
                dataset=dataset,
                skill_row=skill_row,
            )

            if error:
                all_question_rows.append(
                    {
                        "dataset": dataset,
                        "skill_id": skill_id,
                        "questions": [],
                        "error": error,
                    }
                )
                continue

            for q_idx, question in enumerate(questions, start=1):
                instance_id = f"exam::{dataset}::{skill_id}::{q_idx}"
                instance = {
                    "instance_id": instance_id,
                    "dataset": dataset,
                    "question": question,
                    "skill_annotations": [skill_id],
                    "meta": {
                        "source": "exam.generate_questions",
                        "skill_id": skill_id,
                        "skill_name": str(skill_row.get("name", "")),
                    },
                }
                instances.append(instance)

            all_question_rows.append(
                {
                    "dataset": dataset,
                    "skill_id": skill_id,
                    "questions": questions,
                    "error": None,
                }
            )

        dataset_path = instances_dir / f"{dataset}.json"
        _write_json(dataset_path, instances)

        manifest["per_dataset"].append(
            {
                "dataset": dataset,
                "instances_file": str(dataset_path),
                "n_instances": len(instances),
                "n_skills": len(skill_ids),
            }
        )
        print(f"[questions] wrote {len(instances)} instances -> {dataset_path}")

    _write_json(questions_json, all_question_rows)
    _write_json(manifest_json, manifest)
    print(f"[questions] wrote raw question dump -> {questions_json}")
    print(f"[questions] wrote manifest -> {manifest_json}")


def main() -> None:
    args = parse_args()
    cfg = _load_json(_resolve_path(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
