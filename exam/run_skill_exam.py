from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_ROOT = _REPO_ROOT / "code"
_PREPARE_ROOT = _REPO_ROOT / "prepare"
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_PREPARE_ROOT) not in sys.path:
    sys.path.append(str(_PREPARE_ROOT))

from skim_inference import build_model, generate_with_skills, load_checkpoint_meta
from llm_client import LLMClient
from skillrag_vendor.prompts import ALL_DATASETS, build_prompt


class QPMRateLimiter:
    """Global request limiter that controls request start rate by QPM."""

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
            "Run a skill benchmark exam: generate per-skill questions, answer with "
            "naive/full_text/compress-k modes, and score against full_text ground truth."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exam/config.example.json",
        help="Path to exam config JSON",
    )
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
    sid = str(row.get("id", "")).strip()
    return sid


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


def load_prompt_config(path_like: str) -> dict[str, Any]:
    path = _resolve_path(path_like)
    cfg = _load_json(path)
    required = ["system_prompt", "user_prompt_template", "output_schema"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Prompt config missing keys {missing}: {path}")
    return cfg


def _as_non_empty_strings(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for x in items:
        text = str(x).strip()
        if text:
            out.append(text)
    return out


def _parse_datasets(cfg: dict[str, Any]) -> list[str]:
    if "datasets" in cfg:
        raw = cfg.get("datasets")
    else:
        raw = cfg.get("dataset")

    if isinstance(raw, str):
        datasets = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, list):
        datasets = _as_non_empty_strings(raw)
    else:
        datasets = []

    if not datasets:
        raise ValueError("dataset/datasets is required in config")

    # Keep input order while removing duplicates.
    seen: set[str] = set()
    uniq: list[str] = []
    for name in datasets:
        if name in seen:
            continue
        seen.add(name)
        uniq.append(name)

    invalid = [name for name in uniq if name not in ALL_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported dataset(s) {invalid}, available: {ALL_DATASETS}")
    return uniq


def _select_skill_ids_for_dataset(corpus: dict[str, dict[str, Any]], dataset: str) -> list[str]:
    return sorted([sid for sid in corpus.keys() if sid.startswith(dataset)])


def _extract_skill_payload(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "skill_id",
        "id",
        "name",
        "description",
        "content",
        "owner",
        "repo",
        "tool_specs",
        "tools",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        if key in row:
            out[key] = row[key]
    if "skill_id" not in out:
        out["skill_id"] = _resolve_skill_id(row)
    return out


def _validate_question_result(payload: Any, n_questions: int) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("Question generator output is not a JSON object")

    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Question generator output must contain a list field 'questions'")

    cleaned: list[str] = []
    for item in questions:
        text = str(item).strip()
        if text:
            cleaned.append(text)

    if len(cleaned) < n_questions:
        raise ValueError(f"Question generator returned {len(cleaned)} < {n_questions}")
    return cleaned[:n_questions]


def generate_questions_for_skill(
    llm_client: LLMClient,
    limiter: QPMRateLimiter,
    prompt_cfg: dict[str, Any],
    dataset: str,
    skill_id: str,
    skill_row: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[str], str | None]:
    q_cfg = cfg.get("question_generation", {})
    n_questions = int(q_cfg.get("n_questions_per_skill", 0) or 0)
    if n_questions <= 0:
        raise ValueError("question_generation.n_questions_per_skill must be > 0")

    payload = _extract_skill_payload(skill_row)
    user_prompt = str(prompt_cfg["user_prompt_template"]).format(
        dataset=dataset,
        n_questions=n_questions,
        output_schema=json.dumps(prompt_cfg["output_schema"], ensure_ascii=False, indent=2),
        skill_json=json.dumps(payload, ensure_ascii=False, indent=2),
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
    for _attempt in range(1, max_retries + 1):
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
            questions = _validate_question_result(result, n_questions=n_questions)
            return questions, None
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_error = exc

    return [], str(last_error)


def _inject_skill_block(
    user_prompt: str,
    skill_ids: list[str],
    skill_mode: str,
    corpus: dict[str, dict[str, Any]],
) -> str:
    if not skill_ids:
        return user_prompt

    if skill_mode == "compress":
        skill_block = "\n---\n".join([f"<skill>{sid}</skill>" for sid in skill_ids])
    else:
        skill_texts: list[str] = []
        for sid in skill_ids:
            row = corpus.get(sid, {})
            content = row.get("content", "") if isinstance(row, dict) else ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            skill_texts.append(content)
        skill_block = "\n---\n".join(skill_texts)

    return f"Relevant Skill:\n{skill_block}\n\n{user_prompt}"


def _build_case_prompt(
    dataset: str,
    question: str,
    skill_ids: list[str],
    corpus: dict[str, dict[str, Any]],
    skill_mode: str,
) -> tuple[str, str]:
    instance = {
        "dataset": dataset,
        "question": question,
    }
    system_text, user_text = build_prompt(instance, method="naive")
    user_text = _inject_skill_block(user_text, skill_ids=skill_ids, skill_mode=skill_mode, corpus=corpus)
    return system_text, user_text


def _build_model_args(cfg: dict[str, Any]) -> argparse.Namespace:
    infer_cfg = cfg.get("inference", {})
    fallback = cfg.get("model_fallback", {})

    args = argparse.Namespace()
    args.checkpoint = str(_resolve_path(str(cfg.get("checkpoint", ""))))
    args.skill_mode = "compress"
    args.k = int((infer_cfg.get("compress_k_list", [64]) or [64])[0])
    args.max_length = int(infer_cfg.get("max_length", 4096) or 4096)
    args.max_new_tokens = int(infer_cfg.get("max_new_tokens", 1024) or 1024)
    args.do_sample = bool(infer_cfg.get("do_sample", False))
    args.temperature = float(infer_cfg.get("temperature", 0.7) or 0.7)
    args.top_p = float(infer_cfg.get("top_p", 0.95) or 0.95)

    args.compressor_model = str(fallback.get("compressor_model", ""))
    args.llm_model = str(fallback.get("llm_model", ""))
    args.projector_layers = int(fallback.get("projector_layers", 3) or 3)
    args.projector_hidden = int(fallback.get("projector_hidden", 4096) or 4096)
    args.max_q = int(fallback.get("max_q", 256) or 256)
    return args


def _build_modes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    infer_cfg = cfg.get("inference", {})
    modes: list[dict[str, Any]] = []

    include_naive = bool(infer_cfg.get("include_naive", True))
    include_full_text = bool(infer_cfg.get("include_full_text", True))

    if include_naive:
        modes.append(
            {
                "name": "naive",
                "skill_mode": "naive",
                "inject_skills": False,
                "k": None,
            }
        )

    if include_full_text:
        modes.append(
            {
                "name": "full_text",
                "skill_mode": "full_text",
                "inject_skills": True,
                "k": None,
            }
        )

    k_list = infer_cfg.get("compress_k_list", [])
    if not isinstance(k_list, list):
        raise ValueError("inference.compress_k_list must be a list")

    seen: set[int] = set()
    for item in k_list:
        k = int(item)
        if k <= 0:
            raise ValueError(f"compress k must be > 0, got {k}")
        if k in seen:
            continue
        seen.add(k)
        modes.append(
            {
                "name": f"compress_k{k}",
                "skill_mode": "compress",
                "inject_skills": True,
                "k": k,
            }
        )

    if not any(x["name"] == "full_text" for x in modes):
        raise ValueError("inference.include_full_text must be true (full_text is required as ground truth)")

    if len(modes) <= 1:
        raise ValueError("Need at least one compared mode besides full_text")

    return modes


def _is_effective_answer(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return False
    blocked = [
        "i don't know",
        "cannot",
        "can't",
        "unable",
        "unknown",
    ]
    for token in blocked:
        if token in stripped:
            return False
    return True


def run_mode_answer(
    model,
    corpus: dict[str, dict[str, Any]],
    dataset: str,
    question: str,
    skill_id: str,
    mode: dict[str, Any],
    infer_cfg: dict[str, Any],
    default_k: int,
) -> tuple[str, str | None]:
    skill_mode = str(mode["skill_mode"])
    use_skill = bool(mode["inject_skills"])
    k_value = int(mode["k"]) if mode.get("k") is not None else int(default_k)

    skill_ids = [skill_id] if use_skill else []
    system_text, user_text = _build_case_prompt(
        dataset=dataset,
        question=question,
        skill_ids=skill_ids,
        corpus=corpus,
        skill_mode=skill_mode,
    )

    generated, _embeds = generate_with_skills(
        model=model,
        corpus=corpus,
        system_text=system_text,
        user_text=user_text,
        skill_mode=skill_mode,
        k=k_value,
        max_length=int(infer_cfg.get("max_length", 4096) or 4096),
        max_new_tokens=int(infer_cfg.get("max_new_tokens", 1024) or 1024),
        do_sample=bool(infer_cfg.get("do_sample", False)),
        temperature=float(infer_cfg.get("temperature", 0.7) or 0.7),
        top_p=float(infer_cfg.get("top_p", 0.95) or 0.95),
    )
    return generated, None


def _split_evenly(items: list[tuple[int, str]], num_parts: int) -> list[list[tuple[int, str]]]:
    num_parts = max(1, int(num_parts))
    shards: list[list[tuple[int, str]]] = [[] for _ in range(num_parts)]
    for idx, item in enumerate(items):
        shards[idx % num_parts].append(item)
    return shards


def _infer_answer_row(
    model,
    corpus: dict[str, dict[str, Any]],
    dataset: str,
    question_id: int,
    question: str,
    skill_id: str,
    modes: list[dict[str, Any]],
    infer_cfg: dict[str, Any],
    default_k: int,
) -> dict[str, Any]:
    q_row: dict[str, Any] = {
        "question_id": int(question_id),
        "question": question,
        "answers": {},
        "judgements": {},
        "groundtruth_mode": "full_text",
        "groundtruth_answer": "",
        "groundtruth_valid": False,
    }

    for mode in modes:
        mode_name = str(mode["name"])
        started = time.time()
        try:
            answer, _err = run_mode_answer(
                model=model,
                corpus=corpus,
                dataset=dataset,
                question=question,
                skill_id=skill_id,
                mode=mode,
                infer_cfg=infer_cfg,
                default_k=default_k,
            )
            q_row["answers"][mode_name] = {
                "raw_output": answer,
                "error": None,
                "elapsed_seconds": time.time() - started,
            }
        except Exception as exc:  # pragma: no cover - runtime dependent
            q_row["answers"][mode_name] = {
                "raw_output": "",
                "error": str(exc),
                "elapsed_seconds": time.time() - started,
            }
    return q_row


def _worker_answer_service(
    worker_id: int,
    gpu_id: int,
    task_queue,
    result_queue,
    model_args_dict: dict[str, Any],
    meta: dict[str, Any],
    corpus_path: str,
) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

        args = argparse.Namespace(**model_args_dict)
        model = build_model(args, meta).to(device)
        model.eval()
        corpus = load_skill_corpus(corpus_path)

        result_queue.put({"kind": "ready", "worker_id": worker_id})
    except Exception as exc:
        result_queue.put(
            {
                "kind": "init_error",
                "worker_id": worker_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            return

        request_id = int(task.get("request_id", -1))
        shard = task.get("shard", [])
        dataset = str(task.get("dataset", ""))
        skill_id = str(task.get("skill_id", ""))
        modes = task.get("modes", [])
        infer_cfg = task.get("infer_cfg", {})
        default_k = int(task.get("default_k", 64))

        try:
            rows: list[dict[str, Any]] = []
            for question_id, question in shard:
                row = _infer_answer_row(
                    model=model,
                    corpus=corpus,
                    dataset=dataset,
                    question_id=int(question_id),
                    question=str(question),
                    skill_id=skill_id,
                    modes=modes,
                    infer_cfg=infer_cfg,
                    default_k=default_k,
                )
                rows.append(row)

            result_queue.put(
                {
                    "kind": "task_result",
                    "worker_id": worker_id,
                    "request_id": request_id,
                    "rows": rows,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "kind": "task_error",
                    "worker_id": worker_id,
                    "request_id": request_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


class _ParallelAnswerPool:
    def __init__(
        self,
        num_workers: int,
        model_args_dict: dict[str, Any],
        meta: dict[str, Any],
        corpus_path: str,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self._ctx = mp.get_context("spawn")
        self._result_queue = self._ctx.Queue()
        self._task_queues = []
        self._processes = []
        self._request_id = 0

        for worker_id in range(self.num_workers):
            task_queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=_worker_answer_service,
                args=(
                    worker_id,
                    worker_id,
                    task_queue,
                    self._result_queue,
                    model_args_dict,
                    meta,
                    corpus_path,
                ),
            )
            process.start()
            self._task_queues.append(task_queue)
            self._processes.append(process)

        ready_workers: set[int] = set()
        init_errors: list[dict[str, Any]] = []
        while len(ready_workers) + len(init_errors) < self.num_workers:
            try:
                payload = self._result_queue.get(timeout=600)
            except Empty:
                break

            kind = str(payload.get("kind", ""))
            if kind == "ready":
                ready_workers.add(int(payload.get("worker_id", -1)))
            elif kind == "init_error":
                init_errors.append(payload)

        abnormal_exit = [f"pid={p.pid}, exitcode={p.exitcode}" for p in self._processes if p.exitcode not in (0, None)]
        if init_errors or len(ready_workers) != self.num_workers or abnormal_exit:
            details = []
            for err in init_errors:
                details.append(
                    f"worker={err.get('worker_id')}, error={err.get('error')}\n{err.get('traceback', '')}"
                )
            if len(ready_workers) != self.num_workers:
                details.append(f"ready_workers={sorted(list(ready_workers))}, expected={self.num_workers}")
            details.extend(abnormal_exit)
            self.stop()
            raise RuntimeError("Failed to initialize parallel answer pool:\n" + "\n".join(details))

    def stop(self) -> None:
        for q in self._task_queues:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._processes:
            if p.is_alive():
                p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def run_skill(
        self,
        dataset: str,
        skill_id: str,
        questions: list[str],
        modes: list[dict[str, Any]],
        infer_cfg: dict[str, Any],
        default_k: int,
    ) -> list[dict[str, Any]]:
        indexed_questions = [(idx, q) for idx, q in enumerate(questions, start=1)]
        shards = _split_evenly(indexed_questions, self.num_workers)

        self._request_id += 1
        request_id = self._request_id

        dispatched = 0
        for worker_id, shard in enumerate(shards):
            if not shard:
                continue
            payload = {
                "request_id": request_id,
                "dataset": dataset,
                "skill_id": skill_id,
                "shard": shard,
                "modes": modes,
                "infer_cfg": infer_cfg,
                "default_k": default_k,
            }
            self._task_queues[worker_id].put(payload)
            dispatched += 1

        if dispatched == 0:
            return []

        merged: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        finished = 0

        while finished < dispatched:
            payload = self._result_queue.get()
            kind = str(payload.get("kind", ""))
            payload_request_id = int(payload.get("request_id", -1))

            if kind in {"task_result", "task_error"} and payload_request_id != request_id:
                continue

            if kind == "task_result":
                merged.extend(payload.get("rows", []))
                finished += 1
            elif kind == "task_error":
                errors.append(payload)
                finished += 1

        if errors:
            details = []
            for err in errors:
                details.append(
                    f"worker={err.get('worker_id')}, error={err.get('error')}\n{err.get('traceback', '')}"
                )
            raise RuntimeError("Parallel answer generation failed:\n" + "\n".join(details))

        merged.sort(key=lambda x: int(x.get("question_id", 0)))
        if len(merged) != len(indexed_questions):
            raise RuntimeError(
                f"Parallel result count mismatch: got {len(merged)}, expected {len(indexed_questions)}"
            )
        return merged


def _run_serial_answer_generation(
    questions: list[str],
    model,
    corpus: dict[str, dict[str, Any]],
    dataset: str,
    skill_id: str,
    modes: list[dict[str, Any]],
    infer_cfg: dict[str, Any],
    default_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_id, question in enumerate(questions, start=1):
        rows.append(
            _infer_answer_row(
                model=model,
                corpus=corpus,
                dataset=dataset,
                question_id=question_id,
                question=question,
                skill_id=skill_id,
                modes=modes,
                infer_cfg=infer_cfg,
                default_k=default_k,
            )
        )
    return rows


def _validate_judge_output(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Judge output is not a JSON object")

    is_correct = payload.get("is_correct")
    if not isinstance(is_correct, bool):
        raise ValueError("Judge output is missing boolean field is_correct")

    score = payload.get("score", 1.0 if is_correct else 0.0)
    if not isinstance(score, (int, float)):
        raise ValueError("Judge output field score must be numeric")

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided"

    return {
        "is_correct": bool(is_correct),
        "score": float(score),
        "reason": reason,
    }


def judge_answer(
    llm_client: LLMClient,
    limiter: QPMRateLimiter,
    prompt_cfg: dict[str, Any],
    cfg: dict[str, Any],
    dataset: str,
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> tuple[dict[str, Any] | None, str | None]:
    j_cfg = cfg.get("judging", {})
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

    max_retries = int(j_cfg.get("max_retries", 3) or 3)
    timeout = float(j_cfg.get("request_timeout_seconds", 300) or 300)
    max_tokens = int(j_cfg.get("max_tokens", 1024) or 1024)
    temperature = float(j_cfg.get("temperature", 0.0) or 0.0)

    last_error: Exception | None = None
    for _attempt in range(1, max_retries + 1):
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
            return _validate_judge_output(result), None
        except Exception as exc:  # pragma: no cover - runtime dependent
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


def _ensure_output_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    output_cfg = cfg.get("output", {})
    out_dir = _resolve_path(str(output_cfg.get("dir", "./exam/output")))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = _resolve_path(str(output_cfg.get("summary_json", "./exam/output/skill_exam_summary.json")))
    raw_path = _resolve_path(str(output_cfg.get("raw_json", "./exam/output/skill_exam_raw.json")))
    overwrite = bool(output_cfg.get("overwrite", False))

    if not overwrite:
        collisions: list[Path] = []
        if summary_path.exists():
            collisions.append(summary_path)
        if raw_path.exists():
            collisions.append(raw_path)
        if collisions:
            raise FileExistsError(
                "Output files already exist. Set output.overwrite=true or change output paths: "
                + ", ".join([str(p) for p in collisions])
            )

    return summary_path, raw_path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_exam(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets = _parse_datasets(cfg)

    checkpoint = str(cfg.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("checkpoint is required in config")

    infer_cfg = cfg.get("inference", {})
    modes = _build_modes(cfg)

    corpus = load_skill_corpus(str(cfg.get("corpus_path", "")))

    # One client instance serves both question generation and judge calls.
    llm_client = LLMClient.from_env(
        model=None,
        max_tokens=max(
            int(cfg.get("question_generation", {}).get("max_tokens", 2048) or 2048),
            int(cfg.get("judging", {}).get("max_tokens", 1024) or 1024),
        ),
        temperature=0.0,
    )

    question_prompt = load_prompt_config(str(cfg.get("question_generation", {}).get("prompt_file", "")))
    judge_prompt = load_prompt_config(str(cfg.get("judging", {}).get("prompt_file", "")))
    q_limiter = QPMRateLimiter(int(cfg.get("question_generation", {}).get("qpm", 60) or 60))
    j_limiter = QPMRateLimiter(int(cfg.get("judging", {}).get("qpm", 60) or 60))

    model_args = _build_model_args(cfg)
    model_args_dict = vars(model_args).copy()
    meta = load_checkpoint_meta(model_args.checkpoint)

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    disable_parallel = bool(infer_cfg.get("disable_parallel", False))
    corpus_path = str(_resolve_path(str(cfg.get("corpus_path", ""))))

    parallel_pool = None
    if not disable_parallel and visible_gpu_count > 1:
        print(f"[exam] initializing parallel answer pool with {visible_gpu_count} workers")
        parallel_pool = _ParallelAnswerPool(
            num_workers=visible_gpu_count,
            model_args_dict=model_args_dict,
            meta=meta,
            corpus_path=corpus_path,
        )

    serial_model = None
    serial_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compress_k_list = infer_cfg.get("compress_k_list", [64])
    default_k = int(compress_k_list[0]) if compress_k_list else 64

    overall_stats: dict[str, dict[str, int]] = {
        mode["name"]: _stats_template() for mode in modes
    }
    summary_datasets: list[dict[str, Any]] = []
    raw_datasets: list[dict[str, Any]] = []
    total_selected_skills = 0

    print(f"[exam] datasets={datasets}, modes={[m['name'] for m in modes]}")

    for dataset in datasets:
        skill_ids = _select_skill_ids_for_dataset(corpus, dataset)
        if not skill_ids:
            print(f"[warn] no skills matched dataset prefix: {dataset}")

        total_selected_skills += len(skill_ids)

        dataset_stats: dict[str, dict[str, int]] = {
            mode["name"]: _stats_template() for mode in modes
        }
        dataset_summary_skills: list[dict[str, Any]] = []
        dataset_raw_skills: list[dict[str, Any]] = []

        print(f"[exam] dataset={dataset}, matched_skills={len(skill_ids)}")

        for idx, skill_id in enumerate(skill_ids, start=1):
            skill_row = corpus[skill_id]
            skill_name = str(skill_row.get("name", "")).strip() or skill_id
            print(f"[exam] [{dataset}] ({idx}/{len(skill_ids)}) skill={skill_id}")

            questions, q_error = generate_questions_for_skill(
                llm_client=llm_client,
                limiter=q_limiter,
                prompt_cfg=question_prompt,
                dataset=dataset,
                skill_id=skill_id,
                skill_row=skill_row,
                cfg=cfg,
            )

            skill_raw = {
                "dataset": dataset,
                "skill_id": skill_id,
                "skill_name": skill_name,
                "question_generation_error": q_error,
                "questions": [],
            }

            skill_stats: dict[str, dict[str, int]] = {
                mode["name"]: _stats_template() for mode in modes
            }

            if q_error:
                dataset_raw_skills.append(skill_raw)
                dataset_summary_skills.append(
                    {
                        "dataset": dataset,
                        "skill_id": skill_id,
                        "skill_name": skill_name,
                        "question_generation_error": q_error,
                        "metrics": {k: _finalize_stats(v) for k, v in skill_stats.items()},
                    }
                )
                continue

            use_parallel = (
                parallel_pool is not None
                and len(questions) > 1
            )

            if use_parallel:
                print(
                    f"[exam] [{dataset}] skill={skill_id} answer generation in parallel, workers={visible_gpu_count}"
                )
                question_rows = parallel_pool.run_skill(
                    dataset=dataset,
                    skill_id=skill_id,
                    questions=questions,
                    modes=modes,
                    infer_cfg=infer_cfg,
                    default_k=default_k,
                )
            else:
                if serial_model is None:
                    serial_model = build_model(model_args, meta).to(serial_device)
                    serial_model.eval()

                if disable_parallel:
                    print(f"[exam] [{dataset}] parallel disabled by config for skill={skill_id}")
                elif visible_gpu_count <= 1:
                    print(f"[exam] [{dataset}] single GPU/CPU fallback for skill={skill_id}")
                elif len(questions) <= 1:
                    print(f"[exam] [{dataset}] small question set fallback for skill={skill_id}")

                question_rows = _run_serial_answer_generation(
                    questions=questions,
                    model=serial_model,
                    corpus=corpus,
                    dataset=dataset,
                    skill_id=skill_id,
                    modes=modes,
                    infer_cfg=infer_cfg,
                    default_k=default_k,
                )

            for q_row in question_rows:
                for mode in modes:
                    mode_name = str(mode["name"])
                    skill_stats[mode_name]["total"] += 1
                    dataset_stats[mode_name]["total"] += 1
                    overall_stats[mode_name]["total"] += 1

                full_text_answer = str(q_row["answers"].get("full_text", {}).get("raw_output", ""))
                gt_valid = _is_effective_answer(full_text_answer)
                q_row["groundtruth_answer"] = full_text_answer
                q_row["groundtruth_valid"] = bool(gt_valid)

                if not gt_valid:
                    for mode in modes:
                        mode_name = str(mode["name"])
                        skill_stats[mode_name]["skip_count"] += 1
                        dataset_stats[mode_name]["skip_count"] += 1
                        overall_stats[mode_name]["skip_count"] += 1
                        q_row["judgements"][mode_name] = {
                            "is_correct": None,
                            "score": None,
                            "reason": "Skipped because full_text ground truth is empty or invalid.",
                            "error": None,
                        }
                    skill_raw["questions"].append(q_row)
                    continue

                # full_text is ground truth itself.
                skill_stats["full_text"]["judged_count"] += 1
                skill_stats["full_text"]["correct_count"] += 1
                dataset_stats["full_text"]["judged_count"] += 1
                dataset_stats["full_text"]["correct_count"] += 1
                overall_stats["full_text"]["judged_count"] += 1
                overall_stats["full_text"]["correct_count"] += 1
                q_row["judgements"]["full_text"] = {
                    "is_correct": True,
                    "score": 1.0,
                    "reason": "Ground truth mode.",
                    "error": None,
                }

                for mode in modes:
                    mode_name = str(mode["name"])
                    if mode_name == "full_text":
                        continue

                    candidate_answer = str(q_row["answers"].get(mode_name, {}).get("raw_output", ""))
                    candidate_error = q_row["answers"].get(mode_name, {}).get("error")
                    if candidate_error:
                        skill_stats[mode_name]["skip_count"] += 1
                        dataset_stats[mode_name]["skip_count"] += 1
                        overall_stats[mode_name]["skip_count"] += 1
                        q_row["judgements"][mode_name] = {
                            "is_correct": None,
                            "score": None,
                            "reason": "Skipped because candidate answer generation failed.",
                            "error": str(candidate_error),
                        }
                        continue

                    if not _is_effective_answer(candidate_answer):
                        # Effective but empty-like output is treated as judged incorrect.
                        skill_stats[mode_name]["judged_count"] += 1
                        dataset_stats[mode_name]["judged_count"] += 1
                        overall_stats[mode_name]["judged_count"] += 1
                        q_row["judgements"][mode_name] = {
                            "is_correct": False,
                            "score": 0.0,
                            "reason": "Candidate answer is empty/invalid.",
                            "error": None,
                        }
                        continue

                    judge_result, judge_error = judge_answer(
                        llm_client=llm_client,
                        limiter=j_limiter,
                        prompt_cfg=judge_prompt,
                        cfg=cfg,
                        dataset=dataset,
                        question=str(q_row.get("question", "")),
                        reference_answer=full_text_answer,
                        candidate_answer=candidate_answer,
                    )

                    if judge_error:
                        skill_stats[mode_name]["skip_count"] += 1
                        dataset_stats[mode_name]["skip_count"] += 1
                        overall_stats[mode_name]["skip_count"] += 1
                        q_row["judgements"][mode_name] = {
                            "is_correct": None,
                            "score": None,
                            "reason": "Skipped because judge model failed.",
                            "error": judge_error,
                        }
                        continue

                    assert judge_result is not None
                    skill_stats[mode_name]["judged_count"] += 1
                    dataset_stats[mode_name]["judged_count"] += 1
                    overall_stats[mode_name]["judged_count"] += 1
                    if bool(judge_result["is_correct"]):
                        skill_stats[mode_name]["correct_count"] += 1
                        dataset_stats[mode_name]["correct_count"] += 1
                        overall_stats[mode_name]["correct_count"] += 1
                    q_row["judgements"][mode_name] = {
                        "is_correct": bool(judge_result["is_correct"]),
                        "score": float(judge_result["score"]),
                        "reason": str(judge_result["reason"]),
                        "error": None,
                    }

                skill_raw["questions"].append(q_row)

            dataset_summary_skills.append(
                {
                    "dataset": dataset,
                    "skill_id": skill_id,
                    "skill_name": skill_name,
                    "question_generation_error": None,
                    "metrics": {k: _finalize_stats(v) for k, v in skill_stats.items()},
                }
            )
            dataset_raw_skills.append(skill_raw)

        summary_datasets.append(
            {
                "dataset": dataset,
                "n_skills": len(skill_ids),
                "per_skill": dataset_summary_skills,
                "overall": {k: _finalize_stats(v) for k, v in dataset_stats.items()},
            }
        )
        raw_datasets.append(
            {
                "dataset": dataset,
                "n_skills": len(skill_ids),
                "skills": dataset_raw_skills,
            }
        )

    if parallel_pool is not None:
        parallel_pool.stop()

    summary = {
        "exam_name": str(cfg.get("exam_name", "skill_exam")),
        "datasets": datasets,
        "checkpoint": str(cfg.get("checkpoint", "")),
        "corpus_path": str(cfg.get("corpus_path", "")),
        "mode_names": [m["name"] for m in modes],
        "n_datasets": len(datasets),
        "n_skills_total": total_selected_skills,
        "n_questions_per_skill": int(cfg.get("question_generation", {}).get("n_questions_per_skill", 0) or 0),
        "per_dataset": summary_datasets,
        "overall": {k: _finalize_stats(v) for k, v in overall_stats.items()},
    }

    raw_output = {
        "meta": {
            "exam_name": str(cfg.get("exam_name", "skill_exam")),
            "datasets": datasets,
            "checkpoint": str(cfg.get("checkpoint", "")),
            "corpus_path": str(cfg.get("corpus_path", "")),
            "mode_names": [m["name"] for m in modes],
            "n_datasets": len(datasets),
            "n_skills_total": total_selected_skills,
            "n_questions_per_skill": int(cfg.get("question_generation", {}).get("n_questions_per_skill", 0) or 0),
        },
        "datasets": raw_datasets,
    }

    return summary, raw_output


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = _load_json(cfg_path)

    summary_path, raw_path = _ensure_output_paths(cfg)
    summary, raw_output = run_exam(cfg)

    _write_json(summary_path, summary)
    _write_json(raw_path, raw_output)

    print(f"[exam] Summary written to: {summary_path}")
    print(f"[exam] Raw output written to: {raw_path}")


if __name__ == "__main__":
    main()
