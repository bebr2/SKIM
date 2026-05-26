"""ToolQA-specific exam runner with ReAct format.

This script runs exam for ToolQA dataset, using ReAct format for answering questions.
Key differences from run_skill_exam.py:
1. Uses ReAct format (Thought/Action/Observation loop) for answering
2. Integrates ToolEnvironment to execute tool actions
3. Dedicated to toolqa dataset only
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
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
from skillrag_vendor.toolqa import ToolEnvironment
from skillrag_vendor.toolqa.fewshots import TOOLQA_EXAMPLES
from skillrag_vendor.toolqa.react import ReActAgent


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
            "Run ToolQA skill benchmark exam with ReAct format: generate per-skill questions, "
            "answer with naive/full_text/compress-k modes using ReAct agent, and judge results."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exam/config.toolqa.example.json",
        help="Path to ToolQA exam config JSON",
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


def _resolve_toolqa_data_dir(cfg: dict[str, Any]) -> Path:
    toolqa_data_dir = str(cfg.get("toolqa_data_dir", "")).strip()
    if toolqa_data_dir:
        path = _resolve_path(toolqa_data_dir)
        if path.exists():
            return path
        raise FileNotFoundError(f"toolqa_data_dir does not exist: {path}")

    skillrag_root = str(cfg.get("skillrag_root", "")).strip()
    if skillrag_root:
        external_dir = _resolve_path(skillrag_root) / "data" / "external" / "toolqa"
        if external_dir.exists():
            return external_dir

    raise FileNotFoundError(
        "ToolQA data directory is required. Set --toolqa_data_dir in config."
    )


def load_skill_corpus(corpus_path: str) -> dict[str, dict[str, Any]]:
    path = _resolve_path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    raw = _load_json(path)
    corpus: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                corpus[str(key)] = value
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                sid = str(item.get("skill_id", "") or item.get("id", ""))
                if sid:
                    corpus[sid] = item
    return corpus


def load_prompt_config(prompt_file: str) -> dict[str, Any]:
    path = _resolve_path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt config file not found: {path}")
    return _load_json(path)


def _select_skill_ids_for_toolqa(corpus: dict[str, dict[str, Any]]) -> list[str]:
    """Select all skills that belong to toolqa dataset."""
    matched: list[str] = []
    for sid in sorted(corpus.keys()):
        if sid.startswith("toolqa_"):
            matched.append(sid)
    return matched


def _extract_skill_payload(skill_row: dict[str, Any]) -> dict[str, Any]:
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
        if key in skill_row:
            out[key] = skill_row[key]
    if "skill_id" not in out:
        sid = str(skill_row.get("skill_id", "") or skill_row.get("id", ""))
        out["skill_id"] = sid
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
        dataset="toolqa",
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
        except Exception as exc:
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


def _load_toolqa_examples(skillrag_root: str) -> str:
    """Load ToolQA few-shot examples, preferring external SkillRAG repo."""
    if skillrag_root.strip():
        ext_src = Path(skillrag_root) / "src"
        if ext_src.exists() and str(ext_src) not in sys.path:
            sys.path.insert(0, str(ext_src))

        try:
            module = importlib.import_module("skillrag.toolqa.fewshots")
            examples = getattr(module, "TOOLQA_EXAMPLES", "")
            if isinstance(examples, str) and examples.strip():
                return examples
        except Exception:
            pass

    return TOOLQA_EXAMPLES


def _strip_prompt_echo(generated: str, step_n: int | None = None) -> str:
    """Strip echoed prompt from generation output."""
    text = generated.strip()
    if step_n is None:
        return text

    marker = f"Thought {step_n}:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text


def _run_toolqa_react_answer(
    model,
    corpus: dict[str, dict[str, Any]],
    question: str,
    skill_id: str,
    skill_mode: str,
    k: int,
    tool_env: ToolEnvironment,
    examples: str,
    infer_cfg: dict[str, Any],
) -> tuple[str, str, int, bool, str | None]:
    """Run ReAct agent for answering a ToolQA question.

    Returns:
        (answer, scratchpad, n_steps, finished, error)
    """
    use_skill = skill_mode in ("full_text", "compress")
    skill_ids = [skill_id] if use_skill else []
    skills: list[str] = []
    for sid in skill_ids:
        row = corpus.get(sid, {})
        content = row.get("content", "") if isinstance(row, dict) else ""
        if isinstance(content, str) and content.strip():
            skills.append(content)

    tool_env.reset()

    max_length = int(infer_cfg.get("max_length", 4096) or 4096)
    max_new_tokens = int(infer_cfg.get("toolqa_step_tokens", 512) or 512)
    do_sample = bool(infer_cfg.get("do_sample", False))
    temperature = float(infer_cfg.get("temperature", 0.7) or 0.7)
    top_p = float(infer_cfg.get("top_p", 0.95) or 0.95)
    max_steps = int(infer_cfg.get("toolqa_max_steps", 20) or 20)

    def _model_generate(
        system_text: str,
        user_text: str,
        max_tokens: int,
        step_n: int,
        stop_token: str | None,
    ) -> str:
        generated, _input_embeds = generate_with_skills(
            model=model,
            corpus=corpus,
            system_text=system_text,
            user_text=user_text,
            skill_mode=skill_mode,
            k=k,
            max_length=max_length,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        text = _strip_prompt_echo(generated, step_n=step_n)
        if stop_token and stop_token in text:
            text = text.split(stop_token, 1)[0]
        return text

    react_method = skill_mode if skill_ids else "naive"
    try:
        agent = ReActAgent(
            question=question,
            tools=tool_env,
            model_generate=_model_generate,
            examples=examples,
            max_steps=max_steps,
            max_tokens=max_new_tokens,
            method=react_method,
            skills=skills,
            skill_ids=skill_ids,
            skill_mode=skill_mode,
        )
        agent.run()
        return agent.answer, agent.scratchpad, max(0, agent.step_n - 1), bool(agent.finished), None
    except Exception as exc:
        return "", "", 0, False, str(exc)


def _build_modes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    infer_cfg = cfg.get("inference", {})
    modes: list[dict[str, Any]] = []

    do_naive = bool(infer_cfg.get("do_naive", True))
    do_full_text = bool(infer_cfg.get("do_full_text", True))
    compress_k_list = infer_cfg.get("compress_k_list", [64])

    if do_naive:
        modes.append({
            "name": "naive",
            "skill_mode": "naive",
            "inject_skills": False,
            "k": None,
        })

    if do_full_text:
        modes.append({
            "name": "full_text",
            "skill_mode": "full_text",
            "inject_skills": True,
            "k": None,
        })

    for k in compress_k_list:
        modes.append({
            "name": f"compress_k{k}",
            "skill_mode": "compress",
            "inject_skills": True,
            "k": int(k),
        })

    if not modes:
        raise ValueError("No answer modes configured")

    has_full_text = any(m["name"] == "full_text" for m in modes)
    if not has_full_text:
        raise ValueError("inference.do_full_text must be true (full_text is required as ground truth)")

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
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> tuple[dict[str, Any] | None, str | None]:
    j_cfg = cfg.get("judging", {})
    user_prompt = str(prompt_cfg["user_prompt_template"]).format(
        dataset="toolqa",
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
        except Exception as exc:
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
    out_dir = _resolve_path(str(output_cfg.get("dir", "./exam/output/toolqa")))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = _resolve_path(str(output_cfg.get("summary_json", "./exam/output/toolqa/toolqa_exam_summary.json")))
    raw_path = _resolve_path(str(output_cfg.get("raw_json", "./exam/output/toolqa/toolqa_exam_raw.json")))
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


def _build_model_args(cfg: dict[str, Any]) -> argparse.Namespace:
    infer_cfg = cfg.get("inference", {})
    fallback = cfg.get("model_fallback", {})

    args = argparse.Namespace()
    args.checkpoint = str(_resolve_path(str(cfg.get("checkpoint", ""))))
    args.skill_mode = "compress"
    args.k = int(infer_cfg.get("k", 64) or 64)
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

    args.skill_qa_llm_train_mode = str(fallback.get("skill_qa_llm_train_mode", "auto"))
    args.skill_qa_lora_r = int(fallback.get("skill_qa_lora_r", 16) or 16)
    args.skill_qa_lora_alpha = int(fallback.get("skill_qa_lora_alpha", 32) or 32)
    args.skill_qa_lora_dropout = float(fallback.get("skill_qa_lora_dropout", 0.05) or 0.05)
    args.skill_qa_lora_target_modules = str(fallback.get("skill_qa_lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"))
    args.skill_qa_lora_bias = str(fallback.get("skill_qa_lora_bias", "none"))
    args.skill_qa_lora_task_type = str(fallback.get("skill_qa_lora_task_type", "CAUSAL_LM"))

    return args


def run_toolqa_exam(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = str(cfg.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("checkpoint is required in config")

    infer_cfg = cfg.get("inference", {})
    modes = _build_modes(cfg)

    corpus = load_skill_corpus(str(cfg.get("corpus_path", "")))
    skill_ids = _select_skill_ids_for_toolqa(corpus)

    if not skill_ids:
        raise ValueError("No toolqa skills found in corpus")

    toolqa_data_dir = _resolve_toolqa_data_dir(cfg)
    tool_env = ToolEnvironment(toolqa_data_dir)
    examples = _load_toolqa_examples(str(cfg.get("skillrag_root", "")))

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
    meta = load_checkpoint_meta(model_args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_args, meta).to(device)
    model.eval()

    compress_k_list = infer_cfg.get("compress_k_list", [64])
    default_k = int(compress_k_list[0]) if compress_k_list else 64

    overall_stats: dict[str, dict[str, int]] = {
        mode["name"]: _stats_template() for mode in modes
    }
    summary_skills: list[dict[str, Any]] = []
    raw_skills: list[dict[str, Any]] = []

    print(f"[toolqa_exam] matched_skills={len(skill_ids)}, modes={[m['name'] for m in modes]}")

    for idx, skill_id in enumerate(skill_ids, start=1):
        skill_row = corpus[skill_id]
        skill_name = str(skill_row.get("name", "")).strip() or skill_id
        print(f"[toolqa_exam] ({idx}/{len(skill_ids)}) skill={skill_id}")

        questions, q_error = generate_questions_for_skill(
            llm_client=llm_client,
            limiter=q_limiter,
            prompt_cfg=question_prompt,
            skill_id=skill_id,
            skill_row=skill_row,
            cfg=cfg,
        )

        skill_raw = {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "question_generation_error": q_error,
            "questions": [],
        }

        skill_stats: dict[str, dict[str, int]] = {
            mode["name"]: _stats_template() for mode in modes
        }

        if q_error:
            raw_skills.append(skill_raw)
            summary_skills.append(
                {
                    "skill_id": skill_id,
                    "skill_name": skill_name,
                    "question_generation_error": q_error,
                    "metrics": {k: _finalize_stats(v) for k, v in skill_stats.items()},
                }
            )
            continue

        for q_idx, question in enumerate(questions, start=1):
            q_row: dict[str, Any] = {
                "question_id": q_idx,
                "question": question,
                "answers": {},
                "judgements": {},
                "groundtruth_mode": "full_text",
                "groundtruth_answer": "",
                "groundtruth_valid": False,
            }

            for mode in modes:
                mode_name = str(mode["name"])
                skill_mode = str(mode["skill_mode"])
                use_skill = bool(mode["inject_skills"])
                k_value = int(mode["k"]) if mode.get("k") is not None else int(default_k)

                print(f"[toolqa_exam] [{skill_id}] Q{q_idx} mode={mode_name} running ReAct...")
                started = time.time()

                answer, scratchpad, n_steps, finished, err = _run_toolqa_react_answer(
                    model=model,
                    corpus=corpus,
                    question=question,
                    skill_id=skill_id,
                    skill_mode=skill_mode,
                    k=k_value,
                    tool_env=tool_env,
                    examples=examples,
                    infer_cfg=infer_cfg,
                )

                elapsed = time.time() - started
                q_row["answers"][mode_name] = {
                    "raw_output": answer,
                    "scratchpad": scratchpad,
                    "n_steps": n_steps,
                    "finished": finished,
                    "elapsed_seconds": elapsed,
                    "error": err,
                }

            for mode in modes:
                mode_name = str(mode["name"])
                skill_stats[mode_name]["total"] += 1
                overall_stats[mode_name]["total"] += 1

            full_text_answer = str(q_row["answers"].get("full_text", {}).get("raw_output", ""))
            gt_valid = _is_effective_answer(full_text_answer)
            q_row["groundtruth_answer"] = full_text_answer
            q_row["groundtruth_valid"] = bool(gt_valid)

            if not gt_valid:
                for mode in modes:
                    mode_name = str(mode["name"])
                    skill_stats[mode_name]["skip_count"] += 1
                    overall_stats[mode_name]["skip_count"] += 1
                    q_row["judgements"][mode_name] = {
                        "is_correct": None,
                        "score": None,
                        "reason": "Skipped because full_text ground truth is empty or invalid.",
                        "error": None,
                    }
                skill_raw["questions"].append(q_row)
                continue

            skill_stats["full_text"]["judged_count"] += 1
            skill_stats["full_text"]["correct_count"] += 1
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
                    overall_stats[mode_name]["skip_count"] += 1
                    q_row["judgements"][mode_name] = {
                        "is_correct": None,
                        "score": None,
                        "reason": "Skipped because candidate answer generation failed.",
                        "error": str(candidate_error),
                    }
                    continue

                if not _is_effective_answer(candidate_answer):
                    skill_stats[mode_name]["judged_count"] += 1
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
                    question=question,
                    reference_answer=full_text_answer,
                    candidate_answer=candidate_answer,
                )

                if judge_error:
                    skill_stats[mode_name]["skip_count"] += 1
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
                overall_stats[mode_name]["judged_count"] += 1
                if bool(judge_result["is_correct"]):
                    skill_stats[mode_name]["correct_count"] += 1
                    overall_stats[mode_name]["correct_count"] += 1
                q_row["judgements"][mode_name] = {
                    "is_correct": bool(judge_result["is_correct"]),
                    "score": float(judge_result["score"]),
                    "reason": str(judge_result["reason"]),
                    "error": None,
                }

            skill_raw["questions"].append(q_row)

        summary_skills.append(
            {
                "skill_id": skill_id,
                "skill_name": skill_name,
                "question_generation_error": None,
                "metrics": {k: _finalize_stats(v) for k, v in skill_stats.items()},
            }
        )
        raw_skills.append(skill_raw)

    summary = {
        "exam_name": str(cfg.get("exam_name", "toolqa_react_exam")),
        "dataset": "toolqa",
        "checkpoint": str(cfg.get("checkpoint", "")),
        "corpus_path": str(cfg.get("corpus_path", "")),
        "toolqa_data_dir": str(toolqa_data_dir),
        "mode_names": [m["name"] for m in modes],
        "n_skills": len(skill_ids),
        "overall": {k: _finalize_stats(v) for k, v in overall_stats.items()},
        "per_skill": summary_skills,
    }

    raw = {
        "exam_name": str(cfg.get("exam_name", "toolqa_react_exam")),
        "dataset": "toolqa",
        "n_skills": len(skill_ids),
        "skills": raw_skills,
    }

    return summary, raw


def main() -> None:
    args = parse_args()
    config_path = _resolve_path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_json(config_path)
    summary_path, raw_path = _ensure_output_paths(cfg)

    print(f"[toolqa_exam] config={config_path}")
    print(f"[toolqa_exam] summary_path={summary_path}")
    print(f"[toolqa_exam] raw_path={raw_path}")

    summary, raw = run_toolqa_exam(cfg)

    _write_json(summary_path, summary)
    _write_json(raw_path, raw)

    print(f"[toolqa_exam] Done. Summary: {summary_path}")
    print(f"[toolqa_exam] Raw: {raw_path}")

    overall = summary.get("overall", {})
    for mode_name, stats in overall.items():
        acc = stats.get("accuracy")
        judged = stats.get("judged_count", 0)
        correct = stats.get("correct_count", 0)
        if acc is not None:
            print(f"[toolqa_exam] mode={mode_name}: accuracy={acc:.4f} ({correct}/{judged})")
        else:
            print(f"[toolqa_exam] mode={mode_name}: no judged items")


if __name__ == "__main__":
    main()