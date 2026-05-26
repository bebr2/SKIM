from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from queue import Empty
from typing import Any

import torch

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

from models.skim_model import SKIMModel

MAX_TOTAL_TOKENS = 40960


class TokenLimitExceededError(RuntimeError):
    """Raised when input tokens plus max_new_tokens exceed the configured limit."""

    pass
from utils.prompts import (
    build_chat_prompt_embeds_from_system_and_user_segments,
    build_chat_prompt_embeds_from_user_segments,
    build_chat_template_system_user_shell_ids,
    build_chat_template_user_shell_ids,
)

SKILL_TAG_PATTERN = re.compile(r"<skill>\s*(.*?)\s*</skill>", re.IGNORECASE | re.DOTALL)
DEFAULT_SKILL_QA_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SKIM skill-conditioned inference")
    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint dir")
    parser.add_argument("--skill_corpus_path", type=str, default="", help="skill corpus json path")

    parser.add_argument("--input_json", type=str, default="", help="json file containing a sample conversation")
    parser.add_argument("--instance_id", type=str, default="", help="select sample by instance_id for dataset-style json")
    parser.add_argument("--sample_index", type=int, default=0, help="legacy index option; json mode now processes all matching samples")
    parser.add_argument("--round_index", type=int, default=0, help="select round in message_rounds format")
    parser.add_argument("--result_path", type=str, default="", help="path to save aggregated json results")
    parser.add_argument("--parse_only", action="store_true", help="only parse input_json and save parsed cases, skip model loading")
    parser.add_argument("--system_prompt", type=str, default="", help="fallback system prompt")
    parser.add_argument("--user_prompt", type=str, default="", help="fallback user prompt")
    parser.add_argument("--assistant_ref", type=str, default="", help="optional reference assistant text")
    parser.add_argument(
        "--skill_mode",
        type=str,
        default="compress",
        choices=["naive", "full_text", "compress"],
        help="how to handle <skill> tags in user content",
    )

    parser.add_argument("--k", type=int, default=64, help="soft token length per skill")
    parser.add_argument("--max_length", type=int, default=4096, help="max skill text token length")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="generation length")

    parser.add_argument("--do_sample", action="store_true", help="enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="sampling top-p")

    parser.add_argument("--compressor_model", type=str, default="", help="fallback compressor model")
    parser.add_argument("--llm_model", type=str, default="", help="fallback llm model")
    parser.add_argument("--projector_layers", type=int, default=3, help="fallback projector layers")
    parser.add_argument("--projector_hidden", type=int, default=4096, help="fallback projector hidden")
    parser.add_argument("--max_q", type=int, default=256, help="fallback max q")
    parser.add_argument(
        "--skill_qa_llm_train_mode",
        type=str,
        default="auto",
        choices=["auto", "false", "true", "lora"],
        help="override llm_train_mode from checkpoint meta",
    )
    parser.add_argument("--skill_qa_lora_r", type=int, default=16, help="fallback LoRA rank")
    parser.add_argument("--skill_qa_lora_alpha", type=int, default=32, help="fallback LoRA alpha")
    parser.add_argument("--skill_qa_lora_dropout", type=float, default=0.05, help="fallback LoRA dropout")
    parser.add_argument(
        "--skill_qa_lora_target_modules",
        type=str,
        default=",".join(DEFAULT_SKILL_QA_LORA_TARGET_MODULES),
        help="fallback comma-separated LoRA target modules",
    )
    parser.add_argument("--skill_qa_lora_bias", type=str, default="none", help="fallback LoRA bias")
    parser.add_argument("--skill_qa_lora_task_type", type=str, default="CAUSAL_LM", help="fallback LoRA task type")
    parser.add_argument(
        "--skill_qa_lora_modules_to_save",
        type=str,
        default="",
        help="fallback comma-separated LoRA modules_to_save",
    )
    parser.add_argument(
        "--skill_qa_lora_use_rslora",
        type=str,
        default="false",
        help="fallback LoRA use_rslora (true/false)",
    )
    parser.add_argument("--disable_parallel", action="store_true", help="disable multi-gpu sharded inference")
    return parser.parse_args()


def load_checkpoint_meta(checkpoint_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(checkpoint_dir, "skim_checkpoint.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
    return default


def _csv_to_list(raw: Any) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _resolve_skill_qa_lora_config(meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw_lora = meta.get("skill_qa_lora") if isinstance(meta.get("skill_qa_lora"), dict) else {}

    target_modules = raw_lora.get("target_modules") if isinstance(raw_lora, dict) else None
    if not isinstance(target_modules, list) or not target_modules:
        target_modules = _csv_to_list(getattr(args, "skill_qa_lora_target_modules", ""))
    if not target_modules:
        target_modules = list(DEFAULT_SKILL_QA_LORA_TARGET_MODULES)

    modules_to_save = raw_lora.get("modules_to_save") if isinstance(raw_lora, dict) else None
    if not isinstance(modules_to_save, list):
        modules_to_save = _csv_to_list(getattr(args, "skill_qa_lora_modules_to_save", ""))

    return {
        "r": int(raw_lora.get("r", getattr(args, "skill_qa_lora_r", 16))),
        "alpha": int(raw_lora.get("alpha", getattr(args, "skill_qa_lora_alpha", 32))),
        "dropout": float(raw_lora.get("dropout", getattr(args, "skill_qa_lora_dropout", 0.05))),
        "target_modules": [str(x).strip() for x in target_modules if str(x).strip()],
        "bias": str(raw_lora.get("bias", getattr(args, "skill_qa_lora_bias", "none"))),
        "task_type": str(raw_lora.get("task_type", getattr(args, "skill_qa_lora_task_type", "CAUSAL_LM"))),
        "modules_to_save": [str(x).strip() for x in modules_to_save if str(x).strip()],
        "use_rslora": _to_bool(
            raw_lora.get("use_rslora", getattr(args, "skill_qa_lora_use_rslora", False)),
            default=False,
        ),
    }


def build_model(
    args: argparse.Namespace,
    meta: dict[str, Any],
    llm_device_map: str | dict | None = None,
    llm_max_memory: dict | None = None,
    torch_dtype: torch.dtype | None = None,
    skip_compressor: bool = False,
) -> SKIMModel:
    compressor_model = meta.get("compressor_model") or args.compressor_model
    llm_model = meta.get("llm_model") or args.llm_model
    if not compressor_model or not llm_model:
        raise ValueError("Model names are missing. Set --compressor_model and --llm_model or provide skim_checkpoint.json")

    model = SKIMModel(
        compressor_model=compressor_model,
        llm_model=llm_model,
        max_q=int(meta.get("max_q", args.max_q)),
        projector_layers=int(meta.get("projector_layers", args.projector_layers)),
        projector_hidden=int(meta.get("projector_hidden", args.projector_hidden)),
        llm_device_map=llm_device_map,
        llm_max_memory=llm_max_memory,
        torch_dtype=torch_dtype,
        skip_compressor=skip_compressor,
    )

    llm_train_mode = str(meta.get("llm_train_mode", getattr(args, "skill_qa_llm_train_mode", "auto"))).strip().lower()
    if llm_train_mode == "lora":
        lora_cfg = _resolve_skill_qa_lora_config(meta=meta, args=args)
        model.enable_llm_lora(
            r=lora_cfg["r"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
            modules_to_save=lora_cfg["modules_to_save"],
            use_rslora=lora_cfg["use_rslora"],
        )

    load_compressor = not skip_compressor
    model.load_modular_checkpoint(args.checkpoint, load_llm=True, load_compressor=load_compressor, strict=False)
    return model


def _collect_skill_items(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "skill_id" in obj:
            out.append(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _collect_skill_items(value, out)
        return

    if isinstance(obj, list):
        for item in obj:
            _collect_skill_items(item, out)


def load_skill_corpus(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skill corpus file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items: list[dict[str, Any]] = []
    _collect_skill_items(payload, items)

    mapping: dict[str, dict[str, Any]] = {}
    for item in items:
        skill_id = item.get("skill_id")
        if skill_id is None:
            continue
        key = str(skill_id).strip()
        if not key:
            continue
        if key not in mapping:
            mapping[key] = item

    if not mapping:
        raise ValueError(f"No skill items with 'skill_id' found in: {path}")
    return mapping


def resolve_skill_text(skill_spec: str, corpus: dict[str, dict[str, Any]], cache: dict[str, str]) -> str:
    cached = cache.get(skill_spec)
    if cached is not None:
        return cached

    spec = skill_spec.strip()
    if not spec:
        raise ValueError("Empty skill spec in <skill>")

    if "." in spec:
        skill_id, field = spec.split(".", 1)
    else:
        skill_id, field = spec, "content"

    row = corpus.get(skill_id)
    if row is None:
        raise KeyError(f"skill_id '{skill_id}' is not found in skill corpus")

    value = row.get(field)
    if value is None and field != "content":
        value = row.get("content")
    if value is None:
        raise KeyError(f"Field '{field}' and fallback 'content' are both missing for skill_id '{skill_id}'")

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    if not text.strip():
        print(f"[resolve_skill_text] WARNING: skill_id '{skill_id}' has empty content")
    elif len(text.strip()) < 10:
        print(f"[resolve_skill_text] WARNING: skill_id '{skill_id}' has very short content: '{text[:50]}...'")

    cache[skill_spec] = text
    return text


def parse_user_segments(user_text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    cursor = 0

    for match in SKILL_TAG_PATTERN.finditer(user_text):
        if match.start() > cursor:
            segments.append(("text", user_text[cursor : match.start()]))

        skill_spec = match.group(1).strip()
        if skill_spec:
            segments.append(("skill", skill_spec))
        else:
            segments.append(("text", match.group(0)))
        cursor = match.end()

    if cursor < len(user_text):
        segments.append(("text", user_text[cursor:]))

    if not segments:
        segments.append(("text", user_text))
    return segments


def build_naive_user_text(user_text: str) -> str:
    without_skill = SKILL_TAG_PATTERN.sub("", user_text)
    without_prefix = re.sub(r"^\s*Relevant Skill:\s*", "", without_skill, flags=re.IGNORECASE)
    return without_prefix.strip()


def _looks_like_messages(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    return isinstance(obj[0], dict) and "role" in obj[0] and "content" in obj[0]


def _extract_messages(obj: Any) -> list[dict[str, Any]] | None:
    if isinstance(obj, dict):
        for key in ("messages", "message_rounds", "conversation", "conversations", "dialogue"):
            candidate = obj.get(key)
            extracted = _extract_messages(candidate)
            if extracted is not None:
                return extracted

        if "role" in obj and "content" in obj:
            return [obj]

    if isinstance(obj, list):
        if _looks_like_messages(obj):
            return obj
        if len(obj) == 1 and _looks_like_messages(obj[0]):
            return obj[0]

        for item in obj:
            extracted = _extract_messages(item)
            if extracted is not None:
                return extracted
    return None


def _messages_from_record(record: dict[str, Any], round_index: int) -> list[dict[str, Any]] | None:
    messages = record.get("messages")
    extracted_messages = _extract_messages(messages)
    if extracted_messages is not None:
        return extracted_messages

    rounds = record.get("message_rounds")
    if isinstance(rounds, list) and rounds:
        idx = max(0, min(int(round_index), len(rounds) - 1))
        round_messages = rounds[idx]
        extracted_round = _extract_messages(round_messages)
        if extracted_round is not None:
            return extracted_round

    return None


def _collect_conversations(obj: Any, out: list[list[dict[str, Any]]]) -> None:
    messages = _extract_messages(obj)
    if messages is not None:
        out.append(messages)
        return

    if isinstance(obj, list):
        for item in obj:
            _collect_conversations(item, out)
        return

    if isinstance(obj, dict):
        for key in ("data", "items", "records", "samples"):
            if key in obj:
                _collect_conversations(obj[key], out)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                chunks.append(str(part.get("text", "")))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    return str(content)


def _extract_triplet_from_messages(messages: list[dict[str, Any]]) -> tuple[str, str, str] | None:
    system_parts: list[str] = []
    user_text = ""
    assistant_ref = ""
    seen_user = False

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = _content_to_text(message.get("content", ""))
        if role == "system" and not assistant_ref:
            system_parts.append(content)
            continue
        if role == "user" and not user_text:
            user_text = content
            seen_user = True
            continue
        if role == "assistant" and seen_user:
            assistant_ref = content
            break

    if not user_text.strip():
        return None
    return "\n\n".join(x for x in system_parts if x.strip()), user_text, assistant_ref


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.input_json:
        if not args.user_prompt.strip():
            raise ValueError("Provide --user_prompt or --input_json")
        return [
            {
                "system_text": args.system_prompt,
                "user_text": args.user_prompt,
                "assistant_ref": args.assistant_ref,
                "meta": {"source": "cli"},
            }
        ]

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cases: list[dict[str, Any]] = []

    # Dedicated support for dataset-style records such as golden_skill_messages.json.
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        key = args.instance_id.strip()
        for i, record in enumerate(payload):
            if key and str(record.get("instance_id", "")).strip() != key:
                continue

            messages = _messages_from_record(record, round_index=args.round_index)
            if messages is not None:
                triplet = _extract_triplet_from_messages(messages)
                if triplet is not None:
                    system_text, user_text, assistant_ref = triplet
                    cases.append(
                        {
                            "system_text": system_text,
                            "user_text": user_text,
                            "assistant_ref": assistant_ref,
                            "meta": {
                                "source": "record",
                                "record_index": i,
                                "round_index": int(args.round_index),
                                "instance_id": str(record.get("instance_id", "")),
                                "dataset": str(record.get("dataset", "")),
                                "model": str(record.get("model", "")),
                            },
                        }
                    )
                    continue

            if "user" in record:
                user_text = str(record.get("user", ""))
                if not user_text.strip():
                    continue
                cases.append(
                    {
                        "system_text": str(record.get("system", "")),
                        "user_text": user_text,
                        "assistant_ref": str(record.get("assistant", "")),
                        "meta": {
                            "source": "record_user",
                            "record_index": i,
                            "instance_id": str(record.get("instance_id", "")),
                        },
                    }
                )

        if key and not cases:
            raise ValueError(f"instance_id not found or unparsable in input_json: {key}")
        if cases:
            return cases

    if isinstance(payload, dict) and "user" in payload:
        system_text = str(payload.get("system", ""))
        user_text = str(payload.get("user", ""))
        assistant_ref = str(payload.get("assistant", ""))
        if not user_text.strip():
            raise ValueError("input_json has empty 'user'")
        return [
            {
                "system_text": system_text,
                "user_text": user_text,
                "assistant_ref": assistant_ref,
                "meta": {"source": "dict_user"},
            }
        ]

    conversations: list[list[dict[str, Any]]] = []
    _collect_conversations(payload, conversations)
    if not conversations:
        raise ValueError("Cannot parse messages from input_json")

    for i, messages in enumerate(conversations):
        triplet = _extract_triplet_from_messages(messages)
        if triplet is not None:
            system_text, user_text, assistant_ref = triplet
            cases.append(
                {
                    "system_text": system_text,
                    "user_text": user_text,
                    "assistant_ref": assistant_ref,
                    "meta": {"source": "conversation", "conversation_index": i},
                }
            )

    if cases:
        return cases

    raise ValueError("No valid user message found in input_json")


def _default_result_path(input_json: str) -> str:
    if input_json:
        base, _ext = os.path.splitext(input_json)
        return f"{base}.result.json"
    return "./skim_inference.result.json"


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < float("inf")):
        return "--:--"
    total = int(seconds)
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class _SimpleProgressBar:
    def __init__(self, total: int, desc: str = "Infer") -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.count = 0
        self.start_time = time.time()
        self._last_render = 0.0
        self._render(force=True)

    def update(self, n: int = 1) -> None:
        self.count = min(self.total, self.count + int(n))
        now = time.time()
        if now - self._last_render >= 0.2 or self.count >= self.total:
            self._render(force=self.count >= self.total)

    def _render(self, force: bool = False) -> None:
        width = 30
        ratio = self.count / self.total
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)

        elapsed = time.time() - self.start_time
        speed = self.count / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.count
        eta = remaining / speed if speed > 0 else float("inf")

        text = (
            f"\r{self.desc} [{bar}] {self.count}/{self.total} "
            f"ETA {_format_duration(eta)}"
        )
        print(text, end="", flush=True)
        self._last_render = time.time()

        if force and self.count >= self.total:
            print()

    def close(self) -> None:
        if self.count < self.total:
            self.count = self.total
            self._render(force=True)


def _create_progress_bar(total: int, desc: str = "Infer") -> Any:
    if _tqdm is not None:
        return _tqdm(total=total, desc=desc, unit="case", dynamic_ncols=True, mininterval=0.2)
    return _SimpleProgressBar(total=total, desc=desc)


def _split_indexed_cases_evenly(
    indexed_cases: list[tuple[int, dict[str, Any]]],
    num_parts: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    num_parts = max(1, int(num_parts))
    shards: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(num_parts)]
    for idx, item in enumerate(indexed_cases):
        shards[idx % num_parts].append(item)
    return shards


def _run_single_case(
    model: SKIMModel,
    corpus: dict[str, dict[str, Any]] | None,
    case_index: int,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system_text = str(case.get("system_text", ""))
    user_text = str(case.get("user_text", ""))
    assistant_ref = str(case.get("assistant_ref", ""))
    meta_info = case.get("meta", {})

    generated, input_embeds = generate_with_skills(
        model=model,
        corpus=corpus,
        system_text=system_text,
        user_text=user_text,
        skill_mode=args.skill_mode,
        k=args.k,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    result_item: dict[str, Any] = {
        "case_index": case_index,
        "meta": meta_info,
        "skill_mode": args.skill_mode,
        "system": system_text,
        "user": user_text,
        "generated": generated,
    }

    if assistant_ref.strip():
        loss_value = compute_teacher_forcing_loss(model, input_embeds, assistant_ref)
        result_item["assistant_ref"] = assistant_ref
        result_item["teacher_forcing_loss"] = loss_value

    return result_item


def _worker_infer_shard(
    worker_id: int,
    gpu_id: int,
    shard: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    meta: dict[str, Any],
    progress_queue: Any,
    result_queue: Any,
) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

        model = build_model(args, meta).to(device)
        model.eval()
        corpus = load_skill_corpus(args.skill_corpus_path) if args.skill_mode in {"compress", "full_text"} else None

        shard_results: list[dict[str, Any]] = []
        for case_index, case in shard:
            shard_results.append(_run_single_case(model=model, corpus=corpus, case_index=case_index, case=case, args=args))
            progress_queue.put(1)

        result_queue.put({"worker_id": worker_id, "results": shard_results})
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": worker_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_parallel_inference(
    indexed_cases: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    meta: dict[str, Any],
    visible_gpu_count: int,
) -> list[dict[str, Any]]:
    shards = _split_indexed_cases_evenly(indexed_cases, visible_gpu_count)
    worker_specs = [(gpu_id, shard) for gpu_id, shard in enumerate(shards) if shard]
    if not worker_specs:
        return []

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes: list[Any] = []

    for worker_id, (gpu_id, shard) in enumerate(worker_specs):
        process = ctx.Process(
            target=_worker_infer_shard,
            args=(worker_id, gpu_id, shard, args, meta, progress_queue, result_queue),
        )
        process.start()
        processes.append(process)

    bar = _create_progress_bar(total=len(indexed_cases), desc="Infer")
    completed_workers = 0
    merged_results: list[dict[str, Any]] = []
    worker_errors: list[dict[str, Any]] = []

    while completed_workers < len(processes):
        try:
            progress_queue.get(timeout=0.2)
            bar.update(1)
        except Empty:
            pass

        try:
            payload = result_queue.get_nowait()
        except Empty:
            continue

        if "error" in payload:
            worker_errors.append(payload)
        else:
            merged_results.extend(payload.get("results", []))
        completed_workers += 1

    while True:
        try:
            progress_queue.get_nowait()
            bar.update(1)
        except Empty:
            break

    bar.close()

    for process in processes:
        process.join()

    abnormal_exit = [f"pid={p.pid}, exitcode={p.exitcode}" for p in processes if p.exitcode not in (0, None)]
    if worker_errors or abnormal_exit:
        details: list[str] = []
        for err in worker_errors:
            details.append(
                f"worker={err.get('worker_id')}, error={err.get('error')}\n{err.get('traceback', '')}"
            )
        details.extend(abnormal_exit)
        raise RuntimeError("Parallel inference failed:\n" + "\n".join(details))

    if len(merged_results) != len(indexed_cases):
        raise RuntimeError(
            f"Result count mismatch: got {len(merged_results)}, expected {len(indexed_cases)}"
        )

    return merged_results


def generate_with_skills(
    model: SKIMModel,
    corpus: dict[str, dict[str, Any]] | None,
    system_text: str,
    user_text: str,
    skill_mode: str,
    k: int,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    bad_words_ids: list[list[int]] | None = None,
    prefix_token_ids: list[int] | None = None,
    skill_pad_repeat: bool = False,
) -> tuple[str, torch.Tensor]:
    func_start = time.time()
    mode = skill_mode.strip().lower()
    if mode not in {"naive", "full_text", "compress"}:
        raise ValueError(f"Unsupported skill_mode: {skill_mode}")

    # print(f"system_text={system_text}...")

    # print(f"user_text={user_text}...")

    cache: dict[str, str] = {}
    parse_start = time.time()
    parsed = parse_user_segments(user_text)
    parse_time = time.time() - parse_start

    user_segments: list[str | torch.Tensor] = []
    emb_layer = model.llm.get_input_embeddings()
    llm_device = emb_layer.weight.device if hasattr(emb_layer, 'weight') else model.llm.device
    compressor_device = None
    projector_device = None
    if mode == "compress":
        compressor_device = model.compressor.backbone.device if hasattr(model.compressor.backbone, 'device') else model.compressor.learnable_q.device
        projector_device = model.projector.net[0].weight.device if hasattr(model.projector, 'net') else llm_device

    compress_start = 0
    compress_time = 0
    if mode == "naive":
        naive_user = build_naive_user_text(user_text)
        user_segments = [naive_user if naive_user else " "]
    elif mode == "full_text":
        if corpus is None:
            raise ValueError("skill_mode=full_text requires skill corpus")
        for kind, value in parsed:
            if kind == "text":
                if value:
                    user_segments.append(value)
                continue
            user_segments.append(resolve_skill_text(value, corpus, cache))

        if not user_segments:
            user_segments = [" "]
    else:
        if corpus is None:
            raise ValueError("skill_mode=compress requires skill corpus")

        spec_to_index: dict[str, int] = {}
        skill_texts: list[str] = []
        for kind, value in parsed:
            if kind != "skill":
                continue
            if value in spec_to_index:
                continue
            spec_to_index[value] = len(skill_texts)
            skill_texts.append(resolve_skill_text(value, corpus, cache))

        if skill_pad_repeat and skill_texts:
            skill_texts = [text * 10 for text in skill_texts]

        if skill_texts:
            compress_start = time.time()
            skill_enc = model.compressor.tokenizer(
                skill_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(compressor_device)  # tokenizer output goes to compressor device

            with torch.no_grad():
                skill_z = model.compressor(skill_enc["input_ids"], skill_enc["attention_mask"], k=k)
                skill_z = skill_z.to(projector_device)
                skill_soft = model.projector(skill_z)  # projector output on projector_device
                skill_soft = skill_soft.to(llm_device)

            emb_layer = model.llm.get_input_embeddings()
            llm_dtype = emb_layer.weight.dtype if hasattr(emb_layer.weight, 'dtype') else 'unknown'
            # print(f"[generate_with_skills] soft_tokens: shape={skill_soft.shape}, dtype={skill_soft.dtype}, device={skill_soft.device}, llm_emb_dtype={llm_dtype}, llm_device={llm_device}")

            compress_time = time.time() - compress_start

            for kind, value in parsed:
                if kind == "text":
                    if value:
                        user_segments.append(value)
                    continue
                skill_idx = spec_to_index[value]
                user_segments.append(skill_soft[skill_idx, :k, :])
        else:
            for kind, value in parsed:
                if kind == "text" and value:
                    user_segments.append(value)

        if not user_segments:
            user_segments = [" "]

    if mode in {"naive", "full_text"} and not user_segments:
        user_segments = [" "]

    if mode == "full_text":
        # Merge adjacent text segments for cleaner tokenization in full_text mode.
        merged: list[str | torch.Tensor] = []
        buffer = ""
        for seg in user_segments:
            if isinstance(seg, str):
                buffer += seg
                continue
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(seg)
        if buffer:
            merged.append(buffer)
        user_segments = merged

    if mode == "naive":
        # Naive mode must contain only plain text.
        user_segments = [str(user_segments[0])]

    if mode == "full_text":
        # Ensure we only pass text segments in full_text mode.
        user_segments = [str(seg) for seg in user_segments]

    if mode == "compress" and not user_segments:
        user_segments = [" "]

    if mode != "compress":
        # For non-compress modes, guarantee plain-text segments.
        user_segments = [str(seg) for seg in user_segments]

    if mode == "full_text":
        # Remove skill tags defensively in case raw text still contains them.
        user_segments = [SKILL_TAG_PATTERN.sub("", seg) for seg in user_segments if seg]
        if not user_segments:
            user_segments = [" "]

    if mode == "naive":
        user_segments = [SKILL_TAG_PATTERN.sub("", str(user_segments[0])).strip() or " "]

    if mode != "compress" and isinstance(user_segments, list) and len(user_segments) == 1:
        user_segments = [str(user_segments[0])]

    if mode == "compress" and not user_segments:
        user_segments = [" "]

    if not user_segments:
        user_segments = [" "]
        for kind, value in parsed:
            if kind == "text" and value:
                user_segments.append(value)

    embeds_start = time.time()
    if system_text.strip():
        shell_ids = build_chat_template_system_user_shell_ids(model.llm_tokenizer)
        input_embeds = build_chat_prompt_embeds_from_system_and_user_segments(
            llm=model.llm,
            tokenizer=model.llm_tokenizer,
            system_text=system_text,
            user_segments=user_segments,
            template_shell_ids=shell_ids,
        )
    else:
        shell_ids = build_chat_template_user_shell_ids(model.llm_tokenizer)
        input_embeds = build_chat_prompt_embeds_from_user_segments(
            llm=model.llm,
            tokenizer=model.llm_tokenizer,
            segments=user_segments,
            template_shell_ids=shell_ids,
        )
    embeds_time = time.time() - embeds_start

    prefix_len = 0
    if prefix_token_ids is not None and len(prefix_token_ids) > 0:
        emb_layer = model.llm.get_input_embeddings()
        prefix_ids_tensor = torch.tensor(prefix_token_ids, dtype=torch.long, device=llm_device).unsqueeze(0)
        prefix_embeds = emb_layer(prefix_ids_tensor)
        input_embeds = torch.cat([input_embeds, prefix_embeds], dim=1)
        prefix_len = len(prefix_token_ids)
        # print(f"[generate_with_skills] Added prefix tokens: ids={prefix_token_ids}, len={prefix_len}")

    gen_kwargs: dict[str, Any] = {
        "inputs_embeds": input_embeds,
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(do_sample),
        "pad_token_id": model.llm_tokenizer.pad_token_id,
        "eos_token_id": model.llm_tokenizer.eos_token_id,
    }
    if bad_words_ids is not None:
        gen_kwargs["bad_words_ids"] = bad_words_ids
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    input_len = int(input_embeds.size(1))

    total_tokens = input_len + max_new_tokens
    if total_tokens > MAX_TOTAL_TOKENS:
        raise TokenLimitExceededError(
            f"Token limit exceeded: input_len={input_len} + max_new_tokens={max_new_tokens} = {total_tokens} > {MAX_TOTAL_TOKENS}"
        )

    llm_gen_start = time.time()
    with torch.no_grad():
        outputs = model.llm.generate(**gen_kwargs)
    llm_gen_time = time.time() - llm_gen_start

    # Decode only newly generated tokens to avoid garbled prefixes from soft-token prompts.
    decode_start = time.time()
    generated_ids = outputs[0]
    # print(f"generated_ids_len={len(generated_ids)}")
    # print(f"generated_ids={generated_ids[:500]}...")

    if prefix_len > 0 and len(generated_ids) >= prefix_len:
        prefix_match = list(generated_ids[:prefix_len]) == prefix_token_ids
        if prefix_match:
            generated_ids = generated_ids[prefix_len:]
            # print(f"[generate_with_skills] Skipped prefix echo, remaining_len={len(generated_ids)}")

    generated = model.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
    # print(f"generated_len={len(generated)}, generated_response={generated[:500]}...")
    decode_time = time.time() - decode_start

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    func_time = time.time() - func_start
    # print(f"[generate_with_skills] total={func_time:.2f}s (parse={parse_time:.3f}s, compress={compress_time:.2f}s, embeds={embeds_time:.3f}s, llm_gen={llm_gen_time:.2f}s, decode={decode_time:.3f}s)")

    return generated, input_embeds


def compute_teacher_forcing_loss(model: SKIMModel, input_embeds: torch.Tensor, assistant_ref: str) -> float:
    if not assistant_ref.strip():
        return float("nan")

    tokenizer = model.llm_tokenizer
    emb_layer = model.llm.get_input_embeddings()
    llm_device = emb_layer.weight.device if hasattr(emb_layer, 'weight') else model.llm.device

    eos = tokenizer.eos_token or ""
    target_enc = tokenizer(
        [assistant_ref + eos],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(llm_device)

    target_embeds = emb_layer(target_enc["input_ids"])
    merged = torch.cat([input_embeds, target_embeds], dim=1)

    context_mask = torch.ones((1, input_embeds.size(1)), device=llm_device, dtype=torch.long)
    attention_mask = torch.cat([context_mask, target_enc["attention_mask"]], dim=1)

    labels = target_enc["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    ignore = torch.full((1, input_embeds.size(1)), -100, device=llm_device, dtype=torch.long)
    labels = torch.cat([ignore, labels], dim=1)

    with torch.no_grad():
        out = model.llm_forward_with_embeds(
            input_embeds=merged,
            attention_mask=attention_mask,
            labels=labels,
        )
    return float(out.loss.item())


def main() -> None:
    mp.freeze_support()
    args = parse_args()

    cases = load_cases(args)

    if args.parse_only:
        print(f"===== Parse Only Mode =====")
        print(f"total_cases={len(cases)}")
        preview_count = min(5, len(cases))
        for i in range(preview_count):
            meta_info = cases[i].get("meta", {})
            user_text = str(cases[i].get("user_text", ""))
            assistant_ref = str(cases[i].get("assistant_ref", ""))
            print(
                f"case[{i}] meta={meta_info}, user_len={len(user_text)}, assistant_len={len(assistant_ref)}"
            )

        result_path = args.result_path.strip() or _default_result_path(args.input_json)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        print(f"Saved parsed cases to: {result_path}")
        return

    if not args.checkpoint.strip():
        raise ValueError("--checkpoint is required unless --parse_only is set")
    if args.skill_mode in {"compress", "full_text"} and not args.skill_corpus_path.strip():
        raise ValueError("--skill_corpus_path is required when --skill_mode is compress/full_text")

    meta = load_checkpoint_meta(args.checkpoint)
    print(f"===== Total Cases: {len(cases)} =====")
    indexed_cases = list(enumerate(cases))

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_parallel = (
        not args.disable_parallel
        and visible_gpu_count > 1
        and len(indexed_cases) > 1
    )

    if use_parallel:
        print(f"Parallel mode enabled: visible_gpus={visible_gpu_count}, one worker per visible GPU")
        results = _run_parallel_inference(
            indexed_cases=indexed_cases,
            args=args,
            meta=meta,
            visible_gpu_count=visible_gpu_count,
        )
    else:
        if visible_gpu_count <= 1:
            print("Parallel mode disabled automatically (single visible GPU or CPU).")
        elif len(indexed_cases) <= 1:
            print("Parallel mode disabled automatically (only one case).")
        elif args.disable_parallel:
            print("Parallel mode disabled by --disable_parallel.")

        model = build_model(args, meta)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        corpus = load_skill_corpus(args.skill_corpus_path) if args.skill_mode in {"compress", "full_text"} else None

        bar = _create_progress_bar(total=len(indexed_cases), desc="Infer")
        results = []
        for case_index, case in indexed_cases:
            results.append(_run_single_case(model=model, corpus=corpus, case_index=case_index, case=case, args=args))
            bar.update(1)
        bar.close()

    results.sort(key=lambda x: int(x.get("case_index", 0)))

    result_path = args.result_path.strip() or _default_result_path(args.input_json)
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} results to: {result_path}")


if __name__ == "__main__":
    main()
