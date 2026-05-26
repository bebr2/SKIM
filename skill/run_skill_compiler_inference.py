from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

import torch

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_ROOT = _REPO_ROOT / "code"
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from skim_inference import (
    build_model,
    generate_with_skills,
    load_checkpoint_meta,
    TokenLimitExceededError,
    MAX_TOTAL_TOKENS,
)
from skillrag_vendor.prompts import ALL_DATASETS, build_prompt
from skillrag_vendor.toolqa import ToolEnvironment
from skillrag_vendor.toolqa.fewshots import TOOLQA_EXAMPLES
from skillrag_vendor.toolqa.react import ReActAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SKIM inference with the SkillRAG-style task flow."
    )
    parser.add_argument("--skillrag_root", type=str, default="", help="Path to SkillRAG-main root")
    parser.add_argument("--dataset", "--name", type=str, default="", help="Dataset name, default all")

    parser.add_argument(
        "--method",
        type=str,
        default="naive",
        help="Skill selection method: naive, golden_skill, or retrieval label",
    )
    parser.add_argument(
        "--retrieval_results",
        type=str,
        default="",
        help="Retrieval JSON path for non-naive/non-golden methods",
    )
    parser.add_argument("--top_k", type=int, default=1, help="Top-k retrieved skills to use")

    parser.add_argument("--instances_dir", type=str, default="", help="Override instances directory")
    parser.add_argument("--corpus_path", type=str, default="", help="Override corpus path")
    parser.add_argument("--toolqa_data_dir", type=str, default="", help="ToolQA corpus directory")

    parser.add_argument("--checkpoint", type=str, required=True, help="SKIM checkpoint directory")
    parser.add_argument(
        "--skill_mode",
        type=str,
        default="compress",
        choices=["naive", "full_text", "compress"],
        help="How to inject skills into the model",
    )

    parser.add_argument("--k", type=int, default=64, help="Soft token length per skill")
    parser.add_argument("--max_length", type=int, default=4096, help="Max skill text token length")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Generation length")
    parser.add_argument("--skill_pad_repeat", action="store_true", help="Repeat skill text 10x before truncation to fill max_length")

    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Sampling top-p")
    parser.add_argument("--fp16", type=str, default="false", help="Use FP16 precision (true/false)")

    parser.add_argument("--compressor_model", type=str, default="", help="Fallback compressor model")
    parser.add_argument("--llm_model", type=str, default="", help="Fallback llm model")
    parser.add_argument("--projector_layers", type=int, default=3, help="Fallback projector layers")
    parser.add_argument("--projector_hidden", type=int, default=4096, help="Fallback projector hidden")
    parser.add_argument("--max_q", type=int, default=256, help="Fallback max_q")

    parser.add_argument(
        "--skill_qa_llm_train_mode",
        type=str,
        default="auto",
        choices=["auto", "false", "true", "lora"],
        help="Override llm_train_mode in checkpoint meta. Use auto to keep checkpoint value.",
    )
    parser.add_argument("--skill_qa_lora_r", type=int, default=16, help="Fallback LoRA rank")
    parser.add_argument("--skill_qa_lora_alpha", type=int, default=32, help="Fallback LoRA alpha")
    parser.add_argument("--skill_qa_lora_dropout", type=float, default=0.05, help="Fallback LoRA dropout")
    parser.add_argument(
        "--skill_qa_lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Fallback comma-separated LoRA target modules",
    )
    parser.add_argument("--skill_qa_lora_bias", type=str, default="none", help="Fallback LoRA bias")
    parser.add_argument(
        "--skill_qa_lora_task_type",
        type=str,
        default="CAUSAL_LM",
        help="Fallback LoRA task type",
    )
    parser.add_argument(
        "--skill_qa_lora_modules_to_save",
        type=str,
        default="",
        help="Fallback comma-separated LoRA modules_to_save",
    )
    parser.add_argument(
        "--skill_qa_lora_use_rslora",
        type=str,
        default="false",
        help="Fallback LoRA use_rslora (true/false)",
    )

    parser.add_argument("--output_root", type=str, default="", help="Output root, default <repo>/results")
    parser.add_argument("--result_path", type=str, default="", help="Single output path, single dataset only")

    parser.add_argument("--resume", action="store_true", help="Skip completed instance_ids in output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files before run")

    parser.add_argument("--disable_parallel", action="store_true", help="Disable multi-GPU sharding")

    parser.add_argument("--toolqa_max_steps", type=int, default=20, help="Max ToolQA ReAct steps")
    parser.add_argument("--toolqa_step_tokens", type=int, default=512, help="Max tokens per ToolQA ReAct step")

    parser.add_argument(
        "--gpus_per_worker",
        type=int,
        default=1,
        help="Number of GPUs per worker process. E.g., 8 GPUs with gpus_per_worker=4 will spawn 2 workers.",
    )

    return parser.parse_args()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    norm = str(value).strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse bool value from: {value!r}")


def _csv_to_list(raw: Any) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _skill_qa_lora_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "r": int(args.skill_qa_lora_r),
        "alpha": int(args.skill_qa_lora_alpha),
        "dropout": float(args.skill_qa_lora_dropout),
        "target_modules": _csv_to_list(args.skill_qa_lora_target_modules),
        "bias": str(args.skill_qa_lora_bias),
        "task_type": str(args.skill_qa_lora_task_type),
        "modules_to_save": _csv_to_list(args.skill_qa_lora_modules_to_save),
        "use_rslora": _to_bool(args.skill_qa_lora_use_rslora),
    }


def _apply_skill_qa_lora_overrides(meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(meta) if isinstance(meta, dict) else {}

    mode_override = str(args.skill_qa_llm_train_mode).strip().lower()
    if mode_override != "auto":
        merged["llm_train_mode"] = mode_override

    effective_mode = str(merged.get("llm_train_mode", "")).strip().lower()
    if effective_mode == "lora" and not isinstance(merged.get("skill_qa_lora"), dict):
        merged["skill_qa_lora"] = _skill_qa_lora_from_args(args)

    return merged


def _resolve_instances_dir(args: argparse.Namespace) -> Path:
    if args.instances_dir.strip():
        return Path(args.instances_dir)
    if args.skillrag_root.strip():
        return Path(args.skillrag_root) / "data" / "bench" / "instances"
    return _REPO_ROOT / "prepare" / "output" / "trainset"


def _resolve_default_corpus_path(args: argparse.Namespace) -> str:
    if args.corpus_path.strip():
        return args.corpus_path.strip()
    if args.skillrag_root.strip():
        return str(Path(args.skillrag_root) / "data" / "bench" / "corpus" / "corpus.json")
    return str(_REPO_ROOT / "prepare" / "output" / "corpus.json")


def _resolve_corpus_file(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path

    if path.is_dir():
        candidate = path / "corpus.json"
        if candidate.is_file():
            return candidate

        nested = path / "corpus.json" / "corpus.json"
        if nested.is_file():
            return nested

    if path.suffix == "":
        nested2 = Path(str(path) + ".json")
        if nested2.is_file():
            return nested2

    raise FileNotFoundError(f"Cannot resolve corpus file from path: {path_like}")


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


def load_skill_corpus(path_like: str) -> dict[str, dict[str, Any]]:
    corpus_file = _resolve_corpus_file(path_like)

    mapping: dict[str, dict[str, Any]] = {}
    if corpus_file.suffix.lower() == ".jsonl":
        with open(corpus_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    continue
                skill_id = str(row.get("skill_id", "")).strip()
                if not skill_id:
                    continue
                mapping[skill_id] = row
    else:
        with open(corpus_file, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        items: list[dict[str, Any]] = []
        _collect_skill_items(payload, items)
        for item in items:
            skill_id = str(item.get("skill_id", "")).strip()
            if not skill_id:
                continue
            mapping[skill_id] = item

    if not mapping:
        raise ValueError(f"No skill rows with skill_id found in corpus: {corpus_file}")
    return mapping


def load_instances(instances_dir: Path, dataset: str) -> list[dict[str, Any]]:
    path = instances_dir / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset instances file not found: {path}")

    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in instances file: {path}")
    return payload


def load_retrieval_map(path: str, top_k: int) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("retrieval_results must contain a top-level 'results' list")

    out: dict[str, list[str]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id", "")).strip()
        retrieved = row.get("retrieved")
        if not instance_id or not isinstance(retrieved, list):
            continue

        skill_ids: list[str] = []
        for item in retrieved[: max(1, int(top_k))]:
            if isinstance(item, dict):
                sid = str(item.get("skill_id", "")).strip()
                if sid:
                    skill_ids.append(sid)
        out[instance_id] = skill_ids

    return out


def _select_skill_ids(
    instance: dict[str, Any],
    method: str,
    retrieval_map: dict[str, list[str]] | None,
) -> list[str]:
    if method == "naive":
        return []

    if method == "golden_skill":
        raw = instance.get("skill_annotations")
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if str(x).strip()]

    if retrieval_map is None:
        return []

    inst_id = str(instance.get("instance_id", "")).strip()
    return list(retrieval_map.get(inst_id, []))


def _valid_skill_ids(skill_ids: list[str], corpus: dict[str, dict[str, Any]]) -> list[str]:
    return [sid for sid in skill_ids if sid in corpus]


def _collect_tools(skill_ids: list[str], corpus: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for sid in skill_ids:
        row = corpus.get(sid)
        if not isinstance(row, dict):
            continue
        row_tools = row.get("tools")
        if isinstance(row_tools, list):
            tools.extend([t for t in row_tools if isinstance(t, dict)])
    return tools


def _inject_skill_block(user_prompt: str, skill_ids: list[str], skill_mode: str, corpus: dict[str, dict[str, Any]]) -> str:
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


def _strip_prompt_echo(generated: str, step_n: int | None = None) -> str:
    text = generated.strip()
    if step_n is None:
        return text

    marker = f"Thought {step_n}:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text


def _load_toolqa_examples(skillrag_root: str) -> str:
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


def _resolve_toolqa_data_dir(args: argparse.Namespace) -> Path:
    if args.toolqa_data_dir.strip():
        path = Path(args.toolqa_data_dir)
        if path.exists():
            return path
        raise FileNotFoundError(f"toolqa_data_dir does not exist: {path}")

    if args.skillrag_root.strip():
        external_dir = Path(args.skillrag_root) / "data" / "external" / "toolqa"
        if external_dir.exists():
            return external_dir

    raise FileNotFoundError(
        "ToolQA data directory is required for dataset=toolqa. Set --toolqa_data_dir explicitly."
    )


def _build_case_prompt(
    instance: dict[str, Any],
    skill_ids: list[str],
    corpus: dict[str, dict[str, Any]],
    skill_mode: str,
) -> tuple[str, str]:
    system, user = build_prompt(instance, method="naive")
    user = _inject_skill_block(user, skill_ids, skill_mode=skill_mode, corpus=corpus)
    return system, user


def _generate_once(
    model,
    corpus: dict[str, dict[str, Any]],
    system_text: str,
    user_text: str,
    args: argparse.Namespace,
) -> str:
    try:
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
            skill_pad_repeat=args.skill_pad_repeat,
        )
        # print(f"generated_len={len(generated)}, generated_response={generated[:500]}...")
        del input_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return generated
    except TokenLimitExceededError as e:
        print(f"[TokenLimitExceeded] {e}")
        return "error"


def _run_qa_case(
    model,
    instance: dict[str, Any],
    method: str,
    corpus: dict[str, dict[str, Any]],
    retrieval_map: dict[str, list[str]] | None,
    args: argparse.Namespace,
    model_short_name: str,
) -> dict[str, Any]:
    instance_id = str(instance.get("instance_id", ""))
    dataset = str(instance.get("dataset", "")) or "unknown"

    skill_ids = _select_skill_ids(instance, method=method, retrieval_map=retrieval_map)
    skill_ids = _valid_skill_ids(skill_ids, corpus)

    system_text, user_text = _build_case_prompt(
        instance,
        skill_ids=skill_ids,
        corpus=corpus,
        skill_mode=args.skill_mode,
    )

    generated = _generate_once(
        model=model,
        corpus=corpus,
        system_text=system_text,
        user_text=user_text,
        args=args,
    )

    out: dict[str, Any] = {
        "instance_id": instance_id,
        "dataset": dataset,
        "method": method,
        "model": model_short_name,
        "raw_output": generated,
    }
    if skill_ids:
        out["skill_ids_used"] = skill_ids
    return out


def _run_toolqa_case(
    model,
    instance: dict[str, Any],
    method: str,
    corpus: dict[str, dict[str, Any]],
    retrieval_map: dict[str, list[str]] | None,
    tool_env: ToolEnvironment,
    examples: str,
    args: argparse.Namespace,
    model_short_name: str,
) -> dict[str, Any]:
    case_start = time.time()
    instance_id = str(instance.get("instance_id", ""))
    question_preview = str(instance.get("question", ""))[:80]
    print(f"\n[ToolQA Case] instance_id={instance_id}, question='{question_preview}...'")

    skill_ids = _select_skill_ids(instance, method=method, retrieval_map=retrieval_map)
    skill_ids = _valid_skill_ids(skill_ids, corpus)
    skills: list[str] = []
    for sid in skill_ids:
        row = corpus.get(sid, {})
        content = row.get("content", "") if isinstance(row, dict) else ""
        if isinstance(content, str) and content.strip():
            skills.append(content)

    tool_env.reset()

    def _model_generate(
        system_text: str,
        user_text: str,
        max_tokens: int,
        step_n: int,
        stop_token: str | None,
    ) -> str:
        gen_start = time.time()
        try:
            generated, input_embeds = generate_with_skills(
                model=model,
                corpus=corpus,
                system_text=system_text,
                user_text=user_text,
                skill_mode=args.skill_mode,
                k=args.k,
                max_length=args.max_length,
                max_new_tokens=max_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                skill_pad_repeat=args.skill_pad_repeat,
            )
            gen_time = time.time() - gen_start
            del input_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            text = _strip_prompt_echo(generated, step_n=step_n)
            if stop_token and stop_token in text:
                text = text.split(stop_token, 1)[0]
            print(f"[ToolQA Case] _model_generate (step {step_n}) took {gen_time:.2f}s, generated_len={len(text)}, generated_response={text[:500]}...")
            return text
        except TokenLimitExceededError as e:
            gen_time = time.time() - gen_start
            print(f"[ToolQA Case] _model_generate (step {step_n}) TokenLimitExceeded after {gen_time:.2f}s: {e}")
            return "error"

    react_method = method if skill_ids else "naive"
    agent = ReActAgent(
        question=str(instance.get("question", "")),
        tools=tool_env,
        model_generate=_model_generate,
        examples=examples,
        max_steps=args.toolqa_max_steps,
        max_tokens=args.toolqa_step_tokens,
        method=react_method,
        skills=skills,
        skill_ids=skill_ids,
        skill_mode=args.skill_mode,
    )
    agent.run()

    case_time = time.time() - case_start
    print(f"[ToolQA Case] instance_id={instance_id} completed in {case_time:.2f}s, steps={agent.step_n - 1}")

    out: dict[str, Any] = {
        "instance_id": instance_id,
        "dataset": "toolqa",
        "method": method,
        "model": model_short_name,
        "raw_output": agent.scratchpad,
        "n_steps": max(0, agent.step_n - 1),
        "finished": bool(agent.finished),
        "halted": bool(agent.is_halted()),
        "time_seconds": round(case_time, 2),
    }
    if skill_ids:
        out["skill_ids_used"] = skill_ids

    return out


def _read_done_instance_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            instance_id = row.get("instance_id")
            if isinstance(instance_id, str) and instance_id.strip():
                done.add(instance_id.strip())
    return done


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_evenly(
    indexed_instances: list[tuple[int, dict[str, Any]]],
    num_parts: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    num_parts = max(1, int(num_parts))
    shards: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(num_parts)]
    for idx, item in enumerate(indexed_instances):
        shards[idx % num_parts].append(item)
    return shards


def _worker_infer_shard(
    worker_id: int,
    gpu_ids: list[int],
    shard: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    meta: dict[str, Any],
    dataset: str,
    method: str,
    corpus_path: str,
    retrieval_map: dict[str, list[str]] | None,
    toolqa_data_dir: str,
    toolqa_examples: str,
    model_short_name: str,
    progress_queue,
    result_queue,
) -> None:
    try:
        import os
        worker_start = time.time()
        gpus_per_worker = len(gpu_ids)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        if torch.cuda.is_initialized():
            torch.cuda.reset()

        use_fp16 = _to_bool(args.fp16) or (dataset in {"toolqa"})
        torch_dtype = torch.bfloat16 if use_fp16 else None

        skip_compressor = (args.skill_mode in {"naive", "full_text"})

        is_toolqa = (dataset == "toolqa")
        toolqa_device_config = is_toolqa and gpus_per_worker >= 2 and not skip_compressor

        model_load_start = time.time()
        if gpus_per_worker >= 2:
            if toolqa_device_config:
                torch.cuda.set_device(0)

                llm_max_memory = {
                    0: "1GiB",
                    1: "40GiB",
                }

                model = build_model(
                    args, meta,
                    llm_device_map="auto",
                    llm_max_memory=llm_max_memory,
                    torch_dtype=torch_dtype,
                    skip_compressor=skip_compressor,
                )

                if model.compressor is not None:
                    compressor_device = torch.device("cuda:1")
                    model.compressor = model.compressor.to(compressor_device)
                    model.compressor.learnable_q = model.compressor.learnable_q.to(compressor_device)

                llm_first_device = model.get_llm_first_device()
                model.projector = model.projector.to(llm_first_device)

                print(f"[Worker {worker_id}] TOOLQA config: SKIM model on cuda:1, tools on cuda:0")
                print(f"[Worker {worker_id}] LLM device_map: {model.llm.hf_device_map if hasattr(model.llm, 'hf_device_map') else 'N/A'}, dtype={torch_dtype}, skip_compressor={skip_compressor}")
            else:
                llm_max_memory = {}
                for i in range(gpus_per_worker):
                    llm_max_memory[i] = "40GiB"

                model = build_model(
                    args, meta,
                    llm_device_map="auto",
                    llm_max_memory=llm_max_memory,
                    torch_dtype=torch_dtype,
                    skip_compressor=skip_compressor,
                )

                llm_first_device = model.get_llm_first_device()
                print(f"[Worker {worker_id}] LLM first device: {llm_first_device}, dtype={torch_dtype}, skip_compressor={skip_compressor}")
                print(f"[Worker {worker_id}] LLM device_map: {model.llm.hf_device_map if hasattr(model.llm, 'hf_device_map') else 'N/A'}")

                if model.compressor is not None:
                    compressor_device = torch.device("cuda:0")
                    model.compressor = model.compressor.to(compressor_device)
                    model.compressor.learnable_q = model.compressor.learnable_q.to(compressor_device)
                model.projector = model.projector.to(llm_first_device)

                print(f"[Worker {worker_id}] physical GPUs {gpu_ids}: compressor on cuda:0, LLM on multiple GPUs")
        elif gpus_per_worker == 1:
            device = torch.device("cuda:0")
            model = build_model(args, meta, torch_dtype=torch_dtype, skip_compressor=skip_compressor)
            model = model.to(device)
            print(f"[Worker {worker_id}] single GPU {gpu_ids[0]}, dtype={torch_dtype}, skip_compressor={skip_compressor}")
        else:
            model = build_model(args, meta, torch_dtype=torch_dtype, skip_compressor=skip_compressor)
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print(f"[Worker {worker_id}] CPU mode, skip_compressor={skip_compressor}")

        model_load_time = time.time() - model_load_start
        print(f"[Worker {worker_id}] Model loading took {model_load_time:.2f}s")

        model.eval()

        needs_corpus = method != "naive" or args.skill_mode in {"compress", "full_text"}
        corpus_load_start = time.time()
        corpus = load_skill_corpus(corpus_path) if needs_corpus else {}
        corpus_load_time = time.time() - corpus_load_start
        if needs_corpus:
            print(f"[Worker {worker_id}] Corpus loading took {corpus_load_time:.2f}s, {len(corpus)} skills")

        tool_env = None
        if dataset == "toolqa":
            tool_env_start = time.time()
            tool_env = ToolEnvironment(Path(toolqa_data_dir))
            tool_env._ensure_retrievers_ready()
            tool_env_time = time.time() - tool_env_start
            print(f"[Worker {worker_id}] ToolEnvironment init + retrievers took {tool_env_time:.2f}s")

        worker_init_time = time.time() - worker_start
        print(f"[Worker {worker_id}] === Worker init total: {worker_init_time:.2f}s ===")
        print(f"[Worker {worker_id}] Starting inference on {len(shard)} instances...")

        shard_results: list[dict[str, Any]] = []
        for order_idx, instance in shard:
            if dataset == "toolqa":
                record = _run_toolqa_case(
                    model=model,
                    instance=instance,
                    method=method,
                    corpus=corpus,
                    retrieval_map=retrieval_map,
                    tool_env=tool_env,
                    examples=toolqa_examples,
                    args=args,
                    model_short_name=model_short_name,
                )
            else:
                record = _run_qa_case(
                    model=model,
                    instance=instance,
                    method=method,
                    corpus=corpus,
                    retrieval_map=retrieval_map,
                    args=args,
                    model_short_name=model_short_name,
                )
            record["__order__"] = order_idx
            shard_results.append(record)
            progress_queue.put(1)

        result_queue.put({"worker_id": worker_id, "results": shard_results})
    except RuntimeError as e:
        error_msg = str(e)
        is_oom = "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower()
        result_queue.put(
            {
                "worker_id": worker_id,
                "fatal_error": True,
                "cuda_oom": is_oom,
                "error": error_msg,
                "traceback": traceback.format_exc(),
            }
        )
    except SystemExit as e:
        result_queue.put(
            {
                "worker_id": worker_id,
                "fatal_error": True,
                "cuda_oom": True,
                "error": f"Process exited with code {e.code}",
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": worker_id,
                "fatal_error": True,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_parallel_dataset(
    indexed_instances: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    meta: dict[str, Any],
    dataset: str,
    method: str,
    corpus_path: str,
    retrieval_map: dict[str, list[str]] | None,
    toolqa_data_dir: str,
    toolqa_examples: str,
    model_short_name: str,
    visible_gpu_count: int,
) -> list[dict[str, Any]]:
    gpus_per_worker = max(1, int(args.gpus_per_worker))
    num_workers = visible_gpu_count // gpus_per_worker
    if num_workers == 0:
        num_workers = 1

    print(f"Parallel config: {visible_gpu_count} GPUs, {gpus_per_worker} GPUs per worker, {num_workers} workers")

    gpu_groups = []
    for worker_idx in range(num_workers):
        start_gpu = worker_idx * gpus_per_worker
        end_gpu = min(start_gpu + gpus_per_worker, visible_gpu_count)
        gpu_group = list(range(start_gpu, end_gpu))
        gpu_groups.append(gpu_group)

    shards = _split_evenly(indexed_instances, num_workers)
    worker_specs = [(gpu_group, shard) for gpu_group, shard in zip(gpu_groups, shards) if shard]
    if not worker_specs:
        return []

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = []

    for worker_id, (gpu_ids, shard) in enumerate(worker_specs):
        process = ctx.Process(
            target=_worker_infer_shard,
            args=(
                worker_id,
                gpu_ids,
                shard,
                args,
                meta,
                dataset,
                method,
                corpus_path,
                retrieval_map,
                toolqa_data_dir,
                toolqa_examples,
                model_short_name,
                progress_queue,
                result_queue,
            ),
        )
        process.start()
        processes.append(process)

    merged: list[dict[str, Any]] = []
    finished_workers = 0

    bar = _tqdm(total=len(indexed_instances), desc=f"Infer-{dataset}", dynamic_ncols=True) if _tqdm else None

    while finished_workers < len(processes):
        try:
            progress_queue.get(timeout=0.2)
            if bar is not None:
                bar.update(1)
        except Empty:
            pass

        try:
            payload = result_queue.get_nowait()
        except Empty:
            continue

        if payload.get("fatal_error") or payload.get("cuda_oom"):
            if bar is not None:
                bar.close()
            error_type = "CUDA OOM" if payload.get("cuda_oom") else "Fatal Error"
            print(f"\n[{error_type}] Worker {payload.get('worker_id')} detected")
            print(f"Error: {payload.get('error')}")
            print(payload.get('traceback', ''))
            for proc in processes:
                proc.terminate()
            for proc in processes:
                proc.join(timeout=2)
            sys.exit(1)

        merged.extend(payload.get("results", []))
        finished_workers += 1

    if bar is not None:
        while True:
            try:
                progress_queue.get_nowait()
                bar.update(1)
            except Empty:
                break
        bar.close()

    for process in processes:
        process.join()

    abnormal_exit = [f"pid={proc.pid}, exitcode={proc.exitcode}" for proc in processes if proc.exitcode not in (0, None)]
    if abnormal_exit:
        raise RuntimeError("Parallel inference failed (abnormal worker exits):\n" + "\n".join(abnormal_exit))

    return merged


def _model_short_name(meta: dict[str, Any], args: argparse.Namespace) -> str:
    llm_name = str(meta.get("llm_model") or args.llm_model or "skill_qa")
    return Path(llm_name).name


def _dataset_output_path(
    dataset: str,
    method: str,
    model_short_name: str,
    args: argparse.Namespace,
    is_single_dataset: bool,
) -> Path:
    if args.result_path.strip():
        if not is_single_dataset:
            raise ValueError("--result_path can only be used with a single dataset")
        return Path(args.result_path)

    output_root = Path(args.output_root) if args.output_root.strip() else (_REPO_ROOT / "results")
    return output_root / dataset / model_short_name / f"{method}.jsonl"


def _run_single_dataset(
    dataset: str,
    args: argparse.Namespace,
    meta: dict[str, Any],
    instances_dir: Path,
    corpus_path: str,
    retrieval_map: dict[str, list[str]] | None,
    toolqa_data_dir: str,
    toolqa_examples: str,
    model_short_name: str,
    is_single_dataset: bool,
) -> None:
    instances = load_instances(instances_dir, dataset)
    out_path = _dataset_output_path(
        dataset=dataset,
        method=args.method,
        model_short_name=model_short_name,
        args=args,
        is_single_dataset=is_single_dataset,
    )

    if args.overwrite and out_path.exists():
        out_path.unlink()

    done_ids = _read_done_instance_ids(out_path) if args.resume else set()
    pending = [row for row in instances if str(row.get("instance_id", "")).strip() not in done_ids]

    if not pending:
        print(f"[{dataset}] No pending instances. Output: {out_path}")
        return

    print(
        f"[{dataset}] total={len(instances)}, done={len(done_ids)}, pending={len(pending)}, "
        f"method={args.method}, skill_mode={args.skill_mode}"
    )

    indexed = list(enumerate(pending))
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_parallel = (
        not args.disable_parallel
        and visible_gpu_count > 1
        and len(indexed) > 1
    )

    if use_parallel:
        print(f"[{dataset}] Parallel workers={visible_gpu_count}")
        rows = _run_parallel_dataset(
            indexed_instances=indexed,
            args=args,
            meta=meta,
            dataset=dataset,
            method=args.method,
            corpus_path=corpus_path,
            retrieval_map=retrieval_map,
            toolqa_data_dir=toolqa_data_dir,
            toolqa_examples=toolqa_examples,
            model_short_name=model_short_name,
            visible_gpu_count=visible_gpu_count,
        )
    else:
        if args.disable_parallel:
            print(f"[{dataset}] Parallel disabled by flag")
        elif visible_gpu_count <= 1:
            print(f"[{dataset}] Parallel disabled due to single GPU/CPU")
        else:
            print(f"[{dataset}] Parallel disabled due to small pending size")

        use_fp16 = _to_bool(args.fp16) or (dataset in {"toolqa"})
        torch_dtype = torch.bfloat16 if use_fp16 else None
        print(f"[{dataset}] torch_dtype={torch_dtype}")

        skip_compressor = (args.skill_mode in {"naive", "full_text"})
        print(f"[{dataset}] skip_compressor={skip_compressor}")

        is_toolqa = (dataset == "toolqa")

        gpus_per_worker = max(1, int(args.gpus_per_worker))
        if gpus_per_worker >= 2 and visible_gpu_count >= gpus_per_worker:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpus_per_worker))

            if is_toolqa and not skip_compressor:
                torch.cuda.set_device(0)

                llm_max_memory = {
                    0: "1GiB",
                    1: "40GiB",
                }

                model = build_model(
                    args, meta,
                    llm_device_map="auto",
                    llm_max_memory=llm_max_memory,
                    torch_dtype=torch_dtype,
                    skip_compressor=skip_compressor,
                )

                if model.compressor is not None:
                    compressor_device = torch.device("cuda:1")
                    model.compressor = model.compressor.to(compressor_device)
                    model.compressor.learnable_q = model.compressor.learnable_q.to(compressor_device)

                llm_first_device = model.get_llm_first_device()
                model.projector = model.projector.to(llm_first_device)

                print(f"[{dataset}] TOOLQA config: SKIM model on cuda:1, tools on cuda:0")
                print(f"[{dataset}] LLM device_map: {model.llm.hf_device_map if hasattr(model.llm, 'hf_device_map') else 'N/A'}")
            else:
                llm_max_memory = {
                    0: "2GiB",
                }
                for i in range(1, gpus_per_worker):
                    llm_max_memory[i] = "30GiB"

                model = build_model(
                    args, meta,
                    llm_device_map="auto",
                    llm_max_memory=llm_max_memory,
                    torch_dtype=torch_dtype,
                    skip_compressor=skip_compressor,
                )

                if model.compressor is not None:
                    compressor_device = torch.device("cuda:0")
                    model.compressor = model.compressor.to(compressor_device)
                    model.compressor.learnable_q = model.compressor.learnable_q.to(compressor_device)

                llm_first_device = model.get_llm_first_device()
                model.projector = model.projector.to(llm_first_device)

                print(f"[{dataset}] compressor on cuda:0, LLM pipeline parallel on cuda:1..cuda:{gpus_per_worker-1}")
                print(f"[{dataset}] LLM device_map: {model.llm.hf_device_map if hasattr(model.llm, 'hf_device_map') else 'N/A'}")
        else:
            model = build_model(args, meta, torch_dtype=torch_dtype, skip_compressor=skip_compressor)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

        model.eval()

        needs_corpus = args.method != "naive" or args.skill_mode in {"compress", "full_text"}
        corpus = load_skill_corpus(corpus_path) if needs_corpus else {}

        tool_env = ToolEnvironment(Path(toolqa_data_dir)) if dataset == "toolqa" else None
        if tool_env is not None:
            tool_env._ensure_retrievers_ready()

        rows = []
        bar = _tqdm(total=len(indexed), desc=f"Infer-{dataset}", dynamic_ncols=True) if _tqdm else None
        for order_idx, instance in indexed:
            if dataset == "toolqa":
                row = _run_toolqa_case(
                    model=model,
                    instance=instance,
                    method=args.method,
                    corpus=corpus,
                    retrieval_map=retrieval_map,
                    tool_env=tool_env,
                    examples=toolqa_examples,
                    args=args,
                    model_short_name=model_short_name,
                )
            else:
                row = _run_qa_case(
                    model=model,
                    instance=instance,
                    method=args.method,
                    corpus=corpus,
                    retrieval_map=retrieval_map,
                    args=args,
                    model_short_name=model_short_name,
                )
            row["__order__"] = order_idx
            rows.append(row)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

    rows.sort(key=lambda x: int(x.get("__order__", 0)))
    for row in rows:
        row.pop("__order__", None)

    _append_jsonl(out_path, rows)
    print(f"[{dataset}] Wrote {len(rows)} rows to {out_path}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.method not in {"naive", "golden_skill"} and not args.retrieval_results.strip():
        raise ValueError("retrieval_results is required when method is not naive/golden_skill")

    if args.top_k <= 0:
        raise ValueError("top_k must be > 0")



def main() -> None:
    mp.freeze_support()
    args = parse_args()
    
    _validate_args(args)

    raw_meta = load_checkpoint_meta(args.checkpoint)
    meta = _apply_skill_qa_lora_overrides(raw_meta, args)
    model_short_name = _model_short_name(meta, args)

    datasets = [args.dataset.strip()] if args.dataset.strip() else list(ALL_DATASETS)
    instances_dir = _resolve_instances_dir(args)
    corpus_path = _resolve_default_corpus_path(args)

    retrieval_map = None
    if args.method not in {"naive", "golden_skill"}:
        retrieval_map = load_retrieval_map(args.retrieval_results, args.top_k)
        print(f"Loaded retrieval map: {len(retrieval_map)} instances")

    toolqa_data_dir = ""
    toolqa_examples = ""
    if "toolqa" in datasets:
        toolqa_data_dir = str(_resolve_toolqa_data_dir(args))
        toolqa_examples = _load_toolqa_examples(args.skillrag_root)

    print(f"Datasets: {datasets}")
    print(f"Instances dir: {instances_dir}")
    print(f"Corpus path: {corpus_path}")

    for dataset in datasets:
        _run_single_dataset(
            dataset=dataset,
            args=args,
            meta=meta,
            instances_dir=instances_dir,
            corpus_path=corpus_path,
            retrieval_map=retrieval_map,
            toolqa_data_dir=toolqa_data_dir,
            toolqa_examples=toolqa_examples,
            model_short_name=model_short_name,
            is_single_dataset=(len(datasets) == 1),
        )


if __name__ == "__main__":
    main()
