import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from llm_client import LLMClient


class QPMRateLimiter:
    """Global request limiter: controls request start rate by Queries Per Minute."""

    def __init__(self, qpm: int):
        if qpm <= 0:
            raise ValueError("qpm must be > 0")
        self.interval = 60.0 / float(qpm)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                if now >= self._next_allowed:
                    self._next_allowed = now + self.interval
                    return
                wait_seconds = self._next_allowed - now
            time.sleep(wait_seconds)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}")
    return rows


def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["system_prompt", "user_prompt_template", "output_schema"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing keys in prompt config: {missing}")
    return cfg


def build_prompt_input(skill_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": skill_item.get("id"),
        "name": skill_item.get("name"),
        "description": skill_item.get("description"),
        "owner": skill_item.get("owner"),
        "repo": skill_item.get("repo"),
        "skill_md_path": skill_item.get("skill_md_path"),
        "document": skill_item.get("document", ""),
    }


def validate_analysis(data: Dict[str, Any], k: int) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("analysis is not a JSON object")

    if "quality_score" not in data:
        raise ValueError("Missing field: quality_score")
    score = data["quality_score"]
    if not isinstance(score, (int, float)):
        raise ValueError("quality_score must be number")
    if score < 0 or score > 5:
        raise ValueError("quality_score must be in [0, 5]")

    problems = data.get("solvable_problems")
    if not isinstance(problems, list):
        raise ValueError("solvable_problems must be a list")
    if len(problems) != k:
        raise ValueError(f"solvable_problems length must be exactly {k}")
    if not all(isinstance(p, str) and p.strip() for p in problems):
        raise ValueError("every solvable_problems item must be non-empty string")

    split_count = data.get("split_skill_count")
    if not isinstance(split_count, int) or split_count < 0:
        raise ValueError("split_skill_count must be non-negative int")

    tools = data.get("tool_specs")
    if not isinstance(tools, list):
        raise ValueError("tool_specs must be a list")

    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tool_specs[{idx}] must be an object")
        for key in ["name", "description", "input_params", "output_params"]:
            if key not in tool:
                raise ValueError(f"tool_specs[{idx}] missing field: {key}")
        if not isinstance(tool["name"], str) or not tool["name"].strip():
            raise ValueError(f"tool_specs[{idx}].name must be non-empty string")
        if not isinstance(tool["description"], str):
            raise ValueError(f"tool_specs[{idx}].description must be string")

        for field_name in ["input_params", "output_params"]:
            params = tool[field_name]
            if not isinstance(params, list):
                raise ValueError(f"tool_specs[{idx}].{field_name} must be list")
            for p_idx, param in enumerate(params):
                if not isinstance(param, dict):
                    raise ValueError(f"tool_specs[{idx}].{field_name}[{p_idx}] must be object")
                if "name" not in param or "description" not in param:
                    raise ValueError(
                        f"tool_specs[{idx}].{field_name}[{p_idx}] needs name and description"
                    )
                if not isinstance(param["name"], str) or not param["name"].strip():
                    raise ValueError(
                        f"tool_specs[{idx}].{field_name}[{p_idx}].name must be non-empty string"
                    )
                if not isinstance(param["description"], str):
                    raise ValueError(
                        f"tool_specs[{idx}].{field_name}[{p_idx}].description must be string"
                    )

    prefer_react = data.get("prefer_react")
    if not isinstance(prefer_react, bool):
        raise ValueError("prefer_react must be boolean")

    reason = data.get("prefer_react_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("prefer_react_reason must be non-empty string")

    data["quality_score"] = float(score)
    return data


def evaluate_one(
    item: Dict[str, Any],
    llm_client: LLMClient,
    prompt_cfg: Dict[str, Any],
    k: int,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    limiter: QPMRateLimiter,
) -> Dict[str, Any]:
    payload = build_prompt_input(item)
    user_prompt = prompt_cfg["user_prompt_template"].format(
        k=k,
        output_schema=json.dumps(prompt_cfg["output_schema"], ensure_ascii=False, indent=2),
        skill_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    messages = [
        llm_client.build_text_message("system", prompt_cfg["system_prompt"]),
        llm_client.build_text_message("user", user_prompt),
    ]

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            limiter.acquire()
            result = llm_client.chat_json(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                max_retries=1,
            )
            validated = validate_analysis(result, k=k)
            return {
                "id": item.get("id"),
                "name": item.get("name"),
                "owner": item.get("owner"),
                "repo": item.get("repo"),
                "skill_md_path": item.get("skill_md_path"),
                "analysis": validated,
                "error": None,
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.2 * attempt)

    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "owner": item.get("owner"),
        "repo": item.get("repo"),
        "skill_md_path": item.get("skill_md_path"),
        "analysis": None,
        "error": str(last_error),
    }


def append_jsonl(path: str, row: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_done_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj_id = obj.get("id")
                if obj_id:
                    done.add(obj_id)
            except Exception:
                continue
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate skills with LLM in batch mode.")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--prompt-file", type=str, default="./prompt.json", help="Prompt config JSON")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--k", type=int, default=5, help="How many solvable problems to generate")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample")
    parser.add_argument("--max-tokens", type=int, default=12000, help="max_tokens for each request")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-workers", type=int, default=8, help="Thread count")
    parser.add_argument("--qpm", type=int, default=60, help="Queries Per Minute")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already in output-file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output-file")
    parser.add_argument("--max-samples", type=int, default=None, help="Only evaluate this many samples (for testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("k must be > 0")
    if args.max_retries <= 0:
        raise ValueError("max-retries must be > 0")

    prompt_cfg = load_prompt_config(args.prompt_file)
    items = load_jsonl(args.input_file)

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    if args.max_samples is not None:
        items = items[:args.max_samples]
        
    if args.resume:
        done_ids = read_done_ids(args.output_file)
        items = [x for x in items if x.get("id") not in done_ids]

    

    if not items:
        print("No items to evaluate.")
        return

    llm_client = LLMClient.from_env(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    limiter = QPMRateLimiter(qpm=args.qpm)

    print(f"Input size: {len(items)}")
    print(f"Provider: {llm_client.config.provider}")
    print(f"Model: {llm_client.config.model}")
    print(f"Workers: {args.max_workers}, QPM: {args.qpm}, Retries: {args.max_retries}")

    file_lock = threading.Lock()
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                item,
                llm_client,
                prompt_cfg,
                args.k,
                args.max_retries,
                args.max_tokens,
                args.temperature,
                limiter,
            )
            for item in items
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            row = future.result()
            append_jsonl(args.output_file, row, file_lock)
            if row.get("error"):
                failed += 1
            else:
                success += 1

    print("Done.")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
