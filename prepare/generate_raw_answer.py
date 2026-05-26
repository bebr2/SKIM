import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set

from openai import AzureOpenAI, OpenAI

from llm_client import LLMClient, LLMConfig


class QPMRateLimiter:
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


class SimpleProgressBar:
    def __init__(self, total: int, enabled: bool = True, width: int = 30):
        self.total = max(0, int(total))
        self.enabled = bool(enabled)
        self.width = max(10, int(width))
        self._last_len = 0

    def update(self, current: int, stats: Optional[Dict[str, int]] = None) -> None:
        if not self.enabled:
            return

        if self.total <= 0:
            text = "Progress: 0/0"
        else:
            cur = min(max(0, int(current)), self.total)
            ratio = cur / float(self.total)
            filled = int(round(ratio * self.width))
            bar = "#" * filled + "-" * (self.width - filled)
            text = f"[{bar}] {cur}/{self.total} ({ratio * 100:5.1f}%)"

        if stats:
            ok = stats.get("success", 0)
            failed = stats.get("failed", 0)
            text += f" | ok={ok} failed={failed}"

        pad = " " * max(0, self._last_len - len(text))
        print("\r" + text + pad, end="", flush=True)
        self._last_len = len(text)

    def close(self) -> None:
        if self.enabled:
            print("")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}")
    return rows


def append_jsonl(path: str, row: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row_key(item_id: str, mode: str, query: str) -> str:
    return f"{item_id}||{mode}||{query}".strip()


def read_done_keys(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    done: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            key = make_row_key(
                item_id=str(obj.get("id", "")),
                mode=str(obj.get("mode", "")),
                query=str(obj.get("query", "")),
            )
            if key:
                done.add(key)
    return done


def normalize_output_text(text: str) -> str:
    return text.strip()


def _get_env_or_value(spec: Dict[str, Any], value_key: str, env_key_key: str, fallback_envs: List[str]) -> Optional[str]:
    value = spec.get(value_key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    env_key = spec.get(env_key_key)
    if isinstance(env_key, str) and env_key.strip():
        env_val = os.getenv(env_key.strip())
        if env_val:
            return env_val

    for fallback in fallback_envs:
        env_val = os.getenv(fallback)
        if env_val:
            return env_val
    return None


def build_client_from_spec(
    spec: Dict[str, Any],
    default_model: Optional[str],
    default_max_tokens: int,
    default_temperature: float,
) -> LLMClient:
    mode = str(spec.get("mode") or spec.get("provider") or "env").strip().lower()
    model = spec.get("model") or default_model
    max_tokens = int(spec.get("max_tokens", default_max_tokens))
    temperature = float(spec.get("temperature", default_temperature))

    if mode == "env":
        return LLMClient.from_env(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if mode in {"openai", "openai_compatible", "vllm"}:
        api_key = _get_env_or_value(
            spec,
            value_key="api_key",
            env_key_key="api_key_env",
            fallback_envs=["LLM_API_KEY", "OPENAI_API_KEY"],
        )
        if not api_key:
            raise ValueError("openai_compatible mode requires api_key")

        base_url = _get_env_or_value(
            spec,
            value_key="base_url",
            env_key_key="base_url_env",
            fallback_envs=["LLM_BASE_URL", "OPENAI_BASE_URL"],
        )
        if not model:
            raise ValueError("openai_compatible mode requires model")

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        api_client = OpenAI(**kwargs)
        return LLMClient(
            api_client=api_client,
            config=LLMConfig(
                provider="openai_compatible",
                model=str(model),
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    if mode == "azure":
        api_key = _get_env_or_value(
            spec,
            value_key="api_key",
            env_key_key="api_key_env",
            fallback_envs=["AZURE_OPENAI_API_KEY"],
        )
        endpoint = _get_env_or_value(
            spec,
            value_key="endpoint",
            env_key_key="endpoint_env",
            fallback_envs=["AZURE_OPENAI_ENDPOINT"],
        )
        api_version = _get_env_or_value(
            spec,
            value_key="api_version",
            env_key_key="api_version_env",
            fallback_envs=["AZURE_OPENAI_API_VERSION"],
        )
        if not api_key or not endpoint or not api_version:
            raise ValueError("azure mode requires api_key, endpoint, api_version")
        if not model:
            raise ValueError("azure mode requires model")

        api_client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        return LLMClient(
            api_client=api_client,
            config=LLMConfig(
                provider="azure",
                model=str(model),
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    raise ValueError(f"Unsupported model mode: {mode}")


def build_raw_messages(query: str, need_system_prompt: bool) -> List[Dict[str, Any]]:
    """Build messages without skill context - just the raw query."""
    messages: List[Dict[str, Any]] = []
    if need_system_prompt:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant."
        })
    messages.append({
        "role": "user",
        "content": query
    })
    return messages


def call_model_text(
    llm_client: LLMClient,
    messages: List[Dict[str, Any]],
    limiter: QPMRateLimiter,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    timeout: Optional[float] = None,
) -> Optional[str]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            limiter.acquire()
            text, _ = llm_client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                timeout=timeout,
            )
            norm = normalize_output_text(text)
            if norm:
                return norm
            raise ValueError("empty_response")
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.1 * attempt)
    if last_error is not None:
        return None
    return None


def generate_raw_sample(
    item: Dict[str, Any],
    main_client: LLMClient,
    main_limiter: QPMRateLimiter,
    generation_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generate a sample without skill context."""
    query = item.get("query")
    if not query:
        return None

    need_system_prompt = bool(generation_cfg.get("need_system_prompt_for_direct", True))
    messages = build_raw_messages(query=query, need_system_prompt=need_system_prompt)

    main_timeout = float(generation_cfg.get("main_request_timeout_seconds", 180.0))

    assistant_text = call_model_text(
        llm_client=main_client,
        messages=messages,
        limiter=main_limiter,
        max_retries=int(generation_cfg.get("main_max_retries", 2)),
        max_tokens=int(generation_cfg.get("main_max_tokens", 4096)),
        temperature=float(generation_cfg.get("main_temperature", 0.2)),
        timeout=main_timeout,
    )
    if not assistant_text:
        return None

    messages.append({"role": "assistant", "content": assistant_text})

    # Output format mirrors input, but skill_map is empty and messages_list has no skill
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "owner": item.get("owner"),
        "repo": item.get("repo"),
        "mode": item.get("mode"),
        "query": query,
        "skill_map": {},  # Empty - no skill used
        "messages_list": [messages],
        "meta": {
            "use_split_skills": False,
            "tool_count": 0,
            "react_style": "single_turn",
            "raw_answer": True,  # Mark as raw answer
        },
    }


def run_item_generation(
    item: Dict[str, Any],
    args: argparse.Namespace,
    main_client: LLMClient,
    main_limiter: QPMRateLimiter,
    generation_cfg: Dict[str, Any],
    file_lock: threading.Lock,
    done_keys: Set[str],
    done_lock: threading.Lock,
) -> Dict[str, int]:
    delta: Dict[str, int] = {
        "attempts": 0,
        "failed": 0,
        "resume_skipped": 0,
        "success": 0,
    }

    item_id = str(item.get("id", ""))
    mode = str(item.get("mode", ""))
    query = str(item.get("query", ""))

    delta["attempts"] += 1
    key = make_row_key(item_id=item_id, mode=mode, query=query)

    with done_lock:
        if key in done_keys:
            delta["resume_skipped"] += 1
            return delta

    sample = generate_raw_sample(
        item=item,
        main_client=main_client,
        main_limiter=main_limiter,
        generation_cfg=generation_cfg,
    )
    if sample is None:
        delta["failed"] += 1
        return delta

    append_jsonl(args.output, sample, file_lock)
    with done_lock:
        done_keys.add(key)
    delta["success"] += 1

    return delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw answers (without skill context) from skill-annotated answer files.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Input file from generate_answer.py (direct or react output)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for raw answers")
    parser.add_argument("--config-file", type=str, default="./generate_answer_config.json")
    parser.add_argument("--main-model", type=str, default=None, help="Override main model name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bar")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config_file)
    generation_cfg = dict(config.get("generation", {}))
    rate_cfg = dict(config.get("rate_limit", {}))

    input_rows = load_jsonl(args.input_file)

    if args.overwrite:
        if os.path.exists(args.output):
            os.remove(args.output)

    done_keys = read_done_keys(args.output) if args.resume else set()

    main_model_spec = dict(config.get("main_model", {}))
    if args.main_model:
        main_model_spec["model"] = args.main_model

    main_client = build_client_from_spec(
        spec=main_model_spec,
        default_model=args.main_model,
        default_max_tokens=int(generation_cfg.get("main_max_tokens", 4096)),
        default_temperature=float(generation_cfg.get("main_temperature", 0.2)),
    )

    main_limiter = QPMRateLimiter(qpm=int(rate_cfg.get("main_qpm", 60)))

    file_lock = threading.Lock()
    done_lock = threading.Lock()

    stats: Dict[str, int] = {
        "total": len(input_rows),
        "success": 0,
        "failed": 0,
        "resume_skipped": 0,
    }

    progress = SimpleProgressBar(total=len(input_rows), enabled=(not args.no_progress))
    completed_count = 0
    max_workers = max(1, int(args.workers))
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in input_rows:
            futures.append(
                executor.submit(
                    run_item_generation,
                    item,
                    args,
                    main_client,
                    main_limiter,
                    generation_cfg,
                    file_lock,
                    done_keys,
                    done_lock,
                )
            )

        for future in as_completed(futures):
            try:
                delta = future.result()
            except Exception:
                stats["failed"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            for key, value in delta.items():
                stats[key] = int(stats.get(key, 0)) + int(value)

            completed_count += 1
            progress.update(completed_count, stats)

    progress.close()

    print("Done.")
    print(f"Main provider/model: {main_client.config.provider} / {main_client.config.model}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()