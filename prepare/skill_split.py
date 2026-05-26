import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

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


def load_env_file_if_exists(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs into environment only if key is not set."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


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


def append_jsonl(path: str, row: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["system_prompt", "user_prompt_template", "output_schema"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing keys in prompt config: {missing}")
    return cfg


def resolve_skill_markdown_path(skill_md_path: str, source_root: str) -> str:
    candidate1 = os.path.join(source_root, skill_md_path)
    if os.path.exists(candidate1):
        return candidate1
    if os.path.exists(skill_md_path):
        return skill_md_path
    raise FileNotFoundError(f"Cannot find skill markdown: {skill_md_path}")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_frontmatter_description(md: str) -> str:
    lines = md.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return ""

    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if line == "---":
            break
        if line.lower().startswith("description:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
        i += 1
    return ""


def build_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    cleaned = text.strip()
    if cleaned:
        candidates.append(cleaned)

    # Any fenced block may contain JSON (sometimes not marked as json explicitly).
    for match in re.finditer(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    # Heuristic: extract from first '{' to last '}' as a potential JSON object payload.
    left = cleaned.find("{")
    right = cleaned.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(cleaned[left : right + 1].strip())

    # Remove duplicates while preserving order.
    seen = set()
    unique: List[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            unique.append(cand)
    return unique


def parse_json_from_text(text: str) -> Dict[str, Any]:
    candidates = build_json_candidates(text)
    errors: List[str] = []
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            errors.append(str(exc))

    head = text[:300].replace("\n", "\\n")
    raise ValueError(f"json_parse_failed; head={head}; errors={errors[:3]}")


def normalize_response_text(response: Any) -> str:
    """Extract content text from multiple response content shapes."""
    try:
        message = response.choices[0].message
    except Exception:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text")
                if txt:
                    chunks.append(str(txt))
                continue

            txt = getattr(part, "text", None)
            if txt:
                chunks.append(str(txt))
            else:
                chunks.append(str(part))
        return "\n".join([x for x in chunks if x])

    return ""


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    marker = "\n\n[TRUNCATED_FOR_MODEL_INPUT]\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars]
    return text[:keep] + marker


def validate_split_result(data: Dict[str, Any], expected_count: int) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("split response is not a JSON object")

    if "split_skills" not in data or not isinstance(data["split_skills"], list):
        raise ValueError("split_skills must be a list")

    split_skills = data["split_skills"]
    if len(split_skills) != expected_count:
        raise ValueError(
            f"split_skills length must be exactly {expected_count}, got {len(split_skills)}"
        )

    for idx, skill in enumerate(split_skills):
        if not isinstance(skill, dict):
            raise ValueError(f"split_skills[{idx}] must be object")

        for key in ["name", "description", "content"]:
            if key not in skill:
                raise ValueError(f"split_skills[{idx}] missing field: {key}")
            if not isinstance(skill[key], str) or not skill[key].strip():
                raise ValueError(f"split_skills[{idx}].{key} must be non-empty string")

        content = skill["content"].strip()
        # Markdown is plain text format. Here we require at least a typical markdown marker
        # to avoid model returning non-content placeholders.
        if "#" not in content and "\n- " not in content and "\n##" not in content:
            raise ValueError(
                f"split_skills[{idx}].content does not look like markdown"
            )

    return data


def build_split_prompt_payload(
    eval_row: Dict[str, Any],
    origin_markdown: str,
    origin_description: str,
    split_count: int,
) -> Dict[str, Any]:
    analysis = eval_row.get("analysis") or {}
    return {
        "id": eval_row.get("id"),
        "name": eval_row.get("name"),
        "owner": eval_row.get("owner"),
        "repo": eval_row.get("repo"),
        "skill_md_path": eval_row.get("skill_md_path"),
        "description": origin_description,
        "analysis": {
            "quality_score": analysis.get("quality_score"),
            "solvable_problems": analysis.get("solvable_problems"),
            "split_skill_count": split_count,
            "tool_specs": analysis.get("tool_specs"),
            "prefer_react": analysis.get("prefer_react"),
            "prefer_react_reason": analysis.get("prefer_react_reason"),
        },
        "original_content_markdown": origin_markdown,
    }


def split_one(
    eval_row: Dict[str, Any],
    llm_client: LLMClient,
    prompt_cfg: Dict[str, Any],
    source_root: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    limiter: QPMRateLimiter,
    max_input_chars: int,
    corpus: Dict[str, Any],
) -> Dict[str, Any]:
    analysis = eval_row.get("analysis") or {}
    split_count = analysis.get("split_skill_count", 0)

    base_result: Dict[str, Any] = {
        "parent_id": eval_row.get("id"),
        "parent_name": eval_row.get("name"),
        "owner": eval_row.get("owner"),
        "repo": eval_row.get("repo"),
        "skill_md_path": eval_row.get("skill_md_path"),
        "split_skill_count": split_count,
        "split_skills": None,
        "error": None,
    }

    try:
        origin_markdown = corpus[eval_row.get("id")]["document"]
        # skill_file = resolve_skill_markdown_path(
        #     eval_row.get("skill_md_path", ""),
        #     source_root=source_root,
        # )
        # origin_markdown = read_text(skill_file)
        # origin_markdown = truncate_text(origin_markdown, max_chars=max_input_chars)
    except Exception as exc:
        base_result["error"] = f"read_source_markdown_failed: {exc}"
        return base_result

    origin_description = parse_frontmatter_description(origin_markdown)
    payload = build_split_prompt_payload(
        eval_row=eval_row,
        origin_markdown=origin_markdown,
        origin_description=origin_description,
        split_count=split_count,
    )

    user_prompt = prompt_cfg["user_prompt_template"].format(
        split_count=split_count,
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
            text, response = llm_client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            if not text.strip():
                text = normalize_response_text(response)

            if not text.strip():
                raise ValueError("Model returned empty response text")

            result = parse_json_from_text(text)
            validated = validate_split_result(result, expected_count=split_count)
            base_result["split_skills"] = validated["split_skills"]
            return base_result
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.2 * attempt)

    base_result["error"] = str(last_error)
    return base_result


def is_eligible(eval_row: Dict[str, Any]) -> bool:
    if eval_row.get("error") is not None:
        return False
    analysis = eval_row.get("analysis")
    if not isinstance(analysis, dict):
        return False
    split_count = analysis.get("split_skill_count")
    return isinstance(split_count, int) and split_count > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split skills based on evaluate output.")
    parser.add_argument("--input-file", type=str, required=True, help="Evaluate output JSONL path")
    parser.add_argument("--corpus-file", type=str, required=True, help="Corpus JSONL path")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL path")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="./skill_split_prompt.json",
        help="Split prompt config JSON",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="./output",
        help="Root folder to resolve skill_md_path",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample")
    parser.add_argument("--max-tokens", type=int, default=12000, help="max_tokens for each request")
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=30000,
        help="Maximum characters of source markdown passed to model",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-workers", type=int, default=8, help="Thread count")
    parser.add_argument("--qpm", type=int, default=60, help="Queries Per Minute")
    parser.add_argument("--max-samples", type=int, default=None, help="Only process first N eligible samples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file before writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_jsonl(args.corpus_file)
    corpus = {x["id"]: x for x in corpus}

    if args.max_retries <= 0:
        raise ValueError("max-retries must be > 0")

    load_env_file_if_exists(".env")

    prompt_cfg = load_prompt_config(args.prompt_file)
    rows = load_jsonl(args.input_file)
    rows = [x for x in rows if is_eligible(x)]

    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if not rows:
        print("No eligible rows to split.")
        return

    if args.overwrite and os.path.exists(args.output_file):
        os.remove(args.output_file)

    llm_client = LLMClient.from_env(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    limiter = QPMRateLimiter(qpm=args.qpm)

    print(f"Eligible size: {len(rows)}")
    print(f"Provider: {llm_client.config.provider}")
    print(f"Model: {llm_client.config.model}")
    print(f"Workers: {args.max_workers}, QPM: {args.qpm}, Retries: {args.max_retries}")

    file_lock = threading.Lock()
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                split_one,
                row,
                llm_client,
                prompt_cfg,
                args.source_root,
                args.max_tokens,
                args.temperature,
                args.max_retries,
                limiter,
                args.max_input_chars,
                corpus,
            )
            for row in rows
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Splitting"):
            result = future.result()
            append_jsonl(args.output_file, result, file_lock)
            if result.get("error"):
                failed += 1
            else:
                success += 1

    print("Done.")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
