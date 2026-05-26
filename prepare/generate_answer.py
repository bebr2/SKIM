import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AzureOpenAI, OpenAI

from llm_client import LLMClient, LLMConfig
from utils import get_qa_message_not_react, get_qa_tool_prompt_react


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
            ok = int(stats.get("success_direct", 0)) + int(stats.get("success_react", 0))
            skipped = (
                int(stats.get("skip_missing_analysis", 0))
                + int(stats.get("skip_quality", 0))
                + int(stats.get("skip_missing_origin", 0))
                + int(stats.get("skip_missing_split", 0))
                + int(stats.get("skip_mode", 0))
                + int(stats.get("skip_react_probability", 0))
                + int(stats.get("skip_no_queries", 0))
                + int(stats.get("skip_react_all_failed", 0))
            )
            text += f" | ok={ok} direct={stats.get('success_direct', 0)} react={stats.get('success_react', 0)} skip={skipped}"

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


def clamp_probability(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def make_row_key(item_id: str, mode: str, query: str) -> str:
    return f"{item_id}||{mode}||{query}".strip()


def append_extra_prompt(query: str, add_extra_prompt: str) -> str:
    if add_extra_prompt:
        return query + add_extra_prompt
    return query


def has_tools(analysis: Dict[str, Any]) -> bool:
    tools = analysis.get("tool_specs")
    return isinstance(tools, list) and len(tools) > 0


def build_origin_index(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    origin: Dict[str, str] = {}
    for row in rows:
        item_id = row.get("id")
        document = row.get("document")
        if isinstance(item_id, str) and isinstance(document, str) and document.strip():
            origin[item_id] = document
    return origin


def build_split_index(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    split_index: Dict[str, List[str]] = {}
    for row in rows:
        parent_id = row.get("parent_id")
        split_skills = row.get("split_skills")
        if not isinstance(parent_id, str) or not isinstance(split_skills, list):
            continue
        contents: List[str] = []
        for split_skill in split_skills:
            if not isinstance(split_skill, dict):
                continue
            content = split_skill.get("content")
            if isinstance(content, str) and content.strip():
                contents.append(content)
        if contents:
            split_index[parent_id] = contents
    return split_index


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


def normalize_output_text(text: str) -> str:
    return text.strip()


SKILL_PLACEHOLDER_PATTERN = re.compile(r"<skill>\s*(.*?)\s*</skill>", flags=re.DOTALL)


def render_skill_placeholders(text: str, skill_map: Dict[str, Any]) -> str:
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        key = match.group(1).strip()
        mapped = skill_map.get(key)
        if isinstance(mapped, str):
            return mapped
        return key

    rendered = SKILL_PLACEHOLDER_PATTERN.sub(_replace, text)
    return rendered.replace("<skill>", "").replace("</skill>", "")


def materialize_messages_with_skills(
    messages: List[Dict[str, Any]],
    skill_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rendered_messages: List[Dict[str, Any]] = []
    for message in messages:
        rendered = dict(message)
        content = rendered.get("content")
        if isinstance(content, str):
            rendered["content"] = render_skill_placeholders(content, skill_map)
        rendered_messages.append(rendered)
    return rendered_messages


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


def parse_action(action_str: str, action_pattern: str) -> Tuple[Optional[str], Optional[str]]:
    action_str = action_str.strip()
    if action_str.startswith("PythonInterpreter[") and action_str.endswith("]"):
        return "PythonInterpreter", action_str[len("PythonInterpreter[") : -1]

    match = re.match(action_pattern, action_str, flags=re.DOTALL)
    if not match:
        return None, None
    return match.group(1), match.group(2)


@dataclass
class ParsedTurn:
    thought: str
    action: str
    action_type: str
    action_args: str


@dataclass
class ItemWork:
    item: Dict[str, Any]
    item_id: str
    mode: str
    direct_queries: List[str]
    react_queries: List[str]
    skills: List[str]
    tools: List[Dict[str, Any]]
    use_split: bool


def parse_react_turn(text: str, step: int, action_pattern: str) -> Optional[ParsedTurn]:
    thought_match = re.search(
        rf"Thought\s*{step}\s*:\s*(.+?)(?=\n\s*Action\s*{step}\s*:)",
        text,
        flags=re.DOTALL,
    )
    action_match = re.search(
        rf"Action\s*{step}\s*:\s*(.+)$",
        text,
        flags=re.DOTALL,
    )
    if thought_match is None or action_match is None:
        return None

    thought = thought_match.group(1).strip()
    action_block = action_match.group(1).strip()

    if action_block.startswith("PythonInterpreter["):
        action = action_block
    else:
        first_line = action_block.splitlines()[0].strip() if action_block else ""
        action = first_line

    if not thought or not action:
        return None

    action_type, action_args = parse_action(action, action_pattern)
    if action_type is None or action_args is None:
        return None

    return ParsedTurn(
        thought=thought,
        action=action,
        action_type=action_type,
        action_args=action_args,
    )


def tail_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def split_react_user_prefix(user_prompt: str) -> str:
    marker = "\nThought 1:"
    idx = user_prompt.rfind(marker)
    if idx == -1:
        return user_prompt.rstrip()
    return user_prompt[:idx].rstrip()


def build_react_user_prompt(base_user_prefix: str, scratchpad: str, step: int) -> str:
    return f"{base_user_prefix}{scratchpad}\nThought {step}:"


def contains_reject_phrase(text: str, reject_phrases: List[str]) -> bool:
    lower = text.lower()
    for phrase in reject_phrases:
        if phrase.lower() in lower:
            return True
    return False


def fallback_observation(action_type: str, action_args: str) -> str:
    args = action_args.strip().replace("\n", " ")
    if len(args) > 200:
        args = args[:200] + "..."
    if args:
        return f"{action_type} returned: {args}"
    return f"{action_type} executed successfully."


def simulate_observation(
    query: str,
    tools: List[Dict[str, Any]],
    action: str,
    action_type: str,
    action_args: str,
    scratchpad: str,
    sim_client: LLMClient,
    sim_limiter: QPMRateLimiter,
    prompt_cfg: Dict[str, Any],
    generation_cfg: Dict[str, Any],
    validation_cfg: Dict[str, Any],
) -> str:
    sim_timeout = float(generation_cfg.get("simulator_request_timeout_seconds", 120.0))

    scratchpad_tail = tail_truncate(
        text=scratchpad,
        max_chars=int(generation_cfg.get("simulator_context_max_chars", 12000)),
    )

    user_template = str(prompt_cfg.get("tool_simulator_user_template", ""))
    sim_user = user_template.format(
        query=query,
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2),
        action=action,
        action_type=action_type,
        action_args=action_args,
        scratchpad=scratchpad_tail,
    )

    sim_messages = [
        sim_client.build_text_message(
            "system",
            str(prompt_cfg.get("tool_simulator_system_prompt", "You simulate tool output.")),
        ),
        sim_client.build_text_message("user", sim_user),
    ]

    reject_phrases = [
        str(x)
        for x in validation_cfg.get("reject_phrases", [])
        if isinstance(x, str) and x.strip()
    ]

    for _ in range(int(generation_cfg.get("simulator_max_retries", 2))):
        text = call_model_text(
            llm_client=sim_client,
            messages=sim_messages,
            limiter=sim_limiter,
            max_retries=1,
            max_tokens=int(generation_cfg.get("simulator_max_tokens", 1024)),
            temperature=float(generation_cfg.get("simulator_temperature", 0.2)),
            timeout=sim_timeout,
        )
        if text and (not reject_phrases or not contains_reject_phrase(text, reject_phrases)):
            return text

        if text:
            rewrite = str(prompt_cfg.get("tool_simulator_rewrite_prompt", "{bad_observation}"))
            rewrite_user = rewrite.format(bad_observation=text)
            rewrite_messages = [
                sim_client.build_text_message(
                    "system",
                    str(prompt_cfg.get("tool_simulator_system_prompt", "You simulate tool output.")),
                ),
                sim_client.build_text_message("user", rewrite_user),
            ]
            rewrite_text = call_model_text(
                llm_client=sim_client,
                messages=rewrite_messages,
                limiter=sim_limiter,
                max_retries=1,
                max_tokens=int(generation_cfg.get("simulator_max_tokens", 1024)),
                temperature=float(generation_cfg.get("simulator_temperature", 0.2)),
                timeout=sim_timeout,
            )
            if rewrite_text and (not reject_phrases or not contains_reject_phrase(rewrite_text, reject_phrases)):
                return rewrite_text

    return fallback_observation(action_type=action_type, action_args=action_args)


def generate_direct_sample(
    item: Dict[str, Any],
    query: str,
    skills: List[str],
    tools: List[Dict[str, Any]],
    use_split: bool,
    main_client: LLMClient,
    main_limiter: QPMRateLimiter,
    generation_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    pack = get_qa_message_not_react(
        skills=skills,
        query=query,
        need_system_prompt=bool(generation_cfg.get("need_system_prompt_for_direct", True)),
    )
    messages: List[Dict[str, Any]] = list(pack["messages"])
    skill_map_raw = pack.get("skill_map", {})
    skill_map = skill_map_raw if isinstance(skill_map_raw, dict) else {}
    model_messages = materialize_messages_with_skills(messages=messages, skill_map=skill_map)
    main_timeout = float(generation_cfg.get("main_request_timeout_seconds", 180.0))

    assistant_text = call_model_text(
        llm_client=main_client,
        messages=model_messages,
        limiter=main_limiter,
        max_retries=int(generation_cfg.get("main_max_retries", 2)),
        max_tokens=int(generation_cfg.get("main_max_tokens", 4096)),
        temperature=float(generation_cfg.get("main_temperature", 0.2)),
        timeout=main_timeout,
    )
    if not assistant_text:
        return None

    messages.append({"role": "assistant", "content": assistant_text})

    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "owner": item.get("owner"),
        "repo": item.get("repo"),
        "mode": "direct",
        "query": query,
        "skill_map": skill_map,
        "messages_list": [messages],
        "meta": {
            "use_split_skills": use_split,
            "tool_count": len(tools),
            "react_style": "single_turn",
        },
    }


def generate_react_sample(
    item: Dict[str, Any],
    query: str,
    skills: List[str],
    tools: List[Dict[str, Any]],
    use_split: bool,
    main_client: LLMClient,
    sim_client: LLMClient,
    main_limiter: QPMRateLimiter,
    sim_limiter: QPMRateLimiter,
    generation_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    validation_cfg: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    pack = get_qa_tool_prompt_react(skills=skills, query=query, tools=tools)
    skill_map_raw = pack.get("skill_map", {})
    skill_map = skill_map_raw if isinstance(skill_map_raw, dict) else {}

    system_prompt_template = str(pack["system_instruction"])
    system_prompt_model = render_skill_placeholders(system_prompt_template, skill_map)
    base_user_prefix_template = split_react_user_prefix(str(pack["user_prompt"]))
    base_user_prefix_model = render_skill_placeholders(base_user_prefix_template, skill_map)
    scratchpad = ""
    messages_list: List[List[Dict[str, Any]]] = []

    max_steps = int(generation_cfg.get("react_max_steps", 8))
    main_timeout = float(generation_cfg.get("main_request_timeout_seconds", 180.0))
    action_pattern = str(
        validation_cfg.get("action_pattern", r"^([A-Za-z_][A-Za-z0-9_]*)\\[(.*)\\]$")
    )
    obs_template = str(
        prompt_cfg.get("observation_message_template", "Observation {step}: {observation}\\nThought {next_step}:")
    )

    for step in range(1, max_steps + 1):
        user_prompt_template = build_react_user_prompt(
            base_user_prefix=base_user_prefix_template,
            scratchpad=scratchpad,
            step=step,
        )
        user_prompt_model = build_react_user_prompt(
            base_user_prefix=base_user_prefix_model,
            scratchpad=scratchpad,
            step=step,
        )

        step_messages_template: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": user_prompt_template},
        ]
        step_messages_model: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_model},
            {"role": "user", "content": user_prompt_model},
        ]

        assistant_text = call_model_text(
            llm_client=main_client,
            messages=step_messages_model,
            limiter=main_limiter,
            max_retries=int(generation_cfg.get("main_max_retries", 2)),
            max_tokens=int(generation_cfg.get("main_max_tokens", 4096)),
            temperature=float(generation_cfg.get("main_temperature", 0.2)),
            timeout=main_timeout,
        )
        if not assistant_text:
            return None, "react_empty_or_main_fail"

        parsed = parse_react_turn(
            text=assistant_text,
            step=step,
            action_pattern=action_pattern,
        )
        if parsed is None:
            return None, "react_parse_failed"

        assistant_msg = f"Thought {step}: {parsed.thought}\nAction {step}: {parsed.action}"
        step_messages_template.append({"role": "assistant", "content": assistant_msg})
        messages_list.append(step_messages_template)

        if parsed.action_type == "Finish":
            return {
                "id": item.get("id"),
                "name": item.get("name"),
                "owner": item.get("owner"),
                "repo": item.get("repo"),
                "mode": "react",
                "query": query,
                "tools": tools,
                "skill_map": skill_map,
                "messages_list": messages_list,
                "meta": {
                    "use_split_skills": use_split,
                    "n_steps": step,
                    "final_answer": parsed.action_args,
                    "react_style": "pseudo_multi_turn",
                },
            }, ""

        observation = simulate_observation(
            query=query,
            tools=tools,
            action=parsed.action,
            action_type=parsed.action_type,
            action_args=parsed.action_args,
            scratchpad=scratchpad + f"\nThought {step}: {parsed.thought}\nAction {step}: {parsed.action}",
            sim_client=sim_client,
            sim_limiter=sim_limiter,
            prompt_cfg=prompt_cfg,
            generation_cfg=generation_cfg,
            validation_cfg=validation_cfg,
        )

        scratchpad += (
            f"\nThought {step}: {parsed.thought}"
            f"\nAction {step}: {parsed.action}"
            f"\nObservation {step}: {observation}"
        )

    _ = obs_template

    return None, "react_max_steps_exceeded"


def run_item_generation(
    work: ItemWork,
    args: argparse.Namespace,
    main_client: LLMClient,
    sim_client: LLMClient,
    main_limiter: QPMRateLimiter,
    sim_limiter: QPMRateLimiter,
    generation_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    validation_cfg: Dict[str, Any],
    file_lock: threading.Lock,
    done_direct: Set[str],
    done_react: Set[str],
    done_lock: threading.Lock,
    add_extra_prompt: str,
) -> Dict[str, int]:
    delta: Dict[str, int] = {
        "direct_query_attempts": 0,
        "react_query_attempts": 0,
        "resume_skipped_direct": 0,
        "resume_skipped_react": 0,
        "react_failed_parse": 0,
        "react_failed_other": 0,
        "direct_failed_generation": 0,
        "success_direct": 0,
        "success_react": 0,
        "skip_react_all_failed": 0,
    }

    if work.mode in {"direct", "both"}:
        for query in work.direct_queries:
            effective_query = append_extra_prompt(query=query, add_extra_prompt=add_extra_prompt)
            delta["direct_query_attempts"] += 1
            key = make_row_key(item_id=work.item_id, mode="direct", query=effective_query)

            with done_lock:
                if key in done_direct:
                    delta["resume_skipped_direct"] += 1
                    continue

            sample = generate_direct_sample(
                item=work.item,
                query=effective_query,
                skills=work.skills,
                tools=work.tools,
                use_split=work.use_split,
                main_client=main_client,
                main_limiter=main_limiter,
                generation_cfg=generation_cfg,
            )
            if sample is None:
                delta["direct_failed_generation"] += 1
                continue

            append_jsonl(args.direct_output, sample, file_lock)
            with done_lock:
                done_direct.add(key)
            delta["success_direct"] += 1

    if work.mode in {"react", "both"}:
        success = False
        for query in work.react_queries:
            effective_query = append_extra_prompt(query=query, add_extra_prompt=add_extra_prompt)
            delta["react_query_attempts"] += 1
            key = make_row_key(item_id=work.item_id, mode="react", query=effective_query)

            with done_lock:
                if key in done_react:
                    delta["resume_skipped_react"] += 1
                    success = True
                    break

            sample, err = generate_react_sample(
                item=work.item,
                query=effective_query,
                skills=work.skills,
                tools=work.tools,
                use_split=work.use_split,
                main_client=main_client,
                sim_client=sim_client,
                main_limiter=main_limiter,
                sim_limiter=sim_limiter,
                generation_cfg=generation_cfg,
                prompt_cfg=prompt_cfg,
                validation_cfg=validation_cfg,
            )
            if sample is None:
                if err == "react_parse_failed":
                    delta["react_failed_parse"] += 1
                else:
                    delta["react_failed_other"] += 1
                continue

            append_jsonl(args.react_output, sample, file_lock)
            with done_lock:
                done_react.add(key)
            delta["success_react"] += 1
            success = True
            break

        if not success:
            delta["skip_react_all_failed"] += 1

    return delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate direct/react QA training data from skill analyses.")
    parser.add_argument("--analysis-file", type=str, required=True)
    parser.add_argument("--origin-file", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, default="./generate_answer_config.json")
    parser.add_argument("--prompt-file", type=str, default="./generate_answer_prompt.json")
    parser.add_argument("--direct-output", type=str, required=True)
    parser.add_argument("--react-output", type=str, required=True)
    parser.add_argument("--main-model", type=str, default=None, help="Override main model name")
    parser.add_argument("--sim-model", type=str, default=None, help="Override simulator model name")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bar")
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config_file)
    prompt_cfg = load_json(args.prompt_file)

    decision_cfg = dict(config.get("decision", {}))
    generation_cfg = dict(config.get("generation", {}))
    rate_cfg = dict(config.get("rate_limit", {}))
    validation_cfg = dict(config.get("react_validation", {}))

    seed = int(config.get("random_seed", 42))
    rng = random.Random(seed)

    analysis_rows = load_jsonl(args.analysis_file)
    origin_rows = load_jsonl(args.origin_file)
    split_rows = load_jsonl(args.split_file)

    if args.max_samples is not None:
        analysis_rows = analysis_rows[: args.max_samples]

    if args.overwrite:
        if os.path.exists(args.direct_output):
            os.remove(args.direct_output)
        if os.path.exists(args.react_output):
            os.remove(args.react_output)

    done_direct = read_done_keys(args.direct_output) if args.resume else set()
    done_react = read_done_keys(args.react_output) if args.resume else set()

    origin_by_id = build_origin_index(origin_rows)
    split_by_parent_id = build_split_index(split_rows)

    main_model_spec = dict(config.get("main_model", {}))
    sim_model_spec = dict(config.get("simulator_model", {}))

    if args.main_model:
        main_model_spec["model"] = args.main_model
    if args.sim_model:
        sim_model_spec["model"] = args.sim_model

    main_client = build_client_from_spec(
        spec=main_model_spec,
        default_model=args.main_model,
        default_max_tokens=int(generation_cfg.get("main_max_tokens", 4096)),
        default_temperature=float(generation_cfg.get("main_temperature", 0.2)),
    )
    sim_client = build_client_from_spec(
        spec=sim_model_spec,
        default_model=args.sim_model,
        default_max_tokens=int(generation_cfg.get("simulator_max_tokens", 1024)),
        default_temperature=float(generation_cfg.get("simulator_temperature", 0.2)),
    )

    main_limiter = QPMRateLimiter(qpm=int(rate_cfg.get("main_qpm", 60)))
    sim_limiter = QPMRateLimiter(qpm=int(rate_cfg.get("simulator_qpm", 90)))

    min_quality = float(decision_cfg.get("min_quality_score", 3.0))
    split_lt = int(decision_cfg.get("split_single_if_lt", 2))
    split_gt = int(decision_cfg.get("split_single_if_gt", 6))
    split_single_p = clamp_probability(float(decision_cfg.get("split_single_probability_in_range", 0.5)))
    react_keep_p = clamp_probability(float(decision_cfg.get("react_keep_probability", 0.3)))
    direct_query_count = int(decision_cfg.get("direct_query_count", 2))
    react_not_direct = bool(decision_cfg.get("react_not_direct", True))

    add_extra_prompt_raw = generation_cfg.get("add_extra_prompt", "")
    if add_extra_prompt_raw is None:
        add_extra_prompt = ""
    elif isinstance(add_extra_prompt_raw, str):
        add_extra_prompt = add_extra_prompt_raw
    else:
        add_extra_prompt = str(add_extra_prompt_raw)

    generation_cfg["react_max_steps"] = int(decision_cfg.get("react_max_steps", generation_cfg.get("react_max_steps", 8)))

    file_lock = threading.Lock()
    done_lock = threading.Lock()

    stats: Dict[str, int] = {
        "total": 0,
        "direct_items": 0,
        "react_items": 0,
        "direct_query_attempts": 0,
        "react_query_attempts": 0,
        "split_selected_items": 0,
        "split_selected_direct_items": 0,
        "split_selected_react_items": 0,
        "skip_missing_analysis": 0,
        "skip_quality": 0,
        "skip_missing_origin": 0,
        "skip_missing_split": 0,
        "skip_mode": 0,
        "skip_react_probability": 0,
        "skip_no_queries": 0,
        "skip_react_all_failed": 0,
        "success_direct": 0,
        "success_react": 0,
        "resume_skipped_direct": 0,
        "resume_skipped_react": 0,
        "react_failed_parse": 0,
        "react_failed_other": 0,
        "direct_failed_generation": 0,
    }

    progress = SimpleProgressBar(total=len(analysis_rows), enabled=(not args.no_progress))
    completed_count = 0
    max_workers = max(1, int(args.workers))
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in analysis_rows:
            stats["total"] += 1
            analysis = item.get("analysis")
            if not isinstance(analysis, dict):
                stats["skip_missing_analysis"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            quality = analysis.get("quality_score")
            if not isinstance(quality, (int, float)) or float(quality) < min_quality:
                stats["skip_quality"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                stats["skip_missing_analysis"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            origin_doc = origin_by_id.get(item_id)
            if not origin_doc:
                stats["skip_missing_origin"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            split_count = analysis.get("split_skill_count", 0)
            use_split = False
            skills: List[str]

            if not isinstance(split_count, int):
                split_count = 0

            if split_count < split_lt or split_count > split_gt:
                skills = [origin_doc]
            else:
                if rng.random() < split_single_p:
                    skills = [origin_doc]
                else:
                    split_docs = split_by_parent_id.get(item_id)
                    if not split_docs:
                        stats["skip_missing_split"] += 1
                        completed_count += 1
                        progress.update(completed_count, stats)
                        continue
                    skills = split_docs
                    use_split = True

            if use_split:
                stats["split_selected_items"] += 1

            tools = analysis.get("tool_specs") if isinstance(analysis.get("tool_specs"), list) else []
            prefer_react = analysis.get("prefer_react")

            mode: Optional[str] = None
            if (prefer_react is False) and (not has_tools(analysis)):
                mode = "direct"
            elif (prefer_react is True) and has_tools(analysis):
                keep_react = rng.random() < react_keep_p
                if keep_react:
                    mode = "react" if react_not_direct else "both"
                else:
                    stats["skip_react_probability"] += 1
                    if react_not_direct:
                        completed_count += 1
                        progress.update(completed_count, stats)
                        continue
                    mode = "direct"
            elif (prefer_react is True) and (not has_tools(analysis)) and (not react_not_direct):
                mode = "direct"
            else:
                stats["skip_mode"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            solvable_problems = analysis.get("solvable_problems")
            if not isinstance(solvable_problems, list):
                stats["skip_no_queries"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            cleaned_queries = [
                q.strip()
                for q in solvable_problems
                if isinstance(q, str) and q.strip()
            ]
            if not cleaned_queries:
                stats["skip_no_queries"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            direct_queries = cleaned_queries[:max(0, direct_query_count)]
            react_queries = cleaned_queries

            if mode in {"direct", "both"}:
                stats["direct_items"] += 1
                if use_split:
                    stats["split_selected_direct_items"] += 1

            if mode == "direct" and not direct_queries:
                stats["skip_no_queries"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            if mode in {"react", "both"}:
                stats["react_items"] += 1
                if use_split:
                    stats["split_selected_react_items"] += 1

            if mode == "react" and not react_queries:
                stats["skip_no_queries"] += 1
                completed_count += 1
                progress.update(completed_count, stats)
                continue

            work = ItemWork(
                item=item,
                item_id=item_id,
                mode=mode,
                direct_queries=direct_queries if mode in {"direct", "both"} else [],
                react_queries=react_queries if mode in {"react", "both"} else [],
                skills=skills,
                tools=tools,
                use_split=use_split,
            )
            futures.append(
                executor.submit(
                    run_item_generation,
                    work,
                    args,
                    main_client,
                    sim_client,
                    main_limiter,
                    sim_limiter,
                    generation_cfg,
                    prompt_cfg,
                    validation_cfg,
                    file_lock,
                    done_direct,
                    done_react,
                    done_lock,
                    add_extra_prompt,
                )
            )

        for future in as_completed(futures):
            try:
                delta = future.result()
            except Exception:
                stats["react_failed_other"] += 1
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
    print(f"Simulator provider/model: {sim_client.config.provider} / {sim_client.config.model}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.summary_output:
        summary = {
            "stats": stats,
            "outputs": {
                "direct_output": args.direct_output,
                "react_output": args.react_output,
            },
            "config": {
                "decision": decision_cfg,
                "generation": generation_cfg,
            },
        }
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
