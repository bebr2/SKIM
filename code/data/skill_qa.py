from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset


def _looks_like_messages(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    return isinstance(first, dict) and "role" in first and "content" in first


def _extract_messages(obj: Any) -> list[dict[str, Any]] | None:
    if isinstance(obj, dict):
        for key in ("messages", "conversation", "conversations", "dialogue"):
            candidate = obj.get(key)
            if _looks_like_messages(candidate):
                return candidate

        if "role" in obj and "content" in obj:
            return [obj]
        return None

    if isinstance(obj, list):
        if _looks_like_messages(obj):
            return obj
        if len(obj) == 1 and _looks_like_messages(obj[0]):
            return obj[0]
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


def _extract_system_user_assistant_pair(messages: list[dict[str, Any]]) -> tuple[str, str, str] | None:
    system_chunks: list[str] = []
    user_text: str | None = None
    assistant_text: str | None = None
    seen_user = False

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = _content_to_text(message.get("content", ""))

        if role == "system" and assistant_text is None:
            system_chunks.append(content)
            continue

        if role == "user" and user_text is None:
            user_text = content
            seen_user = True
            continue

        if role == "assistant" and seen_user:
            assistant_text = content
            break

    if user_text is None or assistant_text is None:
        return None
    return "\n\n".join(x for x in system_chunks if x.strip()), user_text, assistant_text


@dataclass
class SkillQASample:
    system_content: str
    user_content: str
    assistant_content: str
    metadata: dict[str, Any]
    rejected_assistant_content: str | None = None
    has_rejected_pair: bool = False


@dataclass
class SkillQASource:
    path: str
    name: str
    size: int
    enabled: bool


@dataclass
class SkillQASourceStat:
    name: str
    path: str
    loaded_samples: int
    selected_samples: int
    requested_size: int


def _coerce_source(item: Any, idx: int) -> SkillQASource:
    if isinstance(item, str):
        path = item.strip()
        if not path:
            raise ValueError(f"skill QA source[{idx}] path is empty")
        return SkillQASource(path=path, name=f"source_{idx}", size=0, enabled=True)

    path = str(getattr(item, "path", "")).strip() if not isinstance(item, dict) else str(item.get("path", "")).strip()
    if not path:
        raise ValueError(f"skill QA source[{idx}] missing required field 'path'")

    if isinstance(item, dict):
        name = str(item.get("name") or f"source_{idx}").strip()
        size_raw = item.get("size", 0)
        enabled_raw = item.get("enabled", True)
    else:
        name = str(getattr(item, "name", "") or f"source_{idx}").strip()
        size_raw = getattr(item, "size", 0)
        enabled_raw = getattr(item, "enabled", True)

    size = int(size_raw) if size_raw is not None else 0
    enabled = bool(enabled_raw)
    if size < 0:
        raise ValueError(f"skill QA source[{idx}] size must be >= 0")

    return SkillQASource(path=path, name=name, size=size, enabled=enabled)


def _normalize_sources(path_or_sources: str | Sequence[Any]) -> list[SkillQASource]:
    if isinstance(path_or_sources, str):
        return [_coerce_source(path_or_sources, 0)]

    sources: list[SkillQASource] = []
    for idx, item in enumerate(path_or_sources):
        src = _coerce_source(item, idx)
        if src.enabled:
            sources.append(src)
    return sources


def _stable_seed(seed: int, name: str, path: str) -> int:
    digest = hashlib.md5(f"{name}|{path}".encode("utf-8")).hexdigest()[:8]
    return seed + int(digest, 16)


def _sample_indices(total: int, size: int, rng: random.Random) -> list[int]:
    if size <= 0 or size >= total:
        return list(range(total))
    return rng.sample(range(total), size)


def _copy_skill_map(skill_map: Any) -> dict[str, Any]:
    if not isinstance(skill_map, dict):
        return {}
    copied: dict[str, Any] = {}
    for key, value in skill_map.items():
        copied[str(key)] = value
    return copied


def _build_pair_key(record_id: str, mode: str, query: str) -> str:
    return f"{record_id}||{mode}||{query}"


def _pair_key_from_meta(meta: dict[str, Any]) -> str | None:
    record_id = str(meta.get("record_id", "")).strip()
    mode = str(meta.get("mode", "")).strip()
    query = str(meta.get("query", "")).strip()
    if not record_id or not mode or not query:
        return None
    return _build_pair_key(record_id=record_id, mode=mode, query=query)


def _collect_conversations_with_meta(
    obj: Any,
    out: list[tuple[list[dict[str, Any]], dict[str, Any]]],
    inherited_meta: dict[str, Any],
) -> None:
    if isinstance(obj, dict):
        local_meta = dict(inherited_meta)
        if "id" in obj:
            local_meta["record_id"] = str(obj.get("id"))
        if "mode" in obj:
            local_meta["mode"] = str(obj.get("mode"))
        if "name" in obj:
            local_meta["record_name"] = str(obj.get("name"))
        if "query" in obj:
            local_meta["query"] = str(obj.get("query"))

        local_skill_map = _copy_skill_map(obj.get("skill_map"))
        if local_skill_map:
            local_meta["skill_map"] = local_skill_map

        messages = _extract_messages(obj)
        if messages is not None:
            out.append((messages, dict(local_meta)))

        messages_list = obj.get("messages_list")
        if isinstance(messages_list, list):
            for round_index, candidate in enumerate(messages_list):
                candidate_messages = _extract_messages(candidate)
                if candidate_messages is None:
                    continue
                round_meta = dict(local_meta)
                round_meta["round_index"] = round_index
                out.append((candidate_messages, round_meta))

        for key in ("data", "items", "records", "samples"):
            nested = obj.get(key)
            if nested is not None:
                _collect_conversations_with_meta(nested, out, local_meta)
        return

    if isinstance(obj, list):
        if _looks_like_messages(obj):
            out.append((obj, dict(inherited_meta)))
            return
        if len(obj) == 1 and _looks_like_messages(obj[0]):
            out.append((obj[0], dict(inherited_meta)))
            return
        for item in obj:
            _collect_conversations_with_meta(item, out, inherited_meta)


def _load_source_samples(
    source: SkillQASource,
    source_idx: int,
    seed: int,
) -> tuple[list[SkillQASample], SkillQASourceStat]:
    collected: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    if source.path.endswith(".jsonl"):
        with open(source.path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                _collect_conversations_with_meta(
                    row,
                    collected,
                    {
                        "source_name": source.name,
                        "source_path": source.path,
                        "source_index": source_idx,
                    },
                )
    else:
        with open(source.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        _collect_conversations_with_meta(
            payload,
            collected,
            {
                "source_name": source.name,
                "source_path": source.path,
                "source_index": source_idx,
            },
        )

    source_samples: list[SkillQASample] = []
    for local_idx, (messages, row_meta) in enumerate(collected):
        pair = _extract_system_user_assistant_pair(messages)
        if pair is None:
            continue
        system_text, user_text, assistant_text = pair
        metadata = dict(row_meta)
        metadata["source_sample_index"] = local_idx
        metadata["turns"] = len(messages)
        metadata["skill_map"] = _copy_skill_map(metadata.get("skill_map"))
        source_samples.append(
            SkillQASample(
                system_content=system_text,
                user_content=user_text,
                assistant_content=assistant_text,
                metadata=metadata,
            )
        )

    loaded_count = len(source_samples)
    request_size = int(source.size)
    if request_size > loaded_count:
        print(
            (
                f"[skill_qa][data] source={source.name} requested size={request_size} "
                f"> available={loaded_count}; truncate to full dataset"
            ),
            flush=True,
        )

    rng = random.Random(_stable_seed(seed, source.name, source.path))
    selected_indices = _sample_indices(loaded_count, request_size, rng)
    selected_samples = [source_samples[selected_local_idx] for selected_local_idx in selected_indices]

    stat = SkillQASourceStat(
        name=source.name,
        path=source.path,
        loaded_samples=loaded_count,
        selected_samples=len(selected_indices),
        requested_size=request_size,
    )
    return selected_samples, stat


class SkillQADataset(Dataset):
    def __init__(
        self,
        path_or_sources: str | Sequence[Any],
        seed: int = 42,
        rejected_path_or_sources: str | Sequence[Any] | None = None,
    ) -> None:
        self.samples: list[SkillQASample] = []
        self.source_stats: list[SkillQASourceStat] = []

        sources = _normalize_sources(path_or_sources)
        if not sources:
            raise ValueError("No valid skill QA data source is configured")

        for source_idx, source in enumerate(sources):
            selected_samples, source_stat = _load_source_samples(
                source=source,
                source_idx=source_idx,
                seed=seed,
            )
            for sample in selected_samples:
                sample.metadata["sample_index"] = len(self.samples)
                self.samples.append(sample)
            self.source_stats.append(source_stat)

        rejected_by_key: dict[str, str] = {}
        if rejected_path_or_sources is not None:
            rejected_sources = _normalize_sources(rejected_path_or_sources)
            for rejected_idx, rejected_source in enumerate(rejected_sources):
                rejected_selected_samples, _ = _load_source_samples(
                    source=rejected_source,
                    source_idx=len(sources) + rejected_idx,
                    seed=seed + 200_000,
                )
                for sample in rejected_selected_samples:
                    pair_key = _pair_key_from_meta(sample.metadata)
                    if not pair_key:
                        continue
                    if pair_key not in rejected_by_key:
                        rejected_by_key[pair_key] = sample.assistant_content

        for sample in self.samples:
            pair_key = _pair_key_from_meta(sample.metadata)
            sample.metadata["pair_key"] = pair_key or ""

            if pair_key and pair_key in rejected_by_key:
                sample.rejected_assistant_content = rejected_by_key[pair_key]
                sample.has_rejected_pair = True
            else:
                sample.rejected_assistant_content = None
                sample.has_rejected_pair = False

            sample.metadata["has_rejected_pair"] = bool(sample.has_rejected_pair)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SkillQASample:
        return self.samples[idx]
