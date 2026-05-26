from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_ROOT = _REPO_ROOT / "code"
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from skillrag_vendor.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge skill inference results into prepare-style skill_qa training JSONL "
            "(messages_list + skill_map)."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Model folder name under results/<dataset>/<model>")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names, supports comma or space separated list",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="Result file name under each dataset/model, e.g. golden_skill.jsonl",
    )

    parser.add_argument("--results-root", type=str, default="./results", help="Root folder of skill results")
    parser.add_argument("--instances-dir", type=str, default="", help="Directory containing <dataset>.json")
    parser.add_argument(
        "--skillrag-root",
        type=str,
        default="",
        help="SkillRAG root; if provided, fallback instances dir is <root>/data/bench/instances",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=str(_REPO_ROOT / "prepare" / "output" / "corpus.json"),
        help="Skill corpus path (json or jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Merged output jsonl path; default prepare/output/<model>_<stem>_prepare.jsonl",
    )

    parser.add_argument(
        "--allow-missing-instances",
        action="store_true",
        help="Skip rows whose instance_id cannot be found instead of raising",
    )
    parser.add_argument(
        "--allow-missing-skills",
        action="store_true",
        help="Allow rows with zero resolved skills (note: skill_qa usually requires at least one <skill> tag)",
    )
    parser.add_argument(
        "--keep-empty-answer",
        action="store_true",
        help="Keep rows with empty raw_output",
    )
    return parser.parse_args()


def _parse_dataset_list(raw_items: list[str]) -> list[str]:
    datasets: list[str] = []
    for item in raw_items:
        for chunk in item.split(","):
            name = chunk.strip()
            if name:
                datasets.append(name)
    return datasets


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


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
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    mapping: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() == ".jsonl":
        rows = _load_jsonl(path)
        for row in rows:
            skill_id = str(row.get("skill_id", "")).strip()
            if skill_id:
                mapping[skill_id] = row
        return mapping

    payload = _load_json(path)
    items: list[dict[str, Any]] = []
    _collect_skill_items(payload, items)
    for item in items:
        skill_id = str(item.get("skill_id", "")).strip()
        if skill_id:
            mapping[skill_id] = item
    return mapping


def _instance_path_candidates(dataset: str, args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []

    if args.instances_dir.strip():
        candidates.append(Path(args.instances_dir) / f"{dataset}.json")

    if args.skillrag_root.strip():
        candidates.append(Path(args.skillrag_root) / "data" / "bench" / "instances" / f"{dataset}.json")

    candidates.extend(
        [
            _REPO_ROOT / "prepare" / "output" / "trainset" / f"{dataset}.json",
            _REPO_ROOT / "prepare" / "output" / "instances" / f"{dataset}.json",
            _REPO_ROOT / "prepare" / "output" / "trainset" / dataset / "instances.json",
            _REPO_ROOT / "prepare" / "output" / "trainset" / dataset / "instances" / f"{dataset}.json",
            _REPO_ROOT / "skillrag" / "data" / "bench" / "instances" / f"{dataset}.json",
        ]
    )

    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def resolve_instances_file(dataset: str, args: argparse.Namespace) -> Path:
    for candidate in _instance_path_candidates(dataset, args):
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Cannot resolve instances file for dataset "
        f"'{dataset}'. Tried: {[str(p) for p in _instance_path_candidates(dataset, args)]}"
    )


def load_instances_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Instances file must be a JSON list: {path}")

    mapping: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id", "")).strip()
        if instance_id:
            mapping[instance_id] = row
    return mapping


def _resolve_skill_map_and_user(
    base_user: str,
    skill_ids: list[str],
    corpus: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any], list[str]]:
    valid_ids: list[str] = []
    skill_map: dict[str, Any] = {}

    for sid in skill_ids:
        clean = str(sid).strip()
        if not clean:
            continue
        row = corpus.get(clean)
        if row is None:
            continue
        valid_ids.append(clean)
        skill_map[clean] = row

    if not valid_ids:
        return base_user, skill_map, valid_ids

    skill_block = "\n---\n".join([f"<skill>{sid}.content</skill>" for sid in valid_ids])
    user_text = f"Relevant Skill:\n{skill_block}\n\n{base_user}"
    return user_text, skill_map, valid_ids


def _as_list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _infer_mode(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "toolqa":
        return "react"
    if any(k in row for k in ("n_steps", "finished", "halted")):
        return "react"
    return "direct"


def merge_dataset_rows(
    dataset: str,
    result_rows: list[dict[str, Any]],
    instances_map: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out_rows: list[dict[str, Any]] = []
    stats = {
        "total": len(result_rows),
        "missing_instance": 0,
        "missing_answer": 0,
        "missing_skill": 0,
        "merged": 0,
    }

    for row in result_rows:
        instance_id = str(row.get("instance_id", "")).strip()
        instance = instances_map.get(instance_id)
        if instance is None:
            stats["missing_instance"] += 1
            if not args.allow_missing_instances:
                raise KeyError(f"Instance not found for dataset={dataset}, instance_id={instance_id}")
            continue

        raw_output = str(row.get("raw_output", "")).strip()
        if not raw_output and not args.keep_empty_answer:
            stats["missing_answer"] += 1
            continue

        system_text, base_user = build_prompt(instance, method="naive")

        skill_ids = _as_list_of_strings(row.get("skill_ids_used"))
        user_text, skill_map, resolved_ids = _resolve_skill_map_and_user(base_user, skill_ids, corpus)

        if not resolved_ids and not args.allow_missing_skills:
            stats["missing_skill"] += 1
            continue

        messages: list[dict[str, Any]] = []
        if system_text.strip():
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": raw_output})

        out_row: dict[str, Any] = {
            "id": instance_id,
            "name": dataset,
            "owner": "skillrag",
            "repo": dataset,
            "mode": _infer_mode(dataset, row),
            "query": str(instance.get("question", "")),
            "skill_map": skill_map,
            "messages_list": [messages],
            "meta": {
                "source": "skill_results_merge",
                "instance_id": instance_id,
                "dataset": dataset,
                "model": str(row.get("model", "")) or args.model,
                "method": str(row.get("method", "")),
                "n_skills": len(resolved_ids),
            },
        }

        for key in ("n_steps", "finished", "halted"):
            if key in row:
                out_row["meta"][key] = row.get(key)

        out_rows.append(out_row)
        stats["merged"] += 1

    return out_rows, stats


def _default_output_path(args: argparse.Namespace) -> Path:
    stem = Path(args.result_file).stem
    return _REPO_ROOT / "prepare" / "output" / f"{args.model}_{stem}_prepare.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    datasets = _parse_dataset_list(args.datasets)
    if not datasets:
        raise ValueError("No dataset is provided")

    corpus = load_skill_corpus(args.corpus_path)

    merged_all: list[dict[str, Any]] = []
    total_stats: dict[str, int] = {
        "datasets": len(datasets),
        "rows_total": 0,
        "rows_merged": 0,
        "missing_instance": 0,
        "missing_answer": 0,
        "missing_skill": 0,
    }

    results_root = Path(args.results_root)

    for dataset in datasets:
        result_path = results_root / dataset / args.model / args.result_file
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")

        instances_path = resolve_instances_file(dataset, args)
        instances_map = load_instances_map(instances_path)
        result_rows = _load_jsonl(result_path)

        merged_rows, ds_stats = merge_dataset_rows(
            dataset=dataset,
            result_rows=result_rows,
            instances_map=instances_map,
            corpus=corpus,
            args=args,
        )
        merged_all.extend(merged_rows)

        total_stats["rows_total"] += ds_stats["total"]
        total_stats["rows_merged"] += ds_stats["merged"]
        total_stats["missing_instance"] += ds_stats["missing_instance"]
        total_stats["missing_answer"] += ds_stats["missing_answer"]
        total_stats["missing_skill"] += ds_stats["missing_skill"]

        print(
            f"[{dataset}] total={ds_stats['total']} merged={ds_stats['merged']} "
            f"missing_instance={ds_stats['missing_instance']} "
            f"missing_answer={ds_stats['missing_answer']} missing_skill={ds_stats['missing_skill']}"
        )

    output_path = Path(args.output) if args.output.strip() else _default_output_path(args)
    _write_jsonl(output_path, merged_all)

    print(f"Merged rows: {len(merged_all)}")
    print(f"Output: {output_path}")
    print(
        "Summary: "
        f"datasets={total_stats['datasets']}, "
        f"rows_total={total_stats['rows_total']}, rows_merged={total_stats['rows_merged']}, "
        f"missing_instance={total_stats['missing_instance']}, "
        f"missing_answer={total_stats['missing_answer']}, "
        f"missing_skill={total_stats['missing_skill']}"
    )


if __name__ == "__main__":
    main()
