from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run answer generation via skill/run_skill_compiler_inference.py for naive/full_text/"
            "multiple compress-k modes."
        )
    )
    parser.add_argument("--config", type=str, default="./exam/config.infer.example.json")
    parser.add_argument("--fp16", type=str, default="false", help="Use FP16/BF16 precision (true/false)")
    return parser.parse_args()


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_non_empty_strings(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if str(x).strip()]


def _parse_datasets(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("datasets") if "datasets" in cfg else cfg.get("dataset")
    if isinstance(raw, str):
        datasets = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, list):
        datasets = _as_non_empty_strings(raw)
    else:
        datasets = []
    if not datasets:
        raise ValueError("dataset/datasets is required")
    return datasets


def _build_runs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    infer_cfg = cfg.get("inference", {})
    runs: list[dict[str, Any]] = []

    if bool(infer_cfg.get("do_naive", True)):
        runs.append({"name": "naive", "method": "naive", "skill_mode": "naive", "k": None})

    if bool(infer_cfg.get("do_full_text", True)):
        runs.append({"name": "full_text", "method": "golden_skill", "skill_mode": "full_text", "k": None})

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
        runs.append({"name": f"compress_k{k}", "method": "golden_skill", "skill_mode": "compress", "k": k})

    if not runs:
        raise ValueError("No inference runs configured")
    return runs


def _run_cmd(cmd: list[str]) -> None:
    pretty = " ".join([f'"{x}"' if " " in x else x for x in cmd])
    print(f"[infer] cmd: {pretty}")
    subprocess.run(cmd, check=True)


def run(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    datasets = _parse_datasets(cfg)
    runs = _build_runs(cfg)

    checkpoint = str(cfg.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("checkpoint is required")

    runner_script = _resolve_path(str(cfg.get("runner_script", "./skill/run_skill_compiler_inference.py")))
    python_executable = str(cfg.get("python_executable", "")).strip() or sys.executable
    instances_dir = _resolve_path(str(cfg.get("instances_dir", "./exam/output/instances")))
    corpus_path = _resolve_path(str(cfg.get("corpus_path", "./prepare/output/corpus.json")))
    output_root = _resolve_path(str(cfg.get("output_root", "./exam/output/answers")))
    toolqa_data_dir = cfg.get("toolqa_data_dir", "")  # optional, required when dataset=toolqa
    output_root.mkdir(parents=True, exist_ok=True)

    infer_cfg = cfg.get("inference", {})
    run_flags = cfg.get("run_flags", {})

    run_manifest: list[dict[str, Any]] = []

    # Build all tasks first for progress bar
    all_tasks: list[tuple[str, dict[str, Any]]] = []
    for dataset in datasets:
        for spec in runs:
            all_tasks.append((dataset, spec))

    total_tasks = len(all_tasks)
    bar = tqdm(total=total_tasks, desc="Inference Progress", dynamic_ncols=True) if tqdm else None

    for dataset, spec in all_tasks:
            mode_name = str(spec["name"])
            result_path = output_root / dataset / f"{mode_name}.jsonl"
            result_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                python_executable,
                str(runner_script),
                "--checkpoint",
                checkpoint,
                "--dataset",
                dataset,
                "--instances_dir",
                str(instances_dir),
                "--corpus_path",
                str(corpus_path),
                "--result_path",
                str(result_path),
                "--method",
                str(spec["method"]),
                "--skill_mode",
                str(spec["skill_mode"]),
                "--max_length",
                str(int(infer_cfg.get("max_length", 4096) or 4096)),
                "--max_new_tokens",
                str(int(infer_cfg.get("max_new_tokens", 1024) or 1024)),
            ]

            # Forward fp16 flag - bf16 auto-enabled for toolqa dataset in skill compiler
            fp16_value = str(infer_cfg.get("fp16", args.fp16) or "false")
            cmd.extend(["--fp16", fp16_value])

            k = spec.get("k")
            if k is not None:
                cmd.extend(["--k", str(int(k))])

            if bool(infer_cfg.get("do_sample", False)):
                cmd.append("--do_sample")
                cmd.extend(["--temperature", str(float(infer_cfg.get("temperature", 0.7) or 0.7))])
                cmd.extend(["--top_p", str(float(infer_cfg.get("top_p", 0.95) or 0.95))])

            if bool(run_flags.get("disable_parallel", False)):
                cmd.append("--disable_parallel")
            if bool(run_flags.get("resume", False)):
                cmd.append("--resume")
            elif bool(run_flags.get("overwrite", True)):
                # Only pass --overwrite when not resuming
                cmd.append("--overwrite")

            # Pass toolqa_data_dir if dataset is toolqa and the path is set
            if dataset == "toolqa" and toolqa_data_dir:
                cmd.extend(["--toolqa_data_dir", str(toolqa_data_dir)])

            _run_cmd(cmd)
            if bar is not None:
                bar.set_postfix_str(f"{dataset}/{mode_name}")
                bar.update(1)
            run_manifest.append(
                {
                    "dataset": dataset,
                    "mode": mode_name,
                    "result_path": str(result_path),
                    "method": spec["method"],
                    "skill_mode": spec["skill_mode"],
                    "k": k,
                }
            )

    manifest_path = _resolve_path(str(cfg.get("manifest_json", "./exam/output/answers_manifest.json")))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    if bar is not None:
        bar.close()

    print(f"[infer] wrote manifest -> {manifest_path}")


def main() -> None:
    args = parse_args()
    cfg = _load_json(_resolve_path(args.config))
    run(cfg, args)


if __name__ == "__main__":
    main()
