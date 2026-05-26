from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Analyze eval results by mode and with exam-based skill routing "
			"for accuracy and extra context token cost."
		)
	)
	parser.add_argument("--tokenizer_path", type=str, required=True)
	parser.add_argument("--eval_result_map_config_path", type=str, required=True)
	parser.add_argument("--dataset_name", type=str, default="")
	parser.add_argument("--corpus_path", type=str, required=True)
	parser.add_argument("--exam_summary_path", type=str, required=True)
	parser.add_argument(
		"--compress_over_naive_threshold",
		type=float,
		default=0.0,
		help=(
			"For thresholded exam filter: if selected compress mode does not exceed "
			"naive accuracy by this threshold, force full_text."
		),
	)
	parser.add_argument(
		"--compress_accuracy_threshold",
		type=float,
		default=1.0,
		help=(
			"For compress_full_text_only mode: if compress accuracy >= this threshold, "
			"accept the most compressed mode. Default 1.0 means compress must >= full_text accuracy."
		),
	)
	parser.add_argument(
		"--skip_count_threshold",
		type=int,
		default=-1,
		help=(
			"If a skill's full_text skip_count exceeds this threshold, "
			"fallback to compress_k512 (the largest compress mode). "
			"Default -1 means disabled. Set to 0 to force largest_compress for all skills with skip_count > 0."
		),
	)
	parser.add_argument("--output_json", type=str, default="")
	return parser.parse_args()


def _resolve_path(path_like: str) -> Path:
	p = Path(path_like)
	if p.is_absolute():
		return p
	return (_REPO_ROOT / p).resolve()


def _load_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _collect_skill_items(obj: Any, out: list[dict[str, Any]]) -> None:
	if isinstance(obj, dict):
		has_skill_id = "skill_id" in obj
		has_compatible_id = "id" in obj and (
			"content" in obj or "description" in obj or "name" in obj
		)
		if has_skill_id or has_compatible_id:
			out.append(obj)
		for value in obj.values():
			if isinstance(value, (dict, list)):
				_collect_skill_items(value, out)
		return

	if isinstance(obj, list):
		for item in obj:
			_collect_skill_items(item, out)


def _resolve_skill_id(row: dict[str, Any]) -> str:
	sid = str(row.get("skill_id", "")).strip()
	if sid:
		return sid
	return str(row.get("id", "")).strip()


def load_skill_corpus(path_like: str) -> dict[str, dict[str, Any]]:
	corpus_path = _resolve_path(path_like)
	if not corpus_path.exists():
		raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

	mapping: dict[str, dict[str, Any]] = {}
	if corpus_path.suffix.lower() == ".jsonl":
		with open(corpus_path, "r", encoding="utf-8") as f:
			for line in f:
				raw = line.strip()
				if not raw:
					continue
				row = json.loads(raw)
				if not isinstance(row, dict):
					continue
				sid = _resolve_skill_id(row)
				if not sid:
					continue
				fixed = dict(row)
				fixed["skill_id"] = sid
				mapping[sid] = fixed
	else:
		payload = _load_json(corpus_path)
		items: list[dict[str, Any]] = []
		_collect_skill_items(payload, items)
		for row in items:
			sid = _resolve_skill_id(row)
			if not sid:
				continue
			fixed = dict(row)
			fixed["skill_id"] = sid
			mapping[sid] = fixed

	if not mapping:
		raise ValueError(f"No skills found in corpus: {corpus_path}")
	return mapping


def _extract_skill_text(row: dict[str, Any]) -> str:
	for key in ["content", "description", "name"]:
		if key not in row:
			continue
		value = row.get(key)
		if isinstance(value, str):
			return value
		if value is not None:
			return json.dumps(value, ensure_ascii=False)
	return ""


def _is_skill_for_dataset(skill_id: str, row: dict[str, Any], dataset_name: str) -> bool:
	sid = skill_id.strip()
	prefix = f"{dataset_name}_"
	if sid.startswith(prefix):
		return True

	dataset_field = row.get("dataset")
	if isinstance(dataset_field, str) and dataset_field.strip() == dataset_name:
		return True

	tags = row.get("datasets")
	if isinstance(tags, list):
		for item in tags:
			if str(item).strip() == dataset_name:
				return True

	return False


def build_full_token_map(
	tokenizer_path: str,
	corpus_path: str,
	dataset_name: str,
) -> dict[str, int]:
	try:
		from transformers import AutoTokenizer  # type: ignore
	except Exception as exc:
		raise RuntimeError(
			"transformers is required for token counting. Install it in your environment."
		) from exc

	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	corpus = load_skill_corpus(corpus_path)

	token_map: dict[str, int] = {}
	for sid, row in corpus.items():
		if not _is_skill_for_dataset(sid, row, dataset_name):
			continue
		text = _extract_skill_text(row)
		token_map[sid] = len(tokenizer.encode(text, add_special_tokens=False))

	if not token_map:
		raise ValueError(
			f"No skills from dataset '{dataset_name}' were found in corpus: {_resolve_path(corpus_path)}"
		)
	return token_map


def _parse_eval_result_map_config(
	config_path: str,
	dataset_override: str,
) -> tuple[str, dict[str, Path]]:
	cfg = _load_json(_resolve_path(config_path))
	if not isinstance(cfg, dict):
		raise ValueError("eval_result_map config must be a JSON object")

	dataset_name = dataset_override.strip()
	if not dataset_name:
		dataset_name = str(cfg.get("dataset_name") or cfg.get("dataset") or "").strip()

	eval_map_obj = cfg.get("eval_result_map")

	datasets_obj = cfg.get("datasets")
	if isinstance(datasets_obj, dict) and dataset_name and dataset_name in datasets_obj:
		dataset_cfg = datasets_obj.get(dataset_name)
		if isinstance(dataset_cfg, dict):
			eval_map_obj = dataset_cfg.get("eval_result_map", eval_map_obj)

	if not dataset_name:
		raise ValueError("dataset_name is required (CLI --dataset_name or config.dataset_name)")
	if not isinstance(eval_map_obj, dict):
		raise ValueError("eval_result_map must be an object mapping mode -> path")

	resolved: dict[str, Path] = {}
	for mode_raw, path_raw in eval_map_obj.items():
		mode = str(mode_raw).strip()
		path_like = str(path_raw).strip()
		if not mode or not path_like:
			continue
		path = _resolve_path(path_like)
		if not path.exists():
			raise FileNotFoundError(f"Eval file not found for mode '{mode}': {path}")
		resolved[mode] = path

	if "naive" not in resolved:
		raise ValueError("eval_result_map must contain mode 'naive'")
	if "full_text" not in resolved:
		raise ValueError("eval_result_map must contain mode 'full_text'")
	return dataset_name, resolved


def _normalize_skill_annotations(raw: Any) -> list[str]:
	if not isinstance(raw, list):
		return []
	out: list[str] = []
	for item in raw:
		sid = str(item).strip()
		if sid:
			out.append(sid)
	return out


def _row_belongs_to_dataset(row: dict[str, Any], dataset_name: str) -> bool:
	dataset_field = row.get("dataset")
	if isinstance(dataset_field, str) and dataset_field.strip() == dataset_name:
		return True

	instance_id = str(row.get("instance_id", "")).strip()
	if instance_id.startswith(f"{dataset_name}_"):
		return True

	anns = _normalize_skill_annotations(row.get("skill_annotations"))
	if any(sid.startswith(f"{dataset_name}_") for sid in anns):
		return True

	# If no robust dataset signal is present, keep the row for compatibility.
	return not instance_id and not anns and not isinstance(dataset_field, str)


def _coerce_correct(value: Any) -> bool | None:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		if float(value) == 1.0:
			return True
		if float(value) == 0.0:
			return False
	return None


def _parse_compress_k(mode: str) -> int | None:
	matched = re.search(r"compress[_-]?k?(\d+)$", mode)
	if not matched:
		return None
	return int(matched.group(1))


def _mode_token_cost(mode: str, skill_ids: list[str], full_token_map: dict[str, int]) -> int:
	if mode == "naive":
		return 0
	if mode == "full_text":
		return sum(int(full_token_map.get(sid, 0)) for sid in skill_ids)

	k = _parse_compress_k(mode)
	if k is None:
		raise ValueError(f"Unsupported mode name for token cost: {mode}")
	return sum(min(k, int(full_token_map.get(sid, 0))) for sid in skill_ids)


def _mode_strictness(mode: str) -> float:
	if mode == "naive":
		return 0.0
	if mode == "full_text":
		return float("inf")
	k = _parse_compress_k(mode)
	if k is None:
		return -1.0
	return float(k)


def _is_compress_mode(mode: str) -> bool:
	return _parse_compress_k(mode) is not None


def load_mode_details(
	eval_map: dict[str, Path],
	dataset_name: str,
) -> dict[str, list[dict[str, Any]]]:
	mode_details: dict[str, list[dict[str, Any]]] = {}
	for mode, path in eval_map.items():
		payload = _load_json(path)
		details = payload.get("details") if isinstance(payload, dict) else None
		if not isinstance(details, list):
			raise ValueError(f"Eval file must contain top-level 'details' list: {path}")

		filtered: list[dict[str, Any]] = []
		for row in details:
			if not isinstance(row, dict):
				continue
			if _row_belongs_to_dataset(row, dataset_name):
				filtered.append(row)
		mode_details[mode] = filtered
	return mode_details


def compute_mode_baselines(
	mode_details: dict[str, list[dict[str, Any]]],
	full_token_map: dict[str, int],
) -> dict[str, dict[str, Any]]:
	# Build full_text correct index for compress mode fallback
	full_text_correct_index: dict[str, bool | None] = {}
	for row in mode_details.get("full_text", []):
		instance_id = str(row.get("instance_id", "")).strip()
		if instance_id:
			full_text_correct_index[instance_id] = _coerce_correct(row.get("correct"))

	result: dict[str, dict[str, Any]] = {}
	for mode, rows in mode_details.items():
		n_total = len(rows)
		acc_vals: list[int] = []
		token_vals: list[int] = []
		missing_skills = 0
		skip_count = 0

		k = _parse_compress_k(mode)
		for row in rows:
			skill_ids = _normalize_skill_annotations(row.get("skill_annotations"))
			for sid in skill_ids:
				if sid not in full_token_map:
					missing_skills += 1
			token_vals.append(_mode_token_cost(mode, skill_ids, full_token_map))

			# For compress mode: check if any skill's full_text token < k
			# If so, use full_text's correct value for accuracy calculation
			use_full_text_correct = False
			if k is not None and skill_ids:
				for sid in skill_ids:
					full_tokens = full_token_map.get(sid, 0)
					if full_tokens < k:
						use_full_text_correct = True
						break

			if use_full_text_correct:
				skip_count += 1
				instance_id = str(row.get("instance_id", "")).strip()
				ft_correct = full_text_correct_index.get(instance_id)
				if ft_correct is not None:
					acc_vals.append(1 if ft_correct else 0)
			else:
				correct = _coerce_correct(row.get("correct"))
				if correct is not None:
					acc_vals.append(1 if correct else 0)

		accuracy = (sum(acc_vals) / len(acc_vals)) if acc_vals else None
		avg_extra_tokens = (sum(token_vals) / len(token_vals)) if token_vals else None
		result[mode] = {
			"n_instances": n_total,
			"n_accuracy_rows": len(acc_vals),
			"accuracy": accuracy,
			"avg_extra_tokens": avg_extra_tokens,
			"missing_skill_refs": missing_skills,
		}
		if k is not None:
			result[mode]["skip_count"] = skip_count
	return result


def _extract_exam_per_skill_accuracy(
	exam_summary: dict[str, Any],
	dataset_name: str,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
	"""Extract per-skill accuracy and full_text skip_count from exam summary.

	Returns:
		- per_skill_acc: dict mapping skill_id -> {mode: accuracy}
		- per_skill_skip_count: dict mapping skill_id -> full_text skip_count
	"""
	per_dataset = exam_summary.get("per_dataset")
	if not isinstance(per_dataset, list):
		raise ValueError("exam summary must contain per_dataset list")

	matched_dataset: dict[str, Any] | None = None
	for item in per_dataset:
		if not isinstance(item, dict):
			continue
		if str(item.get("dataset", "")).strip() == dataset_name:
			matched_dataset = item
			break

	if matched_dataset is None:
		raise ValueError(f"Dataset '{dataset_name}' not found in exam summary per_dataset")

	per_skill = matched_dataset.get("per_skill")
	if not isinstance(per_skill, list):
		raise ValueError("exam summary per_dataset item must contain per_skill list")

	per_skill_acc: dict[str, dict[str, float]] = {}
	per_skill_skip_count: dict[str, int] = {}
	for row in per_skill:
		if not isinstance(row, dict):
			continue
		sid = str(row.get("skill_id", "")).strip()
		if not sid:
			continue

		metrics = row.get("metrics")
		if not isinstance(metrics, dict):
			continue

		mode_acc: dict[str, float] = {}
		for mode, stat in metrics.items():
			if not isinstance(stat, dict):
				continue
			acc = stat.get("accuracy")
			if isinstance(acc, (int, float)):
				mode_acc[str(mode)] = float(acc)
		if mode_acc:
			per_skill_acc[sid] = mode_acc

		# Extract full_text skip_count
		full_text_metrics = metrics.get("full_text")
		if isinstance(full_text_metrics, dict):
			skip_count = full_text_metrics.get("skip_count")
			if isinstance(skip_count, (int, float)):
				per_skill_skip_count[sid] = int(skip_count)

	return per_skill_acc, per_skill_skip_count


def _skill_mode_token_cost(mode: str, full_tokens: int) -> int:
	if mode == "naive":
		return 0
	if mode == "full_text":
		return int(full_tokens)
	k = _parse_compress_k(mode)
	if k is None:
		raise ValueError(f"Unsupported mode name: {mode}")
	return min(int(full_tokens), int(k))


def _find_largest_compress_mode(available_modes: list[str]) -> str | None:
	"""Find the compress mode with the largest k value."""
	compress_modes = [m for m in available_modes if _is_compress_mode(m)]
	if not compress_modes:
		return None
	return max(compress_modes, key=lambda x: _parse_compress_k(x) or 0)


def select_mode_for_each_skill(
	full_token_map: dict[str, int],
	per_skill_acc: dict[str, dict[str, float]],
	available_modes: list[str],
	compress_over_naive_threshold: float | None = None,
	force_full_text_when_compress_not_meet_threshold: bool = False,
	compress_accuracy_threshold: float | None = None,
	select_most_compressed_when_meets_threshold: bool = False,
	per_skill_skip_count: dict[str, int] | None = None,
	skip_count_threshold: int = 0,
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
	selected: dict[str, str] = {}
	fallback_count = 0
	missing_exam_count = 0
	threshold_forced_count = 0
	skip_count_fallback_count = 0
	skip_count_filtered_skills: list[str] = []
	threshold_value = float(compress_over_naive_threshold or 0.0)
	acc_threshold_value = float(compress_accuracy_threshold or 1.0)
	skip_threshold = int(skip_count_threshold or 0)
	skill_skips = per_skill_skip_count or {}

	# Find the largest compress mode as the fallback when skip_count > threshold
	largest_compress = _find_largest_compress_mode(available_modes)

	for sid, full_tokens in full_token_map.items():
		# Check if full_text skip_count > threshold, force largest_compress
		# Only apply when threshold >= 0 (threshold < 0 means disabled)
		skill_skip = skill_skips.get(sid, 0)
		if skip_threshold >= 0 and skill_skip > skip_threshold and largest_compress is not None:
			selected[sid] = largest_compress
			skip_count_fallback_count += 1
			skip_count_filtered_skills.append(sid)
			continue

		mode_acc = per_skill_acc.get(sid, {})
		if not mode_acc:
			missing_exam_count += 1

		candidates: list[str] = []
		for mode in available_modes:
			if mode == "naive" or mode == "full_text":
				candidates.append(mode)
				continue

			k = _parse_compress_k(mode)
			if k is None:
				continue
			if k <= full_tokens:
				candidates.append(mode)

		best_mode: str | None = None
		best_acc = -math.inf
		best_cost = math.inf

		for mode in candidates:
			if mode not in mode_acc:
				continue

			acc = float(mode_acc[mode])
			cost = _skill_mode_token_cost(mode, full_tokens)
			if acc > best_acc or (acc == best_acc and cost < best_cost):
				best_mode = mode
				best_acc = acc
				best_cost = cost

		if best_mode is None:
			fallback_count += 1
			# Conservative fallback: pick smallest token-cost candidate.
			best_mode = min(candidates, key=lambda x: _skill_mode_token_cost(x, full_tokens))

		if force_full_text_when_compress_not_meet_threshold and _is_compress_mode(best_mode):
			naive_acc = mode_acc.get("naive")
			compress_acc = mode_acc.get(best_mode)
			full_text_acc = mode_acc.get("full_text")
			# Check if compress meets the threshold over naive
			meets_threshold = False
			if isinstance(naive_acc, (int, float)) and isinstance(compress_acc, (int, float)):
				meets_threshold = float(compress_acc) - float(naive_acc) > threshold_value
			# Check if compress >= golden full_text accuracy
			beats_full_text = False
			if isinstance(full_text_acc, (int, float)) and isinstance(compress_acc, (int, float)):
				beats_full_text = float(compress_acc) >= float(full_text_acc)
			# If compress doesn't meet threshold or doesn't beat full_text, fallback to full_text
			if not meets_threshold or not beats_full_text:
				best_mode = "full_text"
				threshold_forced_count += 1

		# New logic: select most compressed mode when accuracy meets threshold
		if select_most_compressed_when_meets_threshold:
			# Find all compress modes that meet the accuracy threshold
			valid_compress_modes: list[str] = []
			for mode in candidates:
				if not _is_compress_mode(mode):
					continue
				acc = mode_acc.get(mode)
				if isinstance(acc, (int, float)) and float(acc) >= acc_threshold_value:
					valid_compress_modes.append(mode)

			if valid_compress_modes:
				# Select the most compressed one (smallest k)
				best_mode = min(valid_compress_modes, key=lambda x: _skill_mode_token_cost(x, full_tokens))
			else:
				# No compress meets threshold, fallback to full_text
				best_mode = "full_text"
				threshold_forced_count += 1

		selected[sid] = best_mode

	diagnostics = {
		"n_skills": len(full_token_map),
		"missing_exam_skill_count": missing_exam_count,
		"fallback_skill_count": fallback_count,
		"threshold_forced_to_full_text_count": threshold_forced_count,
		"skip_count_fallback_count": skip_count_fallback_count,
		"compress_over_naive_threshold": threshold_value,
		"compress_accuracy_threshold": acc_threshold_value,
		"skip_count_threshold": skip_threshold,
	}
	return selected, diagnostics, skip_count_filtered_skills


def _build_instance_skill_map(mode_details: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
	mapping: dict[str, list[str]] = {}
	for rows in mode_details.values():
		for row in rows:
			instance_id = str(row.get("instance_id", "")).strip()
			if not instance_id:
				continue
			anns = _normalize_skill_annotations(row.get("skill_annotations"))
			if anns:
				mapping[instance_id] = anns
	return mapping


def _build_instance_correct_index(
	mode_details: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, bool | None]]:
	idx: dict[str, dict[str, bool | None]] = {}
	for mode, rows in mode_details.items():
		m: dict[str, bool | None] = {}
		for row in rows:
			instance_id = str(row.get("instance_id", "")).strip()
			if not instance_id:
				continue
			m[instance_id] = _coerce_correct(row.get("correct"))
		idx[mode] = m
	return idx


def _strictest_mode(modes: list[str]) -> str:
	if not modes:
		raise ValueError("modes must not be empty")
	return max(modes, key=_mode_strictness)


def compute_exam_filtered_metrics(
	mode_details: dict[str, list[dict[str, Any]]],
	full_token_map: dict[str, int],
	selected_skill_mode: dict[str, str],
	available_modes: list[str],
) -> dict[str, Any]:
	instance_skill_map = _build_instance_skill_map(mode_details)
	correct_index = _build_instance_correct_index(mode_details)

	fallback_default_mode = "naive" if "naive" in available_modes else min(
		available_modes,
		key=_mode_strictness,
	)

	correct_vals: list[int] = []
	token_vals: list[int] = []
	missing_eval_rows = 0
	unknown_skills = 0
	mode_counts: dict[str, int] = {}

	for instance_id, skill_ids in instance_skill_map.items():
		skill_modes: list[str] = []
		for sid in skill_ids:
			selected = selected_skill_mode.get(sid)
			if selected is None:
				unknown_skills += 1
				continue
			skill_modes.append(selected)

		final_mode = _strictest_mode(skill_modes) if skill_modes else fallback_default_mode
		mode_counts[final_mode] = mode_counts.get(final_mode, 0) + 1

		correct = correct_index.get(final_mode, {}).get(instance_id)
		if correct is None:
			missing_eval_rows += 1
		else:
			correct_vals.append(1 if correct else 0)

		token_vals.append(_mode_token_cost(final_mode, skill_ids, full_token_map))

	n_instances = len(instance_skill_map)
	accuracy = (sum(correct_vals) / len(correct_vals)) if correct_vals else None
	avg_extra_tokens = (sum(token_vals) / len(token_vals)) if token_vals else None
	mode_distribution_ratio = {
		mode: (count / n_instances) if n_instances > 0 else 0.0
		for mode, count in mode_counts.items()
	}

	return {
		"n_instances": n_instances,
		"n_accuracy_rows": len(correct_vals),
		"accuracy": accuracy,
		"avg_extra_tokens": avg_extra_tokens,
		"missing_eval_rows": missing_eval_rows,
		"unknown_skill_refs": unknown_skills,
		"mode_distribution": mode_counts,
		"mode_distribution_ratio": mode_distribution_ratio,
	}


def _build_exam_filtered_instance_rows(
	mode_details: dict[str, list[dict[str, Any]]],
	selected_skill_mode: dict[str, str],
	available_modes: list[str],
) -> list[dict[str, Any]]:
	instance_skill_map = _build_instance_skill_map(mode_details)
	correct_index = _build_instance_correct_index(mode_details)

	fallback_default_mode = "naive" if "naive" in available_modes else min(
		available_modes,
		key=_mode_strictness,
	)

	rows: list[dict[str, Any]] = []
	for instance_id, skill_ids in instance_skill_map.items():
		skill_modes: list[str] = []
		for sid in skill_ids:
			selected = selected_skill_mode.get(sid)
			if selected is None:
				continue
			skill_modes.append(selected)

		final_mode = _strictest_mode(skill_modes) if skill_modes else fallback_default_mode
		rows.append(
			{
				"instance_id": instance_id,
				"skill_ids": skill_ids,
				"final_mode": final_mode,
				"exam_filtered_correct": correct_index.get(final_mode, {}).get(instance_id),
				"full_text_correct": correct_index.get("full_text", {}).get(instance_id),
			}
		)

	return rows


def _find_exam_filtered_full_text_regret_skills(
	selected_skill_mode: dict[str, str],
	mode_details: dict[str, list[dict[str, Any]]],
	available_modes: list[str],
) -> list[dict[str, Any]]:
	instance_rows = _build_exam_filtered_instance_rows(
		mode_details=mode_details,
		selected_skill_mode=selected_skill_mode,
		available_modes=available_modes,
	)

	stats: dict[str, dict[str, Any]] = {}
	for row in instance_rows:
		skill_ids = row.get("skill_ids")
		if not isinstance(skill_ids, list):
			continue

		exam_correct = row.get("exam_filtered_correct")
		full_correct = row.get("full_text_correct")
		final_mode = str(row.get("final_mode", ""))

		for sid_raw in skill_ids:
			sid = str(sid_raw).strip()
			if not sid:
				continue
			if sid not in stats:
				stats[sid] = {
					"total_instances": 0,
					"paired_instances": 0,
					"exam_correct_count": 0,
					"full_text_correct_count": 0,
					"final_mode_distribution": {},
				}

			stats[sid]["total_instances"] += 1
			final_mode_dist = stats[sid]["final_mode_distribution"]
			final_mode_dist[final_mode] = final_mode_dist.get(final_mode, 0) + 1

			if isinstance(exam_correct, bool) and isinstance(full_correct, bool):
				stats[sid]["paired_instances"] += 1
				stats[sid]["exam_correct_count"] += 1 if exam_correct else 0
				stats[sid]["full_text_correct_count"] += 1 if full_correct else 0

	out: list[dict[str, Any]] = []
	for sid, item in stats.items():
		if sid not in selected_skill_mode:
			continue
		selected_mode = selected_skill_mode.get(sid, "")
		if selected_mode == "full_text":
			continue

		paired = int(item.get("paired_instances", 0))
		if paired <= 0:
			continue

		exam_acc = float(item.get("exam_correct_count", 0)) / float(paired)
		full_acc = float(item.get("full_text_correct_count", 0)) / float(paired)
		if full_acc <= exam_acc:
			continue

		out.append(
			{
				"skill_id": sid,
				"selected_mode": selected_mode,
				"paired_instances": paired,
				"exam_filtered_accuracy": exam_acc,
				"full_text_accuracy": full_acc,
				"accuracy_gap": full_acc - exam_acc,
				"final_mode_distribution": item.get("final_mode_distribution", {}),
			}
		)

	out.sort(key=lambda x: float(x["accuracy_gap"]), reverse=True)
	return out


def _format_distribution_line(dist_ratio: dict[str, Any]) -> str:
	parts: list[str] = []
	for mode, ratio in sorted(dist_ratio.items(), key=lambda x: _mode_strictness(str(x[0]))):
		parts.append(f"{mode}:{float(ratio):.4f}")
	return ", ".join(parts)


def run(
	tokenizer_path: str,
	eval_result_map_config_path: str,
	dataset_name_cli: str,
	corpus_path: str,
	exam_summary_path: str,
	compress_over_naive_threshold: float,
	compress_accuracy_threshold: float,
	skip_count_threshold: int,
	output_json: str,
) -> dict[str, Any]:
	dataset_name, eval_map = _parse_eval_result_map_config(
		config_path=eval_result_map_config_path,
		dataset_override=dataset_name_cli,
	)
	available_modes = list(eval_map.keys())

	full_token_map = build_full_token_map(
		tokenizer_path=tokenizer_path,
		corpus_path=corpus_path,
		dataset_name=dataset_name,
	)

	mode_details = load_mode_details(eval_map=eval_map, dataset_name=dataset_name)
	baselines = compute_mode_baselines(mode_details=mode_details, full_token_map=full_token_map)

	exam_summary = _load_json(_resolve_path(exam_summary_path))
	if not isinstance(exam_summary, dict):
		raise ValueError("exam summary must be a JSON object")

	per_skill_acc, per_skill_skip_count = _extract_exam_per_skill_accuracy(
		exam_summary=exam_summary,
		dataset_name=dataset_name,
	)

	ablation_modes = ["naive", "full_text"]
	compress_full_only_modes = [
		mode for mode in available_modes if mode == "full_text" or _is_compress_mode(mode)
	]
	selected_skill_mode, selection_diag, skip_filtered_skills = select_mode_for_each_skill(
		full_token_map=full_token_map,
		per_skill_acc=per_skill_acc,
		available_modes=available_modes,
		per_skill_skip_count=per_skill_skip_count,
		skip_count_threshold=skip_count_threshold,
	)
	selected_skill_mode_nf_only, selection_diag_nf_only, _ = select_mode_for_each_skill(
		full_token_map=full_token_map,
		per_skill_acc=per_skill_acc,
		available_modes=ablation_modes,
	)
	selected_skill_mode_thresholded, selection_diag_thresholded, skip_filtered_skills_thresholded = select_mode_for_each_skill(
		full_token_map=full_token_map,
		per_skill_acc=per_skill_acc,
		available_modes=available_modes,
		compress_over_naive_threshold=compress_over_naive_threshold,
		force_full_text_when_compress_not_meet_threshold=True,
		per_skill_skip_count=per_skill_skip_count,
		skip_count_threshold=skip_count_threshold,
	)
	selected_skill_mode_compress_full_only, selection_diag_compress_full_only, skip_filtered_skills_compress_full = select_mode_for_each_skill(
		full_token_map=full_token_map,
		per_skill_acc=per_skill_acc,
		available_modes=compress_full_only_modes,
		compress_accuracy_threshold=compress_accuracy_threshold,
		select_most_compressed_when_meets_threshold=True,
		per_skill_skip_count=per_skill_skip_count,
		skip_count_threshold=skip_count_threshold,
	)

	# Print skills filtered by skip_count threshold
	if skip_filtered_skills:
		print(f"[analysis] Skills filtered by skip_count_threshold (n={len(skip_filtered_skills)}):")
		for sid in skip_filtered_skills:
			skip_val = per_skill_skip_count.get(sid, 0)
			print(f"[analysis]   {sid}: skip_count={skip_val} -> forced to largest_compress")

	exam_filtered = compute_exam_filtered_metrics(
		mode_details=mode_details,
		full_token_map=full_token_map,
		selected_skill_mode=selected_skill_mode,
		available_modes=available_modes,
	)
	exam_filtered_naive_full_text_only = compute_exam_filtered_metrics(
		mode_details=mode_details,
		full_token_map=full_token_map,
		selected_skill_mode=selected_skill_mode_nf_only,
		available_modes=ablation_modes,
	)
	exam_filtered_compress_thresholded = compute_exam_filtered_metrics(
		mode_details=mode_details,
		full_token_map=full_token_map,
		selected_skill_mode=selected_skill_mode_thresholded,
		available_modes=available_modes,
	)
	exam_filtered_compress_full_text_only = compute_exam_filtered_metrics(
		mode_details=mode_details,
		full_token_map=full_token_map,
		selected_skill_mode=selected_skill_mode_compress_full_only,
		available_modes=compress_full_only_modes,
	)

	full_text_regret_skills = _find_exam_filtered_full_text_regret_skills(
		selected_skill_mode=selected_skill_mode,
		mode_details=mode_details,
		available_modes=available_modes,
	)

	mode_summary: dict[str, dict[str, Any]] = {}
	for mode, metric in baselines.items():
		mode_summary[mode] = {
			"accuracy": metric.get("accuracy"),
			"avg_extra_tokens": metric.get("avg_extra_tokens"),
		}
	mode_summary["exam_filtered"] = {
		"accuracy": exam_filtered.get("accuracy"),
		"avg_extra_tokens": exam_filtered.get("avg_extra_tokens"),
	}
	mode_summary["exam_filtered_naive_full_text_only"] = {
		"accuracy": exam_filtered_naive_full_text_only.get("accuracy"),
		"avg_extra_tokens": exam_filtered_naive_full_text_only.get("avg_extra_tokens"),
	}
	mode_summary["exam_filtered_compress_thresholded"] = {
		"accuracy": exam_filtered_compress_thresholded.get("accuracy"),
		"avg_extra_tokens": exam_filtered_compress_thresholded.get("avg_extra_tokens"),
	}
	mode_summary["exam_filtered_compress_full_text_only"] = {
		"accuracy": exam_filtered_compress_full_text_only.get("accuracy"),
		"avg_extra_tokens": exam_filtered_compress_full_text_only.get("avg_extra_tokens"),
	}

	result = {
		"dataset_name": dataset_name,
		"tokenizer_path": str(_resolve_path(tokenizer_path)),
		"corpus_path": str(_resolve_path(corpus_path)),
		"eval_result_map": {k: str(v) for k, v in eval_map.items()},
		"n_dataset_skills": len(full_token_map),
		"skill_full_token_map": full_token_map,
		"mode_baselines": baselines,
		"exam_filtered": exam_filtered,
		"exam_filtered_naive_full_text_only": exam_filtered_naive_full_text_only,
		"exam_filtered_compress_thresholded": exam_filtered_compress_thresholded,
		"exam_filtered_compress_full_text_only": exam_filtered_compress_full_text_only,
		"compress_over_naive_threshold": float(compress_over_naive_threshold),
		"skip_count_threshold": int(skip_count_threshold),
		"skill_selection": {
			"diagnostics": selection_diag,
			"selected_mode_by_skill": selected_skill_mode,
			"skip_count_filtered_skills": skip_filtered_skills,
		},
		"skill_selection_naive_full_text_only": {
			"diagnostics": selection_diag_nf_only,
			"selected_mode_by_skill": selected_skill_mode_nf_only,
		},
		"skill_selection_compress_thresholded": {
			"diagnostics": selection_diag_thresholded,
			"selected_mode_by_skill": selected_skill_mode_thresholded,
			"skip_count_filtered_skills": skip_filtered_skills_thresholded,
		},
		"skill_selection_compress_full_text_only": {
			"diagnostics": selection_diag_compress_full_only,
			"selected_mode_by_skill": selected_skill_mode_compress_full_only,
			"skip_count_filtered_skills": skip_filtered_skills_compress_full,
		},
		"exam_filtered_full_text_regret_skills": full_text_regret_skills,
		"method_summary": mode_summary,
	}

	if output_json.strip():
		out_path = _resolve_path(output_json)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(result, f, ensure_ascii=False, indent=2)
		print(f"[analysis] wrote output -> {out_path}")

	print(f"[analysis] dataset={dataset_name}")
	for mode, metric in mode_summary.items():
		print(
			f"[analysis] {mode:>12} | accuracy={metric['accuracy']} | "
			f"avg_extra_tokens={metric['avg_extra_tokens']}"
		)

	for tag, metrics in [
		("exam_filtered", exam_filtered),
		("exam_filtered_naive_full_text_only", exam_filtered_naive_full_text_only),
		("exam_filtered_compress_thresholded", exam_filtered_compress_thresholded),
		("exam_filtered_compress_full_text_only", exam_filtered_compress_full_text_only),
	]:
		dist_ratio = metrics.get("mode_distribution_ratio")
		if isinstance(dist_ratio, dict):
			print(f"[analysis] {tag} mode_distribution_ratio -> {_format_distribution_line(dist_ratio)}")

	if full_text_regret_skills:
		print(
			"[analysis] exam_filtered full_text-regret skills "
			f"(n={len(full_text_regret_skills)}): "
			+ ", ".join([str(x["skill_id"]) for x in full_text_regret_skills])
		)
		for item in full_text_regret_skills:
			print(
				"[analysis] full_text-regret-skill "
				f"skill_id={item['skill_id']} "
				f"selected_mode={item['selected_mode']} "
				f"exam_filtered_acc={item['exam_filtered_accuracy']:.4f} "
				f"full_text_acc={item['full_text_accuracy']:.4f} "
				f"gap={item['accuracy_gap']:.4f}"
			)
	else:
		print("[analysis] exam_filtered full_text-regret skills (n=0)")

	return result


def main() -> None:
	args = parse_args()
	run(
		tokenizer_path=args.tokenizer_path,
		eval_result_map_config_path=args.eval_result_map_config_path,
		dataset_name_cli=args.dataset_name,
		corpus_path=args.corpus_path,
		exam_summary_path=args.exam_summary_path,
		compress_over_naive_threshold=args.compress_over_naive_threshold,
		compress_accuracy_threshold=args.compress_accuracy_threshold,
		skip_count_threshold=args.skip_count_threshold,
		output_json=args.output_json,
	)


if __name__ == "__main__":
	main()
