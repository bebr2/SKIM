"""
Data cleaning script for generate_answer.py output files.
Filters entries based on token length constraints.
"""

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


class SimpleProgressBar:
    """Simple terminal progress bar."""

    def __init__(self, total: int, enabled: bool = True, width: int = 30):
        self.total = max(0, int(total))
        self.enabled = bool(enabled)
        self.width = max(10, int(width))
        self._last_len = 0

    def update(self, current: int, extra: str = "") -> None:
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

        if extra:
            text += f" | {extra}"

        pad = " " * max(0, self._last_len - len(text))
        print("\r" + text + pad, end="", flush=True)
        self._last_len = len(text)

    def close(self) -> None:
        if self.enabled:
            print("")


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


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Regex pattern for skill flags
SKILL_FLAG_PATTERN = re.compile(r"<skill>.*?</skill>", flags=re.DOTALL)


def remove_skill_flags(content: str) -> str:
    """Remove <skill>...</skill> tags and their content from text."""
    if not content:
        return content
    return SKILL_FLAG_PATTERN.sub("", content)


def count_skill_soft_tokens(skill_map: Dict[str, Any]) -> int:
    """Return 512 * number of skills for soft token budget."""
    if not isinstance(skill_map, dict):
        return 0
    return 512 * len(skill_map)


def calculate_context_token_length(
    messages: List[Dict[str, Any]],
    tokenizer: Any,
    skill_count: int,
) -> int:
    """
    Calculate context token length.
    - Remove assistant message (last message)
    - Remove <skill>...</skill> flags from content
    - Apply chat template with generation=True
    - Add 512 * skill_count for soft tokens
    """
    # Copy messages without assistant
    context_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        if role.lower() == "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            cleaned_content = remove_skill_flags(content)
        else:
            cleaned_content = content
        context_messages.append({"role": role, "content": cleaned_content})

    # Apply chat template
    try:
        # Use tokenize=False then manually encode to avoid tokenizer bugs
        chat_text = tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(chat_text, str):
            token_ids = tokenizer.encode(chat_text, add_special_tokens=False)
        else:
            # Fallback
            token_ids = tokenizer.encode(str(chat_text), add_special_tokens=False)
    except Exception:
        # Fallback: encode concatenated text
        text_parts = []
        for msg in context_messages:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            text_parts.append(f"{role}: {content}")
        full_text = "\n".join(text_parts)
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)

    return len(token_ids) + 512 * skill_count


def extract_few_shot_examples(user_content: str) -> Tuple[str, List[str], str, str]:
    """
    Extract few-shot examples from user prompt.
    Returns: (examples_prefix, list_of_examples, examples_suffix, rest_content)

    Structure:
    "Here are some examples:\n" + [examples] + "(END OF EXAMPLES)\n" + rest
    """
    examples_marker = "Here are some examples:\n"
    end_marker = "(END OF EXAMPLES)\n"

    examples_start = user_content.find(examples_marker)
    examples_end = user_content.find(end_marker)

    if examples_start == -1 or examples_end == -1 or examples_end < examples_start:
        # No examples section found
        return "", [], "", user_content

    # Extract parts
    examples_prefix = examples_marker
    examples_section = user_content[examples_start + len(examples_marker):examples_end]
    examples_suffix = end_marker
    rest_content = user_content[examples_end + len(end_marker):]

    # Split examples by "\n\nQuestion:"
    raw_examples = examples_section.split("\n\nQuestion:")
    parsed_examples: List[str] = []

    for i, ex in enumerate(raw_examples):
        ex = ex.strip()
        if not ex:
            continue
        if i == 0:
            # First piece already starts with "Question:"
            if ex.startswith("Question:"):
                parsed_examples.append(ex)
            else:
                parsed_examples.append("Question: " + ex)
        else:
            # Add "Question:" prefix back
            parsed_examples.append("Question: " + ex)

    return examples_prefix, parsed_examples, examples_suffix, rest_content


def calculate_assistant_token_length(
    messages_list: List[List[Dict[str, Any]]],
    tokenizer: Any,
) -> int:
    """
    Calculate total assistant token length across all turns.
    """
    total_length = 0
    for turn_messages in messages_list:
        for msg in turn_messages:
            role = str(msg.get("role", ""))
            if role.lower() == "assistant":
                content = str(msg.get("content", ""))
                token_ids = tokenizer.encode(content, add_special_tokens=False)
                total_length += len(token_ids)
    return total_length


def clean_direct_entry(
    row: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    """
    Clean direct mode entry.
    Keep only if context token length <= max_length - 10.
    """
    messages_list = row.get("messages_list")
    if not isinstance(messages_list, list) or not messages_list:
        return None

    messages = messages_list[0]
    if not isinstance(messages, list):
        return None

    skill_map = row.get("skill_map")
    if not isinstance(skill_map, dict):
        skill_map = {}
    skill_count = len(skill_map)

    context_length = calculate_context_token_length(messages, tokenizer, skill_count)

    if context_length <= max_length - 10:
        return row
    return None


def clean_react_entry(
    row: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
    stats: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Clean react mode entry.
    - Preserve system prompt, question, skill soft tokens
    - Trim few-shot examples to fit budget
    - Budget = max_length - assistant_token_length
    """
    messages_list = row.get("messages_list")
    if not isinstance(messages_list, list) or not messages_list:
        return None

    skill_map = row.get("skill_map")
    if not isinstance(skill_map, dict):
        skill_map = {}
    skill_count = len(skill_map)

    # Calculate assistant token length
    assistant_length = calculate_assistant_token_length(messages_list, tokenizer)
    budget = max_length - assistant_length

    if budget <= 0:
        stats["discard_budget_zero"] += 1
        return None

    # Get first turn messages
    first_turn = messages_list[0]
    if not isinstance(first_turn, list):
        return None

    system_msg = None
    user_msg = None
    for msg in first_turn:
        role = str(msg.get("role", ""))
        if role.lower() == "system":
            system_msg = msg
        elif role.lower() == "user":
            user_msg = msg

    if system_msg is None or user_msg is None:
        return None

    system_content = str(system_msg.get("content", ""))
    user_content = str(user_msg.get("content", ""))

    # Extract base_user_prefix (before "\nThought 1:")
    # Use rfind to find the LAST occurrence, which is after the examples section
    thought1_marker = "\nThought 1:"
    thought1_idx = user_content.rfind(thought1_marker)

    if thought1_idx == -1:
        # Fallback: treat entire content as base
        base_user_prefix = user_content.rstrip()
        scratchpad_suffix = ""
    else:
        base_user_prefix = user_content[:thought1_idx].rstrip()
        scratchpad_suffix = user_content[thought1_idx:]  # Keep "\nThought 1:" onward

    # Extract few-shot examples
    examples_prefix, examples_list, examples_suffix, rest_content = extract_few_shot_examples(base_user_prefix)

    # If no examples, just check if fits budget
    if not examples_list:
        # Calculate context length without examples
        test_messages = [
            {"role": "system", "content": remove_skill_flags(system_content)},
            {"role": "user", "content": remove_skill_flags(base_user_prefix)},
        ]
        base_length = calculate_context_token_length(test_messages, tokenizer, skill_count)

        if base_length <= budget:
            return row
        stats["discard_no_examples"] += 1
        return None

    # Sort examples by character length (ascending)
    sorted_examples = sorted(examples_list, key=len)

    # Calculate base context length (without examples)
    # Essential parts = rest_content (after "(END OF EXAMPLES)\n")
    essential_user_content = rest_content

    test_messages = [
        {"role": "system", "content": remove_skill_flags(system_content)},
        {"role": "user", "content": remove_skill_flags(essential_user_content)},
    ]
    base_length = calculate_context_token_length(test_messages, tokenizer, skill_count)

    if base_length > budget:
        stats["discard_base_over_budget"] += 1
        return None

    # Greedily add examples
    selected_examples: List[str] = []
    current_user_content = essential_user_content

    for ex in sorted_examples:
        # Try adding this example
        trial_content = examples_prefix + "\n".join(selected_examples + [ex]) + "\n" + examples_suffix + essential_user_content
        trial_messages = [
            {"role": "system", "content": remove_skill_flags(system_content)},
            {"role": "user", "content": remove_skill_flags(trial_content)},
        ]
        trial_length = calculate_context_token_length(trial_messages, tokenizer, skill_count)

        if trial_length <= budget:
            selected_examples.append(ex)
            current_user_content = trial_content
        else:
            # This example doesn't fit, stop
            break

    stats["examples_kept"] += len(selected_examples)
    stats["examples_total"] += len(examples_list)

    if len(selected_examples) == 0:
        stats["entries_zero_examples"] += 1

    # Construct new base_user_prefix
    if selected_examples:
        # Join selected examples with proper format
        joined_examples = selected_examples[0]
        for ex in selected_examples[1:]:
            joined_examples += "\n\n" + ex
        new_base_user_prefix = examples_prefix + joined_examples + "\n" + examples_suffix + essential_user_content
    else:
        # No examples kept, remove examples section entirely
        new_base_user_prefix = essential_user_content

    # Update all turns' user prompts
    new_messages_list: List[List[Dict[str, Any]]] = []

    # Need to find the old base_user_prefix in each turn and replace it
    # For first turn, we know the structure
    # For subsequent turns, the user prompt is: base_user_prefix + scratchpad + "\nThought N:"
    # The scratchpad grows with each turn

    for turn_idx, turn_messages in enumerate(messages_list):
        new_turn: List[Dict[str, Any]] = []
        for msg in turn_messages:
            role = str(msg.get("role", ""))
            content = msg.get("content")

            if role.lower() == "user" and isinstance(content, str):
                # Replace old base_user_prefix with new one
                if turn_idx == 0:
                    # First turn: use the scratchpad_suffix we extracted
                    new_user_content = new_base_user_prefix + scratchpad_suffix
                else:
                    # Subsequent turns: find and replace the base prefix part
                    # The base prefix is at the start, followed by accumulated scratchpad
                    # We need to find where the old base_user_prefix ends
                    if content.startswith(base_user_prefix):
                        new_user_content = new_base_user_prefix + content[len(base_user_prefix):]
                    else:
                        # Fallback: keep original if structure doesn't match
                        new_user_content = content
                new_turn.append({"role": role, "content": new_user_content})
            else:
                new_turn.append(msg)
        new_messages_list.append(new_turn)

    # Return modified row
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "owner": row.get("owner"),
        "repo": row.get("repo"),
        "mode": row.get("mode"),
        "query": row.get("query"),
        "tools": row.get("tools"),
        "skill_map": skill_map,
        "messages_list": new_messages_list,
        "meta": row.get("meta"),
    }


def detect_mode(rows: List[Dict[str, Any]]) -> str:
    """Auto-detect mode from rows."""
    for row in rows:
        mode = str(row.get("mode", ""))
        if mode in {"direct", "react"}:
            return mode
    return "direct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean generate_answer output files based on token length constraints."
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer path (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token length (default: 2048)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "react"],
        default=None,
        help="Processing mode (auto-detected if not specified)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        print(f"Error loading tokenizer: {exc}")
        sys.exit(1)

    # Load input
    rows = load_jsonl(args.input)

    # Detect mode
    mode = args.mode or detect_mode(rows)
    print(f"Processing mode: {mode}")
    print(f"Input rows: {len(rows)}")

    # Statistics
    stats: Dict[str, Any] = {
        "input_count": len(rows),
        "output_count": 0,
        "discarded_count": 0,
        "before_lengths": [],
        "after_lengths": [],
        # React-specific stats
        "examples_kept": 0,
        "examples_total": 0,
        "entries_zero_examples": 0,
        "discard_budget_zero": 0,
        "discard_no_examples": 0,
        "discard_base_over_budget": 0,
    }

    # Calculate before lengths (with progress bar)
    print("Calculating original token lengths...")
    progress = SimpleProgressBar(total=len(rows))
    for idx, row in enumerate(rows):
        messages_list = row.get("messages_list")
        if not messages_list:
            continue
        skill_map = row.get("skill_map", {})
        skill_count = len(skill_map) if isinstance(skill_map, dict) else 0

        if mode == "direct":
            length = calculate_context_token_length(messages_list[0], tokenizer, skill_count)
            stats["before_lengths"].append(length)
        else:
            # For react, calculate first turn context length
            first_turn = messages_list[0]
            length = calculate_context_token_length(first_turn, tokenizer, skill_count)
            stats["before_lengths"].append(length)
        progress.update(idx + 1, f"mode={mode}")
    progress.close()

    # Process rows (with progress bar)
    print("Cleaning entries...")
    output_rows: List[Dict[str, Any]] = []
    progress = SimpleProgressBar(total=len(rows))

    for idx, row in enumerate(rows):
        if mode == "direct":
            result = clean_direct_entry(row, tokenizer, args.max_length)
        else:
            result = clean_react_entry(row, tokenizer, args.max_length, stats)

        if result is not None:
            output_rows.append(result)
            # Calculate after length
            messages_list = result.get("messages_list")
            skill_map = result.get("skill_map", {})
            skill_count = len(skill_map) if isinstance(skill_map, dict) else 0
            if messages_list:
                if mode == "direct":
                    length = calculate_context_token_length(messages_list[0], tokenizer, skill_count)
                else:
                    length = calculate_context_token_length(messages_list[0], tokenizer, skill_count)
                stats["after_lengths"].append(length)
        else:
            stats["discarded_count"] += 1

        progress.update(idx + 1, f"kept={len(output_rows)} discarded={stats['discarded_count']}")

    progress.close()

    stats["output_count"] = len(output_rows)

    # Write output
    write_jsonl(args.output, output_rows)

    # Print statistics
    print(f"\n=== Cleaning Statistics ===")
    print(f"Input count: {stats['input_count']}")
    print(f"Output count: {stats['output_count']}")
    print(f"Discarded count: {stats['discarded_count']}")
    print(f"Discard ratio: {stats['discarded_count'] / stats['input_count']:.2%}")

    if stats["before_lengths"]:
        print(f"\nBefore cleaning lengths:")
        print(f"  Min: {min(stats['before_lengths'])}")
        print(f"  Max: {max(stats['before_lengths'])}")
        print(f"  Avg: {sum(stats['before_lengths']) / len(stats['before_lengths']):.1f}")

    if stats["after_lengths"]:
        print(f"\nAfter cleaning lengths:")
        print(f"  Min: {min(stats['after_lengths'])}")
        print(f"  Max: {max(stats['after_lengths'])}")
        print(f"  Avg: {sum(stats['after_lengths']) / len(stats['after_lengths']):.1f}")

    if mode == "react":
        print(f"\nReact-specific statistics:")
        print(f"  Examples total: {stats['examples_total']}")
        print(f"  Examples kept: {stats['examples_kept']}")
        if stats['examples_total'] > 0:
            print(f"  Examples kept ratio: {stats['examples_kept'] / stats['examples_total']:.2%}")
        print(f"  Entries with 0 examples: {stats['entries_zero_examples']}")
        print(f"  Discarded (budget zero): {stats['discard_budget_zero']}")
        print(f"  Discarded (no examples, over budget): {stats['discard_no_examples']}")
        print(f"  Discarded (base over budget): {stats['discard_base_over_budget']}")

    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()