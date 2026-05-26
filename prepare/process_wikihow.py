import argparse
import csv
import json
import os
import re


def clean_text(text: str) -> str:
    """Remove common wiki-style formatting issues."""
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    sentences = text.split(". ")
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    return ". ".join(sentences)


def title_to_query(title: str) -> str:
    """Convert a WikiHow title to a question."""
    title = re.sub(r"\s+\d+$", "", title)
    title = re.sub(r"\s+([,.!?;:])", r"\1", title)
    if title.startswith("how to"):
        return f"How do I {title[6:].strip()}?"
    title = title.capitalize()
    return title if title.endswith("?") else f"{title}?"


def process_wikihow_csv(input_path: str, output_path: str, max_rows: int | None = None) -> None:
    """Process WikiHow CSV into SKIM skill QA JSONL format."""
    with open(input_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if max_rows is not None:
        rows = rows[:max_rows]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            query = title_to_query(row["title"])
            skill_content = clean_text(row["text"])
            answer = clean_text(row["summary"])
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Relevant Skill:\n<skill>skill_0</skill>\n\n{query}"},
                {"role": "assistant", "content": answer},
            ]
            record = {
                "id": f"wikihow/{i}",
                "name": row["title"].replace(" ", "-").replace("/", "-"),
                "owner": "wikihow",
                "repo": "wikihow-cleaned",
                "mode": "direct",
                "query": query,
                "skill_map": {"skill_0": skill_content},
                "messages_list": [messages],
                "meta": {
                    "use_split_skills": False,
                    "tool_count": 0,
                    "react_style": "single_turn",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Processed {len(rows)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert WikiHow CSV into SKIM skill QA JSONL.")
    parser.add_argument("--input", required=True, help="Path to wikihow-cleaned CSV")
    parser.add_argument("--output", default="./prepare/output/wikihow_processed.jsonl")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_wikihow_csv(args.input, args.output, max_rows=args.max_rows)
