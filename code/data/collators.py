from __future__ import annotations

import torch

from data.skill_qa import SkillQASample
from data.reconstruction import ReconstructionSample


class ReconstructionCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[ReconstructionSample]) -> dict[str, torch.Tensor | list[dict]]:
        docs = [x.document for x in batch]
        enc = self.tokenizer(
            docs,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            "doc_input_ids": enc["input_ids"],
            "doc_attention_mask": enc["attention_mask"],
            "metadata": [x.metadata for x in batch],
        }


class SkillQACollator:
    def __call__(self, batch: list[SkillQASample]) -> dict[str, list[str] | list[dict] | list[bool]]:
        return {
            "system_contents": [x.system_content for x in batch],
            "user_contents": [x.user_content for x in batch],
            "assistant_contents": [x.assistant_content for x in batch],
            "rejected_assistant_contents": [x.rejected_assistant_content or "" for x in batch],
            "has_rejected_pair": [bool(x.has_rejected_pair) for x in batch],
            "metadata": [x.metadata for x in batch],
        }
