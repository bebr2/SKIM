from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_hidden_size(config) -> int:
    """Return hidden_size across common Hugging Face config shapes."""
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError(f"Cannot find hidden_size in config: {type(config).__name__}")


def _build_chat_prompt_input_ids(tokenizer, user_text: str) -> torch.Tensor:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        # Some chat templates do not support enable_thinking.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def _find_subsequence(sequence: list[int], subseq: list[int]) -> tuple[int, int]:
    if not subseq:
        raise ValueError("Subsequence token ids cannot be empty")

    max_i = len(sequence) - len(subseq)
    for i in range(max_i + 1):
        if sequence[i : i + len(subseq)] == subseq:
            return i, i + len(subseq)
    raise ValueError("Cannot find subsequence in chat prompt token ids")


def _build_chat_template_user_shell_ids(tokenizer) -> tuple[list[int], list[int]]:
    probe = "__SKIM_COMPRESSOR_USER_PROBE_6f37d7a4__"
    full_ids = _build_chat_prompt_input_ids(tokenizer, probe)[0].tolist()
    probe_ids = tokenizer.encode(probe, add_special_tokens=False)
    start, end = _find_subsequence(full_ids, probe_ids)
    return full_ids[:start], full_ids[end:]


class SKIMCompressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_q: int,
        use_chat_shell: bool = True,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        self.hidden_size = _get_hidden_size(self.backbone.config)
        self.max_q = max_q
        self.use_chat_shell = use_chat_shell
        dtype = torch_dtype if torch_dtype is not None else torch.float32
        self.learnable_q = nn.Parameter(torch.randn(max_q, self.hidden_size, dtype=dtype) * 0.02)

        if self.use_chat_shell:
            user_prefix_ids, user_suffix_ids = _build_chat_template_user_shell_ids(self.tokenizer)
        else:
            user_prefix_ids, user_suffix_ids = [], []
        self.register_buffer(
            "_chat_user_prefix_ids",
            torch.tensor(user_prefix_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_chat_user_suffix_ids",
            torch.tensor(user_suffix_ids, dtype=torch.long),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, k: int) -> torch.Tensor:
        if k > self.max_q:
            raise ValueError(f"k={k} exceeds max_q={self.max_q}")
        bs = input_ids.size(0)

        emb_layer = self.backbone.get_input_embeddings()
        input_embeds = emb_layer(input_ids)

        prefix_ids = self._chat_user_prefix_ids.to(input_ids.device)
        suffix_ids = self._chat_user_suffix_ids.to(input_ids.device)
        prefix_len = int(prefix_ids.numel())
        suffix_len = int(suffix_ids.numel())

        pieces = []
        masks = []

        if prefix_len > 0:
            prefix_embeds = emb_layer(prefix_ids.unsqueeze(0)).expand(bs, -1, -1)
            prefix_mask = torch.ones((bs, prefix_len), device=attention_mask.device, dtype=attention_mask.dtype)
            pieces.append(prefix_embeds)
            masks.append(prefix_mask)

        pieces.append(input_embeds)
        masks.append(attention_mask)

        if suffix_len > 0:
            suffix_embeds = emb_layer(suffix_ids.unsqueeze(0)).expand(bs, -1, -1)
            suffix_mask = torch.ones((bs, suffix_len), device=attention_mask.device, dtype=attention_mask.dtype)
            pieces.append(suffix_embeds)
            masks.append(suffix_mask)

        q = self.learnable_q[:k].to(device=input_ids.device, dtype=input_embeds.dtype).unsqueeze(0).expand(bs, -1, -1)
        q_mask = torch.ones((bs, k), device=attention_mask.device, dtype=attention_mask.dtype)

        pieces.append(q)
        masks.append(q_mask)

        merged_embeds = torch.cat(pieces, dim=1)
        merged_mask = torch.cat(masks, dim=1)

        outputs = self.backbone.model(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        q_start = prefix_len + input_ids.size(1) + suffix_len
        return last_hidden[:, q_start : q_start + k, :]
