from __future__ import annotations

import torch


def stage1_prompt() -> str:
    return (
        "The compressed representation of the document is provided. "
        "Reconstruct the original passage exactly. Repeat the text exactly:"
    )


def _apply_chat_template(tokenizer, messages: list[dict[str, str]]):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        # Some templates do not support enable_thinking.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def build_chat_prompt_input_ids(tokenizer, user_text: str, system_text: str | None = None):
    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return _apply_chat_template(tokenizer, messages)


def _find_subsequence(sequence: list[int], subseq: list[int]) -> tuple[int, int]:
    if not subseq:
        raise ValueError("Subsequence token ids cannot be empty")

    max_i = len(sequence) - len(subseq)
    for i in range(max_i + 1):
        if sequence[i : i + len(subseq)] == subseq:
            return i, i + len(subseq)
    raise ValueError("Cannot find subsequence in chat prompt token ids")


def build_chat_template_user_shell_ids(tokenizer) -> tuple[list[int], list[int]]:
    # Use a stable probe once, then extract template prefix/suffix around user content.
    probe = "__ZR_USER_CONTENT_PROBE_6f37d7a4__"
    full_ids = build_chat_prompt_input_ids(tokenizer, probe)[0].tolist()
    probe_ids = tokenizer.encode(probe, add_special_tokens=False)
    start, end = _find_subsequence(full_ids, probe_ids)
    return full_ids[:start], full_ids[end:]


def build_chat_template_system_user_shell_ids(tokenizer) -> tuple[list[int], list[int], list[int]]:
    sys_probe = "__ZR_SYSTEM_CONTENT_PROBE_6f37d7a4__"
    user_probe = "__ZR_USER_CONTENT_PROBE_6f37d7a4__"
    full_ids = build_chat_prompt_input_ids(tokenizer, user_probe, system_text=sys_probe)[0].tolist()

    sys_probe_ids = tokenizer.encode(sys_probe, add_special_tokens=False)
    user_probe_ids = tokenizer.encode(user_probe, add_special_tokens=False)

    sys_start, sys_end = _find_subsequence(full_ids, sys_probe_ids)
    user_start, user_end = _find_subsequence(full_ids, user_probe_ids)
    if sys_end > user_start:
        raise ValueError("Unexpected chat template order for system/user probes")

    return full_ids[:sys_start], full_ids[sys_end:user_start], full_ids[user_end:]


def _get_llm_embedding_device(llm) -> torch.device:
    """Return the embedding device, including device_map models."""
    emb_layer = llm.get_input_embeddings()
    if hasattr(emb_layer, 'weight') and hasattr(emb_layer.weight, 'device'):
        return emb_layer.weight.device
    # fallback
    if hasattr(llm, 'device'):
        return llm.device
    return torch.device('cpu')


def _append_text_or_soft_segment(pieces: list[torch.Tensor], seg: str | torch.Tensor, llm, tokenizer) -> None:
    emb_layer = llm.get_input_embeddings()
    emb_device = _get_llm_embedding_device(llm)
    emb_dtype = emb_layer.weight.dtype if hasattr(emb_layer, 'weight') and hasattr(emb_layer.weight, 'dtype') else torch.float32

    if isinstance(seg, str):
        if not seg:
            return
        seg_ids = tokenizer.encode(seg, add_special_tokens=False)
        if not seg_ids:
            return
        ids_t = torch.tensor(seg_ids, device=emb_device, dtype=torch.long).unsqueeze(0)
        pieces.append(emb_layer(ids_t))
        return

    soft = seg
    if soft.dim() == 2:
        soft = soft.unsqueeze(0)
    if soft.dim() != 3 or soft.size(0) != 1:
        raise ValueError("Soft embedding segment must be [k, d] or [1, k, d]")
    soft = soft.to(device=emb_device, dtype=emb_dtype)
    pieces.append(soft)


def build_chat_prompt_embeds_from_user_segments(
    llm,
    tokenizer,
    segments: list[str | torch.Tensor],
    template_shell_ids: tuple[list[int], list[int]] | None = None,
) -> torch.Tensor:
    if template_shell_ids is None:
        prefix_ids, suffix_ids = build_chat_template_user_shell_ids(tokenizer)
    else:
        prefix_ids, suffix_ids = template_shell_ids
    emb_layer = llm.get_input_embeddings()
    emb_device = _get_llm_embedding_device(llm)

    pieces: list[torch.Tensor] = []

    if prefix_ids:
        p = torch.tensor(prefix_ids, device=emb_device, dtype=torch.long).unsqueeze(0)
        pieces.append(emb_layer(p))

    for seg in segments:
        _append_text_or_soft_segment(pieces, seg, llm, tokenizer)

    if suffix_ids:
        s = torch.tensor(suffix_ids, device=emb_device, dtype=torch.long).unsqueeze(0)
        pieces.append(emb_layer(s))

    if not pieces:
        raise ValueError("No segments produced for chat prompt embeddings")
    return torch.cat(pieces, dim=1)


def build_chat_prompt_embeds_from_system_and_user_segments(
    llm,
    tokenizer,
    system_text: str,
    user_segments: list[str | torch.Tensor],
    template_shell_ids: tuple[list[int], list[int], list[int]] | None = None,
    force_system_template: bool = False,
) -> torch.Tensor:
    if (not force_system_template) and (not system_text.strip()):
        return build_chat_prompt_embeds_from_user_segments(
            llm=llm,
            tokenizer=tokenizer,
            segments=user_segments,
        )

    if template_shell_ids is None:
        sys_prefix_ids, sys_user_middle_ids, user_suffix_ids = build_chat_template_system_user_shell_ids(tokenizer)
    else:
        sys_prefix_ids, sys_user_middle_ids, user_suffix_ids = template_shell_ids

    emb_layer = llm.get_input_embeddings()
    emb_device = _get_llm_embedding_device(llm)
    pieces: list[torch.Tensor] = []

    if sys_prefix_ids:
        p = torch.tensor(sys_prefix_ids, device=emb_device, dtype=torch.long).unsqueeze(0)
        pieces.append(emb_layer(p))

    system_ids = tokenizer.encode(system_text, add_special_tokens=False)
    if system_ids:
        system_t = torch.tensor(system_ids, device=emb_device, dtype=torch.long).unsqueeze(0)
        pieces.append(emb_layer(system_t))

    if sys_user_middle_ids:
        m = torch.tensor(sys_user_middle_ids, device=emb_device, dtype=torch.long).unsqueeze(0)
        pieces.append(emb_layer(m))

    for seg in user_segments:
        _append_text_or_soft_segment(pieces, seg, llm, tokenizer)

    if user_suffix_ids:
        s = torch.tensor(user_suffix_ids, device=emb_device, dtype=torch.long).unsqueeze(0)
        pieces.append(emb_layer(s))

    if not pieces:
        raise ValueError("No segments produced for system+user chat prompt embeddings")
    return torch.cat(pieces, dim=1)

