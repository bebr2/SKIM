from __future__ import annotations

import string


def get_alpha_identifiers(n: int, limit: int = 26) -> list[str]:
    if n > limit:
        raise ValueError(f"Requested {n} identifiers exceeds limit={limit}")
    return [f"[{ch}]" for ch in string.ascii_uppercase[:n]]


def identifier_to_token_map(tokenizer, identifiers: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for ident in identifiers:
        char = ident[1]
        token_id = tokenizer.convert_tokens_to_ids(char)
        if token_id is None or token_id == tokenizer.unk_token_id:
            encoded = tokenizer.encode(char, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Identifier char {char} does not map to a single token")
            token_id = encoded[0]
        mapping[ident] = token_id
    return mapping
