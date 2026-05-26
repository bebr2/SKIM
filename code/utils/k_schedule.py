from __future__ import annotations

import random


def pick_k(
    k_values: list[int],
    strategy: str,
    step: int,
    total_steps: int,
    warmup_ratio: float,
) -> int:
    if not k_values:
        raise ValueError("k_values is empty")

    sorted_k = sorted(k_values)

    if strategy in {"random_k", "random"}:
        return random.choice(sorted_k)

    if strategy == "curriculum":
        # k from large to small: warmup at largest k, then linear descend by progress.
        if total_steps <= 0:
            return sorted_k[-1]
        warmup_steps = int(total_steps * warmup_ratio)
        if step < warmup_steps:
            return sorted_k[-1]
        remain_steps = max(total_steps - warmup_steps, 1)
        p = min(max((step - warmup_steps) / remain_steps, 0.0), 0.999999)
        idx = int((1.0 - p) * (len(sorted_k) - 1))
        return sorted_k[idx]

    if strategy == "multi_k_per_case":
        # Single-k fallback for legacy callers.
        return sorted_k[-1]

    raise ValueError(f"Unknown k strategy: {strategy}")


def pick_ks_for_case(
    k_values: list[int],
    strategy: str,
    step: int,
    total_steps: int,
    warmup_ratio: float,
) -> list[int]:
    if not k_values:
        raise ValueError("k_values is empty")

    sorted_k = sorted(k_values)

    if strategy in {"random_k", "random", "curriculum"}:
        return [
            pick_k(
                k_values=sorted_k,
                strategy=strategy,
                step=step,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
            )
        ]

    if strategy == "multi_k_per_case":
        # Train all compression rates for each case.
        return sorted_k

    raise ValueError(f"Unknown k strategy: {strategy}")
