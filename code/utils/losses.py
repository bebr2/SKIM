from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_rank_loss(scores: torch.Tensor, target_order: torch.Tensor) -> torch.Tensor:
    """
    scores: [B, N] higher means more relevant
    target_order: [B, N] permutation of doc indices in descending relevance
    """
    bsz, n_docs = scores.shape
    loss = torch.zeros((), device=scores.device)

    for b in range(bsz):
        inv_rank = torch.empty(n_docs, device=scores.device, dtype=torch.long)
        inv_rank[target_order[b]] = torch.arange(n_docs, device=scores.device)

        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                di = target_order[b, i]
                dj = target_order[b, j]
                si = scores[b, di]
                sj = scores[b, dj]
                weight = 1.0 / float((i + 1) + (j + 1))
                loss = loss + weight * torch.log1p(torch.exp(sj - si))

    denom = max(bsz * n_docs * max(n_docs - 1, 1) // 2, 1)
    return loss / denom


def sequence_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample summed token log-probability from causal LM logits/labels.

    logits: [B, S, V]
    labels: [B, S] with ignored positions marked as -100
    returns:
      sum_log_probs: [B]
      token_counts: [B] (number of valid target tokens)
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be rank-3 [B, S, V], got shape={tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be rank-2 [B, S], got shape={tuple(labels.shape)}")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    valid_mask = shift_labels != -100
    safe_labels = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))

    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = torch.where(valid_mask, token_log_probs, torch.zeros_like(token_log_probs))

    sum_log_probs = token_log_probs.sum(dim=-1)
    token_counts = valid_mask.sum(dim=-1)
    return sum_log_probs, token_counts


def simpo_loss(
    chosen_logp_sum: torch.Tensor,
    rejected_logp_sum: torch.Tensor,
    chosen_token_count: torch.Tensor,
    rejected_token_count: torch.Tensor,
    beta: float,
    gamma: float,
    pair_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SimPO objective without reference model.

    reward = mean token log-prob over valid tokens.
    loss = -logsigmoid(beta * (reward_pos - reward_neg) - gamma)
    """
    dtype = chosen_logp_sum.dtype
    device = chosen_logp_sum.device

    chosen_den = chosen_token_count.to(dtype=dtype).clamp_min(1.0)
    rejected_den = rejected_token_count.to(dtype=dtype).clamp_min(1.0)

    chosen_reward = chosen_logp_sum / chosen_den
    rejected_reward = rejected_logp_sum / rejected_den

    margin = float(beta) * (chosen_reward - rejected_reward) - float(gamma)
    per_sample = -F.logsigmoid(margin)

    if pair_mask is None:
        return per_sample.mean(), chosen_reward, rejected_reward

    mask = pair_mask.to(device=device, dtype=dtype)
    denom = mask.sum().clamp_min(1.0)
    masked_loss = (per_sample * mask).sum() / denom
    return masked_loss, chosen_reward, rejected_reward
