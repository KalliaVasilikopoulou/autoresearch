"""Fast XAI methods: real (non-mocked) attention-head ablation."""

import torch
from typing import Callable, Dict, List


class FastXAIMethods:
    """Fast, practical XAI techniques for Agent 2.

    These methods only make sense while a trained model is still live in
    memory (train.py runs as a subprocess/SSH session and never checkpoints
    to disk), so `top_k_ablation_study` is called from inside train.py right
    after its final evaluation — not from Agent 2, which only ever receives
    a metrics dict.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def top_k_ablation_study(
        self,
        model,
        eval_fn: Callable[[object], float],
        k: int = 10,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Ablate the top-K attention heads (ranked by the norm of the
        `c_proj.weight` input-column slice that carries each head's output
        into the residual stream) and measure the change in `eval_fn`.

        A head is ablated by zeroing that column slice — this removes the
        head's contribution to the block's output without needing a
        per-head scale parameter (the model has none).

        Args:
            model: live GPT instance (see train.py's CausalSelfAttention —
                per-head slices live in `c_proj`'s input columns, not in a
                separate `c_attn`/`head_scales`).
            eval_fn: callable(model) -> bpb. Should be a *cheap* evaluation
                (small fixed step count), independent of the official
                full-budget `evaluate_bpb` metric, since this runs k+1
                forward passes on top of training.
            k: number of heads to ablate, ranked by weight magnitude.

        Returns:
            {"L{layer}_H{head}": impact} where impact = baseline_bpb - ablated_bpb
            (positive = ablating this head hurt bpb, i.e. head is important).
        """
        baseline_bpb = eval_fn(model)

        candidates = []
        for layer_idx, block in enumerate(model.transformer.h):
            attn = block.attn
            head_dim = attn.head_dim
            for head_idx in range(attn.n_head):
                start, end = head_idx * head_dim, (head_idx + 1) * head_dim
                magnitude = torch.norm(attn.c_proj.weight[:, start:end]).item()
                candidates.append((layer_idx, head_idx, magnitude))

        candidates.sort(key=lambda item: item[2], reverse=True)
        top_candidates = candidates[:k]

        impacts: Dict[str, float] = {}
        for count, (layer_idx, head_idx, _magnitude) in enumerate(top_candidates, start=1):
            attn = model.transformer.h[layer_idx].attn
            head_dim = attn.head_dim
            start, end = head_idx * head_dim, (head_idx + 1) * head_dim

            with torch.no_grad():
                original = attn.c_proj.weight[:, start:end].clone()
                attn.c_proj.weight[:, start:end] = 0.0
            try:
                ablated_bpb = eval_fn(model)
            finally:
                with torch.no_grad():
                    attn.c_proj.weight[:, start:end] = original

            impacts[f"L{layer_idx}_H{head_idx}"] = baseline_bpb - ablated_bpb
            if verbose and count % 5 == 0:
                print(f"[FastXAI] Ablated {count}/{len(top_candidates)} heads...")

        return impacts

    def detect_stuck_signal(
        self, all_impacts: List[Dict[str, float]], threshold: int = 5
    ) -> bool:
        """
        Detect if model is stuck (same top-impact heads across N previous
        ablation runs — a sign the architecture search isn't exploring
        anything new).
        """
        if len(all_impacts) < threshold:
            return False

        recent_tops = []
        for impacts_dict in all_impacts[-threshold:]:
            if not impacts_dict:
                continue
            top_5 = sorted(impacts_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            recent_tops.append([head for head, _ in top_5])

        if not recent_tops:
            return False

        first_top = set(recent_tops[0])
        return all(set(tops) == first_top for tops in recent_tops)
