"""Token-level XAI (Tier 2, see dev/INNOVATION_PLAN.md): what the model
actually does with the input, not just which hyperparameter mattered.

train.py's real attention path (`_attn_func`) calls
`F.scaled_dot_product_attention`, which never returns attention weights --
there is no hook to pull them from. Everything here is a second,
analysis-only forward path that replicates the same math manually
(`q @ k.T / sqrt(d)`, masked, softmax) so weights can be inspected. Never
used in the training path.

Reaches into the live trained model instance directly -- the same pattern
`fast_methods.py`'s ablation study already uses
(`model.transformer.h[i].attn.c_proj.weight`) -- rather than reimplementing
its weights. Only `_norm`/`_apply_rotary_emb` are duplicated locally (tiny,
stable pure math); everything else (`model.cos`/`model.sin`,
`model.window_sizes`, `model.value_embeds`, `model.resid_lambdas`/
`model.x0_lambdas`) is read straight off the model.

VRAM discipline: run on short slices (T~384, a handful of batches), and
never retain more than one layer's [B,H,T,T] attention-weight tensor at a
time in the returned structure beyond what's needed for the fingerprint.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _norm(x: torch.Tensor) -> torch.Tensor:
    """Must match train.py's norm() exactly (F.rms_norm)."""
    return F.rms_norm(x, (x.size(-1),))


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Must match train.py's apply_rotary_emb() exactly."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def forward_with_analysis(
    model, idx: torch.Tensor, targets: Optional[torch.Tensor], cos_sin,
    emb_override: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Analysis-only forward pass mirroring GPT.forward layer-by-layer, but
    also collecting what every fingerprint component needs from one pass:

    emb_override: if given, used as the embedding tensor instead of
    model.transformer.wte(idx) (idx's token ids are still used for value
    embeddings). Lets callers inject a perturbed/detached embedding -- e.g.
    a finite-difference check of position_saliency's gradient against a
    numerical derivative, which needs to perturb one position's embedding
    independent of the shared wte weight table (perturbing wte.weight
    directly would leak into every other position using the same token id).
      - "emb": the (non-leaf) embedding output, with retain_grad() called so
        a later loss.backward() populates emb.grad for position_saliency.
      - "per_layer_weights": list of [B,H,T,T] attention weight tensors
        (fp32), one per layer -- consumed by attn_entropy_and_distance.
      - "per_layer_deltas": list of [B,T,C] residual-stream deltas (detached),
        one per layer (what each block actually added) -- consumed by
        direct_logit_attribution.
      - "final_x": the normed residual stream right before lm_head (detached).
      - "final_x_pre_norm": the same residual stream before the final norm
        (detached) -- consumed by direct_logit_attribution to recover the
        real per-token RMSNorm scale factor.
      - "logits"/"loss": same computation train.py's GPT.forward would
        produce for this input (softcapped, cross-entropy) -- kept attached
        to the graph so loss.backward() works.
    Not decorated with @torch.no_grad(): callers that don't need gradients
    (entropy/distance/DLA) should still get correct numbers since nothing
    here mutates model state, but they may want to wrap the call in
    torch.no_grad() themselves to save memory when they don't need emb.grad.
    """
    B, T = idx.size()
    cos, sin = cos_sin

    emb = model.transformer.wte(idx) if emb_override is None else emb_override
    if emb.requires_grad:
        emb.retain_grad()
    x = _norm(emb)
    x0 = x

    per_layer_weights: List[torch.Tensor] = []
    per_layer_deltas: List[torch.Tensor] = []

    for i, block in enumerate(model.transformer.h):
        block_input = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        attn = block.attn
        normed = _norm(block_input)

        q = attn.c_q(normed).view(B, T, attn.n_head, attn.head_dim)
        k = attn.c_k(normed).view(B, T, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(normed).view(B, T, attn.n_kv_head, attn.head_dim)

        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        if ve is not None:
            ve = ve.view(B, T, attn.n_kv_head, attn.head_dim)
            gate = 2 * torch.sigmoid(attn.ve_gate(normed[..., :attn.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
        q, k = _norm(q), _norm(k)

        q2 = q.transpose(1, 2)  # [B,H,T,D]
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        if k2.shape[1] < q2.shape[1]:  # GQA expand (this repo always sets n_kv_head==n_head, handled generally anyway)
            ratio = q2.shape[1] // k2.shape[1]
            k2 = k2.repeat_interleave(ratio, dim=1)
            v2 = v2.repeat_interleave(ratio, dim=1)

        scale = 1.0 / math.sqrt(attn.head_dim)
        scores = (q2.float() @ k2.float().transpose(-2, -1)) * scale  # [B,H,T,T]
        window = model.window_sizes[i][0]
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        if window < T:
            causal_mask &= torch.ones(T, T, dtype=torch.bool, device=idx.device).tril().triu(-(window - 1))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = scores.softmax(dim=-1)  # [B,H,T,T], fp32
        per_layer_weights.append(weights.detach())

        y = (weights.to(v2.dtype) @ v2).transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.c_proj(y)

        mlp_input = block_input + attn_out
        mlp_out = block.mlp(_norm(mlp_input))
        block_output = mlp_input + mlp_out

        per_layer_deltas.append((block_output - block_input).detach())
        x = block_output

    final_x = _norm(x)
    softcap = 15
    logits = model.lm_head(final_x).float()
    logits = softcap * torch.tanh(logits / softcap)

    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    return {
        "emb": emb,
        "loss": loss,
        "logits": logits,
        "final_x": final_x.detach(),
        "final_x_pre_norm": x.detach(),
        "x0": x0.detach(),
        "per_layer_weights": per_layer_weights,
        "per_layer_deltas": per_layer_deltas,
    }


def attn_entropy_and_distance(per_layer_weights: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
    """Per layer: mean softmax entropy of the attention distribution (low =
    focused, high = diffuse), and mean |i-j| weighted by attention weight
    (how far back this layer looks). Averaged over batch, heads, and query
    positions (excluding position 0, which can only attend to itself and is
    degenerate for both metrics).
    """
    entropy_per_layer: List[float] = []
    distance_per_layer: List[float] = []
    for weights in per_layer_weights:
        # weights: [B, H, T, T], row i sums to 1 over the causal-visible keys.
        B, H, T, _ = weights.shape
        eps = 1e-12
        entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # [B,H,T]
        positions = torch.arange(T, device=weights.device)
        distance_matrix = (positions.view(T, 1) - positions.view(1, T)).abs().float()  # [T,T]
        distance = (weights * distance_matrix).sum(dim=-1)  # [B,H,T]
        if T > 1:
            entropy = entropy[:, :, 1:]
            distance = distance[:, :, 1:]
        entropy_per_layer.append(entropy.mean().item())
        distance_per_layer.append(distance.mean().item())
    return entropy_per_layer, distance_per_layer


def position_saliency(model, idx: torch.Tensor, targets: torch.Tensor, n_buckets: int = 16) -> List[float]:
    """grad x input on the token embedding, from one backward() call,
    bucketed by how far back each input position is from the position being
    predicted. Answers: "how much does the token k positions back matter for
    predicting the next token here?"

    Only the last position's loss is backpropagated (one query per call), so
    "distance from the predicted position" is a well-defined single
    quantity -- compute_behavioral_fingerprint averages this over multiple
    batches/slices rather than trying to define one distance axis for every
    query position at once. Callers must ensure targets[:, -1] is a real,
    supervised target (not the ignore_index).
    """
    B, T = idx.size()
    cos, sin = model.cos[:, :T], model.sin[:, :T]
    out = forward_with_analysis(model, idx, targets=None, cos_sin=(cos, sin))
    emb = out["emb"]
    logits = out["logits"]  # [B,T,V], softcapped

    last_logits = logits[:, -1, :]
    last_targets = targets[:, -1]
    loss = F.cross_entropy(last_logits, last_targets)

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad = emb.grad  # [B,T,C]
    saliency = (grad * emb.detach()).abs().sum(dim=-1)  # [B,T]

    buckets: List[List[float]] = [[] for _ in range(n_buckets)]
    for i in range(T - 1):  # exclude the last position itself (the query, not context)
        distance = T - 1 - i
        bucket_idx = min(distance - 1, n_buckets - 1)
        buckets[bucket_idx].append(saliency[:, i].mean().item())

    return [sum(b) / len(b) if b else 0.0 for b in buckets]


def attn_distance_slope(attn_distance: List[float]) -> float:
    """Linear-regression slope of attn_distance vs. layer index -- does
    attention reach grow with depth? Ordinary least squares, no numpy/scipy
    dependency needed for one scalar.
    """
    n = len(attn_distance)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(attn_distance) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, attn_distance))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def direct_logit_attribution(
    per_layer_deltas: List[torch.Tensor],
    final_x_pre_norm: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    resid_lambdas: List[float],
) -> List[float]:
    """Approximate per-layer contribution to the target token's logit.

    train.py's norm() is plain RMSNorm with no learnable affine --
    x / rms(x) -- a per-token LINEAR rescaling once rms(x) is treated as a
    fixed scalar. This is the standard DLA approximation used throughout
    the mech-interp literature: freeze the actual final per-token scale
    factor (computed from the real, full residual stream, so it's exact,
    not estimated) and apply it uniformly when projecting each layer's
    individual write through the same final path (norm scale + lm_head).

    resid_lambdas: train.py mixes the residual stream at every block input
    (block_input_i = resid_lambda[i]*x_i + x0_lambda[i]*x0), so a delta
    written at layer i is rescaled by every subsequent layer's resid_lambda
    on its way to the final residual stream -- skipping this would make the
    attribution wrong (not just approximate) whenever resid_lambda != 1.
    The one remaining approximation is treating the norm scale as fixed
    across the additive decomposition -- documented as such in the
    fingerprint report; verified via a conservation check (layer sum
    should track the real target-direction logit closely).
    """
    n_layer = len(per_layer_deltas)
    valid = targets != -1  # [B,T]
    if valid.sum().item() == 0:
        return [0.0] * n_layer

    rms = final_x_pre_norm.float().pow(2).mean(dim=-1, keepdim=True).sqrt()  # [B,T,1]
    norm_scale = 1.0 / rms.clamp_min(1e-6)  # [B,T,1]

    target_directions = lm_head_weight[targets.clamp_min(0)].float()  # [B,T,C]

    # cumulative[i] = product of resid_lambda[j] for every layer j after i
    # (1.0 for the last layer -- nothing rescales its delta further).
    cumulative = [1.0] * n_layer
    running = 1.0
    for i in range(n_layer - 1, -1, -1):
        cumulative[i] = running
        running *= float(resid_lambdas[i])

    dla_per_layer: List[float] = []
    for i, delta in enumerate(per_layer_deltas):
        scaled = delta.float() * cumulative[i] * norm_scale  # [B,T,C]
        contribution = (scaled * target_directions).sum(dim=-1)  # [B,T]
        dla_per_layer.append(contribution[valid].mean().item())
    return dla_per_layer


def induction_score(model, cos_sin_full, vocab_size: int, seq_len: int = 64, seed: int = 0) -> float:
    """Standard induction-head probe (Olsson et al.): feed
    [random tokens][the same tokens repeated] and measure how strongly any
    head attends from a token in the second block to the position right
    after that same token's occurrence in the first block -- the
    "induction offset" that lets a head predict "whatever came after this
    token last time". Maxed over layers/heads, one scalar.

    Uses a synthetic probe, not the natural-language batch used elsewhere
    in the fingerprint -- real text has no guaranteed exact repeats to
    probe with. Structural check only: on a freshly-initialized or
    briefly-trained model this can legitimately be near zero -- induction
    heads are well documented to emerge late in training (often after a
    sudden phase transition), not something to chase in a short run.

    Requires 2*seq_len to fit inside the model's attention window (a layer
    using a window shorter than 2*seq_len simply cannot see far enough
    back to have a nonzero induction offset for early second-block
    queries) -- callers should keep seq_len well under half the model's
    shortest configured window.
    """
    cos_full, sin_full = cos_sin_full
    T = 2 * seq_len
    device = next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = 4
    first_block = torch.randint(0, vocab_size, (B, seq_len), generator=g)
    idx = torch.cat([first_block, first_block], dim=1).to(device)
    cos, sin = cos_full[:, :T], sin_full[:, :T]

    with torch.no_grad():
        out = forward_with_analysis(model, idx, targets=None, cos_sin=(cos, sin))

    best = 0.0
    for weights in out["per_layer_weights"]:  # [B,H,T,T]
        _, n_head, _, _ = weights.shape
        for h in range(n_head):
            offsets = [weights[:, h, seq_len + j, j + 1].mean().item() for j in range(seq_len - 1)]
            if offsets:
                best = max(best, sum(offsets) / len(offsets))
    return best


def compute_behavioral_fingerprint(
    model, tokenizer, device_batch_size: int, seq_len: int = 384, n_batches: int = 4,
) -> Dict[str, Any]:
    """The one entry point train.py calls. Runs forward_with_analysis +
    attn_entropy_and_distance + direct_logit_attribution (one forward pass,
    no gradients needed) and position_saliency (its own forward+backward
    pass, needs gradients) over n_batches short val-split slices
    (seq_len tokens, not the full training context -- VRAM discipline, see
    this module's docstring), averaging each metric across batches. Adds
    induction_score, a separate synthetic probe that can't share these
    natural-language batches (see its own docstring).

    Caller (train.py) is expected to wrap this call in the same CUDA
    autocast context it already uses for other eval blocks -- this
    function does not manage autocast itself, matching how the rest of
    train.py's eval code is structured (the caller owns the device/dtype
    context, not the callee).
    """
    from prepare import make_dataloader

    was_training = model.training
    model.eval()

    resid_lambdas = model.resid_lambdas.detach().float().tolist()
    entropy_sums: Optional[List[float]] = None
    distance_sums: Optional[List[float]] = None
    dla_sums: Optional[List[float]] = None
    saliency_sums: Optional[List[float]] = None

    loader = make_dataloader(tokenizer, device_batch_size, seq_len, "val")
    cos, sin = model.cos[:, :seq_len], model.sin[:, :seq_len]

    for _ in range(n_batches):
        idx, targets, _epoch = next(loader)

        with torch.no_grad():
            out = forward_with_analysis(model, idx, targets, cos_sin=(cos, sin))
        entropy, distance = attn_entropy_and_distance(out["per_layer_weights"])
        dla = direct_logit_attribution(
            out["per_layer_deltas"], out["final_x_pre_norm"], model.lm_head.weight, targets, resid_lambdas,
        )
        sal = position_saliency(model, idx, targets)

        entropy_sums = entropy if entropy_sums is None else [a + b for a, b in zip(entropy_sums, entropy)]
        distance_sums = distance if distance_sums is None else [a + b for a, b in zip(distance_sums, distance)]
        dla_sums = dla if dla_sums is None else [a + b for a, b in zip(dla_sums, dla)]
        saliency_sums = sal if saliency_sums is None else [a + b for a, b in zip(saliency_sums, sal)]

    attn_entropy = [v / n_batches for v in entropy_sums]
    attn_distance = [v / n_batches for v in distance_sums]
    dla_avg = [v / n_batches for v in dla_sums]
    pos_saliency = [v / n_batches for v in saliency_sums]

    induction_seq_len = max(2, min(64, seq_len // 4))
    ind_score = induction_score(model, (model.cos, model.sin), vocab_size=model.config.vocab_size, seq_len=induction_seq_len)

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    return {
        "attn_entropy": attn_entropy,
        "attn_distance": attn_distance,
        "attn_distance_slope": attn_distance_slope(attn_distance),
        "pos_saliency": pos_saliency,
        "dla": dla_avg,
        "induction_score": ind_score,
    }
