"""Tier 2 token-level XAI (agents/xai_methods/token_methods.py).

train.py executes training eagerly at import time, so it can't be
`import`ed directly in CPU-only tests. These tests exec only the
class/function-definition prefix of the file (everything before the
training loop starts) to get GPT/GPTConfig/the real attention function,
the same technique used elsewhere in this session for CPU-safe train.py
testing.
"""
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from agents.xai_methods.token_methods import (
    attn_distance_slope,
    attn_entropy_and_distance,
    compute_behavioral_fingerprint,
    direct_logit_attribution,
    forward_with_analysis,
    induction_score,
    position_saliency,
)

import agents.xai_methods.token_methods as tm

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_train_defs():
    train_src = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    boundary = train_src.index("t_start = time.time()")
    ns = {"__name__": "train_defs_only", "__file__": str(REPO_ROOT / "train.py")}
    exec(compile(train_src[:boundary], str(REPO_ROOT / "train.py"), "exec"), ns)
    return ns["GPT"], ns["GPTConfig"], ns["_attn_func"]


_GPT, _GPTConfig, _real_attn_func = _load_train_defs()


def _make_toy_model(n_layer=3, n_head=2, n_kv_head=2, n_embd=32, vocab_size=64,
                     sequence_len=32, window_pattern="L", seed=0, resid_lambdas=None, x0_lambdas=None):
    torch.manual_seed(seed)
    cfg = _GPTConfig(
        sequence_len=sequence_len, vocab_size=vocab_size, n_layer=n_layer,
        n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd, window_pattern=window_pattern,
    )
    model = _GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            # init_weights() zero-initializes these (residual-safe init) --
            # real signal is needed for any of these tests to mean anything.
            block.attn.c_proj.weight.normal_(mean=0.0, std=0.02)
            block.mlp.c_proj.weight.normal_(mean=0.0, std=0.02)
        if resid_lambdas is not None:
            model.resid_lambdas.copy_(torch.tensor(resid_lambdas[:n_layer]))
        if x0_lambdas is not None:
            model.x0_lambdas.copy_(torch.tensor(x0_lambdas[:n_layer]))
    model.eval()
    return model, cfg


@pytest.fixture
def toy_model():
    return _make_toy_model()


# ---------------------------------------------------------------------------
# forward_with_analysis
# ---------------------------------------------------------------------------

def test_forward_with_analysis_matches_real_sdpa_attention(toy_model):
    """The analysis-only manual-softmax path must compute the same
    attention output as the real SDPA-based path on the same q/k/v -- the
    core "does the analysis path match training semantics" check.
    """
    model, cfg = toy_model
    B, T = 2, 12
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    cos_sin = (model.cos[:, :T], model.sin[:, :T])

    with torch.no_grad():
        x = tm._norm(model.transformer.wte(idx).float())
        block = model.transformer.h[0]
        attn = block.attn
        normed = tm._norm(model.resid_lambdas[0] * x + model.x0_lambdas[0] * x)
        q = attn.c_q(normed.to(attn.c_q.weight.dtype)).float().view(B, T, attn.n_head, attn.head_dim)
        k = attn.c_k(normed.to(attn.c_k.weight.dtype)).float().view(B, T, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(normed.to(attn.c_v.weight.dtype)).float().view(B, T, attn.n_kv_head, attn.head_dim)
        q = tm._norm(tm._apply_rotary_emb(q, cos_sin[0].float(), cos_sin[1].float()))
        k = tm._norm(tm._apply_rotary_emb(k, cos_sin[0].float(), cos_sin[1].float()))

        y_real = _real_attn_func(q, k, v, model.window_sizes[0]).contiguous().view(B, T, -1)

        q2, k2, v2 = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = 1.0 / math.sqrt(attn.head_dim)
        scores = (q2 @ k2.transpose(-2, -1)) * scale
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        y_manual = (weights @ v2).transpose(1, 2).contiguous().view(B, T, -1)

    diff = (y_real - y_manual).abs().max().item()
    assert diff < 1e-4, f"manual-softmax attention diverges from real SDPA attention by {diff} (fp32, should be near-exact)"


def test_forward_with_analysis_runs_and_returns_correct_shapes(toy_model):
    model, cfg = toy_model
    B, T = 2, 12
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    cos_sin = (model.cos[:, :T], model.sin[:, :T])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = forward_with_analysis(model, idx, targets, cos_sin)

    assert len(out["per_layer_weights"]) == cfg.n_layer
    assert len(out["per_layer_deltas"]) == cfg.n_layer
    assert out["per_layer_weights"][0].shape == (B, cfg.n_head, T, T)
    assert out["final_x_pre_norm"].shape == (B, T, cfg.n_embd)
    assert out["x0"].shape == (B, T, cfg.n_embd)
    assert out["loss"] is not None and torch.isfinite(out["loss"])


def test_forward_with_analysis_works_under_no_grad(toy_model):
    """Regression test: retain_grad() used to be called unconditionally,
    which crashes when the caller wraps the call in torch.no_grad() (as
    position_saliency's own internal helpers, and callers that don't need
    gradients, are documented to do).
    """
    model, cfg = toy_model
    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = forward_with_analysis(model, idx, targets=None, cos_sin=cos_sin)
    assert out["logits"].shape == (B, T, cfg.vocab_size)


# ---------------------------------------------------------------------------
# attn_entropy_and_distance / attn_distance_slope
# ---------------------------------------------------------------------------

def test_attn_entropy_and_distance_hand_computed_uniform():
    T = 4
    uniform_causal = torch.zeros(1, 1, T, T)
    for i in range(T):
        uniform_causal[0, 0, i, :i + 1] = 1.0 / (i + 1)
    entropy, _distance = attn_entropy_and_distance([uniform_causal])
    expected = sum(math.log(i + 1) for i in range(1, T)) / (T - 1)
    assert abs(entropy[0] - expected) < 1e-5


def test_attn_entropy_and_distance_hand_computed_peaked():
    T = 4
    peaked = torch.zeros(1, 1, T, T)
    for i in range(T):
        peaked[0, 0, i, 0] = 1.0
    entropy, distance = attn_entropy_and_distance([peaked])
    assert abs(entropy[0]) < 1e-6
    expected_dist = sum(range(1, T)) / (T - 1)
    assert abs(distance[0] - expected_dist) < 1e-5


def test_attn_distance_slope():
    assert abs(attn_distance_slope([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-9
    assert abs(attn_distance_slope([2.0, 2.0, 2.0])) < 1e-9
    assert attn_distance_slope([1.0]) == 0.0


# ---------------------------------------------------------------------------
# position_saliency
# ---------------------------------------------------------------------------

def test_position_saliency_matches_finite_difference(toy_model):
    """Gradient correctness check: the analytic grad x input saliency must
    match a numerical (finite-difference) derivative of the last-position
    loss w.r.t. each position's embedding. fp32/eps=0.1 (calibrated -- the
    intentional fp32 upcast inside forward_with_analysis puts a real floor
    on how small eps can be before loss_plus/loss_minus round to
    bit-identical fp32 values).
    """
    model, cfg = toy_model
    B, T = 1, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    cos, sin = model.cos[:, :T].float(), model.sin[:, :T].float()

    def last_position_loss_from_emb(emb_tensor):
        out = forward_with_analysis(model, idx, targets=None, cos_sin=(cos, sin), emb_override=emb_tensor)
        return F.cross_entropy(out["logits"][:, -1, :], targets[:, -1])

    emb0 = model.transformer.wte(idx).float().detach().clone().requires_grad_(True)
    loss = last_position_loss_from_emb(emb0)
    model.zero_grad(set_to_none=True)
    loss.backward()
    analytic_grad = emb0.grad.clone()

    torch.manual_seed(1)
    eps = 0.1
    for p in range(T - 1):
        r = torch.randn(cfg.n_embd)
        r = r / r.norm()
        emb_plus = emb0.detach().clone()
        emb_plus[0, p, :] += eps * r
        emb_minus = emb0.detach().clone()
        emb_minus[0, p, :] -= eps * r
        with torch.no_grad():
            loss_plus = last_position_loss_from_emb(emb_plus).item()
            loss_minus = last_position_loss_from_emb(emb_minus).item()
        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = (analytic_grad[0, p, :] * r).sum().item()

        same_sign = (numeric * analytic) >= 0
        abs_err = abs(numeric - analytic)
        denom = max(abs(numeric), abs(analytic), 1e-6)
        rel_err = abs_err / denom
        assert same_sign, f"position {p}: analytic ({analytic}) and numeric ({numeric}) gradients disagree in sign"
        assert abs_err < 3e-6 or rel_err < 0.30, (
            f"position {p}: gradient mismatch too large (numeric={numeric}, analytic={analytic}, rel_err={rel_err:.2%})"
        )


def test_position_saliency_matches_direct_reimplementation(toy_model):
    model, cfg = toy_model
    B, T = 1, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    cos, sin = model.cos[:, :T], model.sin[:, :T]

    n_buckets = T - 1
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sal = position_saliency(model, idx, targets, n_buckets=n_buckets)
        out = forward_with_analysis(model, idx, targets=None, cos_sin=(cos, sin))
        loss_direct = F.cross_entropy(out["logits"][:, -1, :], targets[:, -1])
    model.zero_grad(set_to_none=True)
    loss_direct.backward()
    direct_saliency = (out["emb"].grad * out["emb"].detach()).abs().sum(dim=-1)[0]

    recomputed_from_buckets = [0.0] * (T - 1)
    for b in range(n_buckets):
        recomputed_from_buckets[T - 2 - b] = sal[b]

    max_diff = max(abs(direct_saliency[p].item() - recomputed_from_buckets[p]) for p in range(T - 1))
    assert max_diff < 1e-4


def test_position_saliency_shape_and_range(toy_model):
    model, cfg = toy_model
    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sal = position_saliency(model, idx, targets, n_buckets=T - 1)
    assert len(sal) == T - 1
    assert all(math.isfinite(s) and s >= 0.0 for s in sal)


# ---------------------------------------------------------------------------
# direct_logit_attribution
# ---------------------------------------------------------------------------

def test_direct_logit_attribution_reconstructs_residual_stream():
    model, cfg = _make_toy_model(
        n_layer=4, window_pattern="SL",
        resid_lambdas=[0.9, 1.05, 0.8, 1.1], x0_lambdas=[0.15, 0.05, 0.2, 0.1],
    )
    B, T = 3, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    cos, sin = model.cos[:, :T], model.sin[:, :T]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = forward_with_analysis(model, idx, targets=None, cos_sin=(cos, sin))

    resid_lambdas = model.resid_lambdas.detach().float().tolist()
    x0_lambdas = model.x0_lambdas.detach().float().tolist()
    x0 = out["x0"].float()
    final_x_pre_norm = out["final_x_pre_norm"].float()

    z = x0.clone()
    for i in range(cfg.n_layer):
        block_input = resid_lambdas[i] * z + x0_lambdas[i] * x0
        z = block_input + out["per_layer_deltas"][i].float()

    recon_rel_diff = (z - final_x_pre_norm).abs().max().item() / final_x_pre_norm.abs().max().item()
    # bf16 forward pass accumulates rounding noise across n_layer steps of
    # this recurrence; a few percent relative error is expected precision
    # noise, not a logic bug (a missing/wrong lambda term would be structural).
    assert recon_rel_diff < 0.03, f"x0 + per_layer_deltas, replayed via the real recurrence, diverged from final_x_pre_norm by {recon_rel_diff:.2%}"


def test_direct_logit_attribution_conservation():
    model, cfg = _make_toy_model(
        n_layer=4, window_pattern="SL",
        resid_lambdas=[0.9, 1.05, 0.8, 1.1], x0_lambdas=[0.15, 0.05, 0.2, 0.1],
    )
    B, T = 3, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    cos, sin = model.cos[:, :T], model.sin[:, :T]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = forward_with_analysis(model, idx, targets=None, cos_sin=(cos, sin))

    resid_lambdas = model.resid_lambdas.detach().float().tolist()
    x0_lambdas = model.x0_lambdas.detach().float().tolist()
    final_x_pre_norm = out["final_x_pre_norm"].float()
    final_x = out["final_x"].float()
    x0 = out["x0"].float()

    dla = direct_logit_attribution(out["per_layer_deltas"], final_x_pre_norm, model.lm_head.weight, targets, resid_lambdas)
    assert len(dla) == cfg.n_layer
    assert all(math.isfinite(v) for v in dla)

    rms = final_x_pre_norm.pow(2).mean(dim=-1, keepdim=True).sqrt()
    norm_scale = 1.0 / rms.clamp_min(1e-6)
    target_directions = model.lm_head.weight[targets.clamp_min(0)].float()

    n_layer = cfg.n_layer
    cumulative = [1.0] * n_layer
    running = 1.0
    for i in range(n_layer - 1, -1, -1):
        cumulative[i] = running
        running *= resid_lambdas[i]
    x0_total_mult = cumulative[0] * resid_lambdas[0] + sum(x0_lambdas[i] * cumulative[i] for i in range(n_layer))

    x0_contribution = (x0 * norm_scale * x0_total_mult * target_directions).sum(dim=-1)
    per_layer_contribution = torch.zeros(B, T)
    for i, delta in enumerate(out["per_layer_deltas"]):
        per_layer_contribution += (delta.float() * cumulative[i] * norm_scale * target_directions).sum(dim=-1)

    predicted_total = x0_contribution + per_layer_contribution
    actual_target_logit = (final_x * target_directions).sum(dim=-1)

    valid = targets != -1
    max_rel_diff = (predicted_total - actual_target_logit)[valid].abs().max().item() / actual_target_logit[valid].abs().mean().item()
    assert max_rel_diff < 0.05, f"DLA sum + x0 contribution diverges from the real target logit by {max_rel_diff:.2%}"


# ---------------------------------------------------------------------------
# induction_score
# ---------------------------------------------------------------------------

def test_induction_score_structural(toy_model):
    model, cfg = toy_model
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        score = induction_score(model, (model.cos, model.sin), vocab_size=cfg.vocab_size, seq_len=12, seed=0)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_induction_score_detects_planted_offset_and_rejects_wrong_offset(toy_model, monkeypatch):
    """Positive/negative control: a hand-planted attention spike at the
    correct induction offset (j+1) must be detected; the same spike shifted
    by one position (j) must not be mistaken for it -- catches an
    off-by-one in the offset indexing that a purely structural check
    ("runs, returns a sane value") would miss.
    """
    model, cfg = toy_model
    dummy_cos = torch.zeros(1, 200, 1, 4)
    dummy_sin = torch.zeros(1, 200, 1, 4)

    def make_fake_forward(spike_offset_delta):
        def fake_forward_with_analysis(model_, idx, targets, cos_sin):
            B, T = idx.shape
            seq_len_ = T // 2
            weights = torch.zeros(B, 2, T, T)
            for q in range(T):
                weights[:, :, q, :q + 1] = 1.0 / (q + 1)
            for j in range(seq_len_ - 1):
                k_pos = j + spike_offset_delta
                if 0 <= k_pos <= (seq_len_ + j):
                    weights[:, 0, seq_len_ + j, k_pos] = 0.9
            return {"per_layer_weights": [weights]}
        return fake_forward_with_analysis

    monkeypatch.setattr(tm, "forward_with_analysis", make_fake_forward(spike_offset_delta=1))
    positive_score = tm.induction_score(model, (dummy_cos, dummy_sin), vocab_size=cfg.vocab_size, seq_len=12, seed=0)
    assert abs(positive_score - 0.9) < 1e-6

    monkeypatch.setattr(tm, "forward_with_analysis", make_fake_forward(spike_offset_delta=0))
    negative_score = tm.induction_score(model, (dummy_cos, dummy_sin), vocab_size=cfg.vocab_size, seq_len=12, seed=0)
    assert negative_score < 0.5


# ---------------------------------------------------------------------------
# compute_behavioral_fingerprint (entry point)
# ---------------------------------------------------------------------------

def test_compute_behavioral_fingerprint_schema(toy_model, monkeypatch):
    model, cfg = toy_model
    model.train()

    def fake_make_dataloader(tokenizer, B, T, split):
        def gen():
            while True:
                yield torch.randint(0, cfg.vocab_size, (B, T)), torch.randint(0, cfg.vocab_size, (B, T)), 1
        return gen()

    import prepare as prepare_module
    monkeypatch.setattr(prepare_module, "make_dataloader", fake_make_dataloader)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        fp = compute_behavioral_fingerprint(model, tokenizer=None, device_batch_size=2, seq_len=16, n_batches=2)

    assert set(fp.keys()) == {"attn_entropy", "attn_distance", "attn_distance_slope", "pos_saliency", "dla", "induction_score"}
    assert len(fp["attn_entropy"]) == cfg.n_layer
    assert len(fp["attn_distance"]) == cfg.n_layer
    assert len(fp["dla"]) == cfg.n_layer
    assert len(fp["pos_saliency"]) == 16
    for k, v in fp.items():
        vals = v if isinstance(v, list) else [v]
        assert all(math.isfinite(x) for x in vals), f"{k} has a non-finite value: {v}"
    assert model.training, "compute_behavioral_fingerprint must restore the caller's original training mode"


# ---------------------------------------------------------------------------
# train.py wiring: exact block, exec'd against a controlled namespace
# ---------------------------------------------------------------------------

def test_train_py_token_xai_block_is_a_true_noop_when_disabled(toy_model, monkeypatch, tmp_path):
    model, cfg = toy_model
    model.train()

    train_text = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    start = train_text.index('try:\n    _token_xai_hp = {}')
    end = train_text.index("sys.stdout.flush()\n", start) + len("sys.stdout.flush()\n")
    block_src = train_text[start:end]

    import io
    import contextlib
    import os as os_module
    import sys as sys_module
    import yaml as yaml_module

    hp_path = tmp_path / "does_not_exist.yaml"
    block_ns = {
        "os": os_module, "sys": sys_module, "json": json, "_yaml": yaml_module,
        "_hp_path": str(hp_path), "MAX_SEQ_LEN": 2048, "DEVICE_BATCH_SIZE": 2,
        "autocast_ctx": torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        "model": model, "tokenizer": None,
        "x0_lambdas": model.x0_lambdas.detach().float().tolist(),
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(block_src, "<token_xai_block>", "exec"), block_ns)
    output = buf.getvalue()
    assert "token_fingerprint:" not in output
    assert "[token_fingerprint] Could not compute" not in output


def test_train_py_token_xai_block_computes_fingerprint_when_enabled(toy_model, monkeypatch, tmp_path):
    model, cfg = toy_model
    model.train()

    def fake_make_dataloader(tokenizer, B, T, split):
        def gen():
            while True:
                yield torch.randint(0, cfg.vocab_size, (B, T)), torch.randint(0, cfg.vocab_size, (B, T)), 1
        return gen()

    import prepare as prepare_module
    monkeypatch.setattr(prepare_module, "make_dataloader", fake_make_dataloader)

    train_text = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    start = train_text.index('try:\n    _token_xai_hp = {}')
    end = train_text.index("sys.stdout.flush()\n", start) + len("sys.stdout.flush()\n")
    block_src = train_text[start:end]

    import io
    import contextlib
    import os as os_module
    import sys as sys_module
    import yaml as yaml_module

    hp_path = tmp_path / "model_hyperparams.yaml"
    hp_path.write_text("token_xai_enabled: true\ntoken_xai_seq_len: 16\ntoken_xai_n_batches: 2\n")
    block_ns = {
        "os": os_module, "sys": sys_module, "json": json, "_yaml": yaml_module,
        "_hp_path": str(hp_path), "MAX_SEQ_LEN": 2048, "DEVICE_BATCH_SIZE": 2,
        "autocast_ctx": torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        "model": model, "tokenizer": None,
        "x0_lambdas": model.x0_lambdas.detach().float().tolist(),
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(block_src, "<token_xai_block>", "exec"), block_ns)
    output = buf.getvalue()
    assert "[token_fingerprint] Could not compute" not in output, output
    line = next(l for l in output.splitlines() if l.startswith("token_fingerprint: "))
    fp = json.loads(line[len("token_fingerprint: "):])
    assert set(fp.keys()) == {"attn_entropy", "attn_distance", "attn_distance_slope", "pos_saliency", "dla", "induction_score", "x0_lambda"}
    assert len(fp["x0_lambda"]) == cfg.n_layer
