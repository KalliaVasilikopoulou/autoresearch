"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"
# Enable progress bars for visibility
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import math
import sys
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

def _attn_func(q, k, v, window_size):
    """
    Attention dispatcher (PyTorch SDPA only).
    q/k/v come in as [B, T, H, D]; transposes to [B, H, T, D], handles GQA + sliding window.
    """
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    q2 = q.transpose(1, 2)          # [B, Hq, T, D]
    k2 = k.transpose(1, 2)          # [B, Hkv, T, D]
    v2 = v.transpose(1, 2)          # [B, Hkv, T, D]
    if Hkv < Hq:                    # expand GQA
        ratio = Hq // Hkv
        k2 = k2.repeat_interleave(ratio, dim=1)
        v2 = v2.repeat_interleave(ratio, dim=1)
    w = window_size[0]
    if w >= T:
        y = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
    else:
        mask = torch.ones(T, T, dtype=torch.bool, device=q2.device).tril()
        mask &= torch.ones(T, T, dtype=torch.bool, device=q2.device).tril().triu(-(w - 1))
        y = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=mask)
    return y.transpose(1, 2)        # [B, T, Hq, D]

from prepare import (
    MAX_SEQ_LEN, MAX_TRAIN_SECONDS, TOKEN_BUDGET, Tokenizer, make_dataloader,
    evaluate_bpb, get_token_bytes,
)
# Also as a module: EVAL_TOKENS has to be read and written THROUGH it, because
# evaluate_bpb resolves that name from prepare's own globals. Rebinding a
# from-import copy here would change nothing, and would do it silently.
import prepare as _prepare

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = _attn_func(q, k, v, window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    # Ensure all operands match dtype of second_momentum_buffer
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 
                                 (1 - beta2).to(dtype=second_momentum_buffer.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step']).to(device=p.device, dtype=p.dtype)
            self._adamw_lr_t.fill_(group['lr']).to(device=p.device, dtype=p.dtype)
            self._adamw_beta1_t.fill_(group['betas'][0]).to(device=p.device, dtype=p.dtype)
            self._adamw_beta2_t.fill_(group['betas'][1]).to(device=p.device, dtype=p.dtype)
            self._adamw_eps_t.fill_(group['eps']).to(device=p.device, dtype=p.dtype)
            self._adamw_wd_t.fill_(group['weight_decay']).to(device=p.device, dtype=p.dtype)
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t.to(device=p.device, dtype=p.dtype),
                            self._adamw_lr_t.to(device=p.device, dtype=p.dtype),
                            self._adamw_beta1_t.to(device=p.device, dtype=p.dtype),
                            self._adamw_beta2_t.to(device=p.device, dtype=p.dtype),
                            self._adamw_eps_t.to(device=p.device, dtype=p.dtype),
                            self._adamw_wd_t.to(device=p.device, dtype=p.dtype))

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"]).to(device=device, dtype=dtype)
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0).to(device=device, dtype=dtype)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5).to(device=device, dtype=dtype)
        self._muon_wd_t.fill_(group["weight_decay"]).to(device=device, dtype=dtype)
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t.to(device=device, dtype=dtype),
                        self._muon_lr_t.to(device=device, dtype=dtype),
                        self._muon_wd_t.to(device=device, dtype=dtype),
                        self._muon_beta2_t.to(device=device, dtype=dtype),
                        group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO (before snapping to N_HEAD)
N_HEAD = 4              # number of attention heads (model_dim is snapped to a multiple of this)
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**13 # ~8K tokens per optimizer step (heavily reduced for VRAM efficiency with SDPA fallback)
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 1    # per-device batch size (2->1 for extreme memory efficiency; 1*2048 = 2K tokens per fwdbwd)

# Reproducibility. SEED controls exactly one thing: the initial weights drawn
# by GPT.init_weights(). The data stream is deterministic (prepare.py's
# _document_batches walks shards in sorted order and TOKEN_BUDGET cuts at a
# fixed point) and the packing dataloader draws no randomness, so nothing else
# in a run depends on it.
#
# It is a RECORDED NUISANCE VARIABLE, never a search dimension: it is
# deliberately absent from agent1's SEARCH_SPACE and from
# results_analysis.HYPERPARAM_COLUMNS. Searching over seeds would find the
# luckiest initialization, which is precisely the overfitting the search is
# trying to avoid -- the correct treatment is to average over seeds, not to
# optimize over them.
#
# It was hardcoded to 42 until now, which meant every run in the campaign
# shared one initialization and no measurement of seed-to-seed spread was
# possible. Note also that a shared seed is NOT a shared initialization across
# architectures: changing n_layer/n_embd/n_head changes how many values are
# drawn and in what shapes, so only runs at identical architecture are truly
# paired.
SEED = 42

# ---------------------------------------------------------------------------
# Override hyperparameters from model_hyperparams.yaml (written by Agent 1)
# All other training constants above are kept as Karpathy's calibrated defaults.
# Every value read here is clamped to a safe range before use — this file is
# written by an autonomous agent and must never be trusted blindly (a bad
# value here should degrade gracefully, not silently corrupt or crash a run).
# ---------------------------------------------------------------------------

_clamp_records = {}  # name -> {"requested": ..., "clamped": ..., "bounds": [lo, hi]}, reported back as
                      # `hyperparam_clamps:` below so Agent 1 can tell "what caused this extreme value"
                      # from structured data instead of it being a mystery (dev/inpsect_workflow_ideas.txt).

def _clamp(name, value, lo, hi):
    clamped = max(lo, min(hi, value))
    if clamped != value:
        print(f"[hyperparams] WARNING: {name}={value} outside safe range [{lo}, {hi}], clamped to {clamped}")
        _clamp_records[name] = {"requested": value, "clamped": clamped, "bounds": [lo, hi]}
    return clamped


def _build_window_pattern(n_layer, s_fraction):
    """Tier 4 (see dev/INNOVATION_PLAN.md): turn a continuous window_s_fraction
    in [0,1] -- the tunable Agent 1 actually searches -- into an actual S/L
    pattern string, one character per layer. Evenly interleaves the S layers
    among the L layers (a Bresenham-style even distribution) rather than
    blocking them at the start or end, to avoid an untested asymmetry.
    _compute_window_sizes already force-overrides the last layer to "L"
    regardless of this string, so no special-casing needed for that here.
    """
    if n_layer <= 0:
        return "L"
    n_s = max(0, min(n_layer, round(n_layer * s_fraction)))
    # Integer-arithmetic Bresenham (not floating-point accumulation of
    # n_s/n_layer per step): a float step can drift below the 1.0 trigger by
    # the last iteration (e.g. n_layer=7 sums 1/7 seven times to
    # 0.9999999999999998, silently dropping the last S) -- this is exact.
    pattern = []
    remainder = 0
    for _ in range(n_layer):
        remainder += n_s
        if remainder >= n_layer:
            pattern.append("S")
            remainder -= n_layer
        else:
            pattern.append("L")
    return "".join(pattern)

_target_embd = DEPTH * ASPECT_RATIO
_hp = {}
try:
    import yaml as _yaml
    # AUTORESEARCH_HP_PATH lets a caller point this process at a per-run
    # hyperparams file instead of the shared default -- needed so multiple
    # concurrent train.py processes (one per GPU) never race on one file.
    _hp_path = os.environ.get("AUTORESEARCH_HP_PATH") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_hyperparams.yaml"
    )
    if os.path.exists(_hp_path):
        with open(_hp_path) as _f:
            _hp = _yaml.safe_load(_f) or {}
        if "n_layer" in _hp:
            DEPTH = _clamp("n_layer", int(_hp["n_layer"]), 1, 48)
        if "n_head" in _hp:
            N_HEAD = _clamp("n_head", int(_hp["n_head"]), 1, 64)
        _target_embd = DEPTH * ASPECT_RATIO
        if "n_embd" in _hp:
            _target_embd = _clamp("n_embd", int(_hp["n_embd"]), N_HEAD, 8192)
        if "embedding_lr" in _hp:
            EMBEDDING_LR = _clamp("embedding_lr", float(_hp["embedding_lr"]), 0.05, 3.0)
        if "unembedding_lr" in _hp:
            UNEMBEDDING_LR = _clamp("unembedding_lr", float(_hp["unembedding_lr"]), 0.0005, 0.02)
        if "matrix_lr" in _hp:
            MATRIX_LR = _clamp("matrix_lr", float(_hp["matrix_lr"]), 0.005, 0.2)
        if "scalar_lr" in _hp:
            SCALAR_LR = _clamp("scalar_lr", float(_hp["scalar_lr"]), 0.05, 2.0)
        if "weight_decay" in _hp:
            WEIGHT_DECAY = _clamp("weight_decay", float(_hp["weight_decay"]), 0.0, 2.0)
        if "warmup_ratio" in _hp:
            WARMUP_RATIO = _clamp("warmup_ratio", float(_hp["warmup_ratio"]), 0.0, 1.0)
        if "batch_size" in _hp:
            _tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
            _raw_batch = _clamp("batch_size", int(_hp["batch_size"]), _tokens_per_fwdbwd, 2**20)
            TOTAL_BATCH_SIZE = max(_tokens_per_fwdbwd, round(_raw_batch / _tokens_per_fwdbwd) * _tokens_per_fwdbwd)
        _window_s_fraction = 0.75  # matches the old hardcoded WINDOW_PATTERN="SSSL" (3-of-4 = 75% S)
        if "window_s_fraction" in _hp:
            _window_s_fraction = _clamp("window_s_fraction", float(_hp["window_s_fraction"]), 0.0, 1.0)
        WINDOW_PATTERN = _build_window_pattern(DEPTH, _window_s_fraction)
        if "seed" in _hp:
            SEED = _clamp("seed", int(_hp["seed"]), 0, 2**31 - 1)
        # THE TWO BUDGETS. Overridable so they can be MEASURED against wall
        # clock rather than argued about, and so a probe can vary one without
        # editing prepare.py on the remote.
        #
        # Both are CAMPAIGN CONSTANTS, never search dimensions -- like `seed`,
        # and for a sharper reason: val_bpb is only comparable between runs
        # that saw the same amount of training and were scored on the same
        # amount of validation. A search allowed to vary either would discover
        # that training less looks better, because a shorter run is a
        # different question, not a better answer. Neither appears in
        # SEARCH_SPACE or HYPERPARAM_COLUMNS, and both are echoed back below so
        # results.tsv records what was actually used.
        if "token_budget" in _hp:
            TOKEN_BUDGET = _clamp("token_budget", int(_hp["token_budget"]),
                                  100_000, 1_000_000_000)
        if "eval_tokens" in _hp:
            # Written through the module, not into a local name: evaluate_bpb
            # reads prepare.EVAL_TOKENS from its own module globals, so
            # rebinding a copy here would change nothing and silently.
            _prepare.EVAL_TOKENS = _clamp("eval_tokens", int(_hp["eval_tokens"]),
                                          65_536, 1_000_000_000)
        print(f"[hyperparams] DEPTH={DEPTH} N_HEAD={N_HEAD} target_n_embd={_target_embd} "
              f"EMBEDDING_LR={EMBEDDING_LR} UNEMBEDDING_LR={UNEMBEDDING_LR} MATRIX_LR={MATRIX_LR} SCALAR_LR={SCALAR_LR} "
              f"WEIGHT_DECAY={WEIGHT_DECAY} WARMUP_RATIO={WARMUP_RATIO} TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} "
              f"WINDOW_PATTERN={WINDOW_PATTERN} (window_s_fraction={_window_s_fraction}) SEED={SEED}")
except Exception as _e:
    print(f"[hyperparams] Could not load model_hyperparams.yaml: {_e} — using defaults")

# Snap the target embedding size to a multiple of N_HEAD so head_dim stays an
# integer, AND make sure head_dim itself is even -- apply_rotary_emb splits
# each head into two equal halves (it rotates 2D pairs), so an odd head_dim
# makes the halves mismatched sizes and crashes. This is a real constraint
# of RoPE, not an arbitrary limit, so we snap up to the nearest even head_dim
# rather than relaxing anything.
_head_dim = max(1, round(_target_embd / N_HEAD))
if _head_dim % 2 != 0:
    _head_dim += 1
MODEL_DIM = _head_dim * N_HEAD

# Loud, end-to-end: if model_hyperparams.yaml requested a specific n_embd and
# what actually got used (after range-clamping AND the head_dim-parity snap
# above) differs, record it under the same "n_embd" key -- this is exactly
# the gap that made a proposal like n_embd=473/n_head=11 look mysterious
# (silently became 484 with nothing telling Agent 1 afterward).
if "n_embd" in _hp and MODEL_DIM != int(_hp["n_embd"]):
    _clamp_records["n_embd"] = {"requested": int(_hp["n_embd"]), "clamped": MODEL_DIM, "bounds": [N_HEAD, 8192]}

if _clamp_records:
    print("hyperparam_clamps: " + json.dumps(_clamp_records))

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth, n_head, model_dim):
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=n_head, n_kv_head=n_head, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH, N_HEAD, MODEL_DIM)
print(f"Model config: {asdict(config)}")
sys.stdout.flush()

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# ---------------------------------------------------------------------------
# Opt-in initialization probe (`init_probe: true` in the hyperparams file).
# Prints a per-tensor hash of the freshly-initialized weights and exits BEFORE
# the dataloader, training and eval -- seconds, not minutes, and it consumes no
# training budget.
#
# It exists to verify the load-bearing claim of the multi-region design: the
# RNG draw order above depends only on vocab_size, n_embd and n_layer, so two
# configurations sharing those (and the seed) should start from BIT-IDENTICAL
# weights no matter how their learning rates, batch size, weight decay, warmup
# or window pattern differ. That is what makes a within-region comparison
# paired, which in turn is what keeps the region's noise floor low enough for
# anything inside it to be measurable.
#
# Per-tensor rather than one whole-model hash on purpose: n_head reshapes
# ve_gate (initialized to zeros, consuming no randomness), so a whole-model
# hash would differ for a reason that says nothing about the RNG stream.
# Tensor-by-tensor shows exactly which weights match and which merely changed
# shape.
#
# Cast to float32 before hashing -- bfloat16 -> float32 is exact and lossless,
# so identical bytes here mean identical weights, while .numpy() on bfloat16
# is not supported at all.
# ---------------------------------------------------------------------------
if _hp.get("init_probe"):
    import hashlib

    _tensor_hashes = {}
    for _name, _param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        _raw = _param.detach().float().cpu().contiguous().numpy().tobytes()
        _tensor_hashes[_name] = {
            "sha": hashlib.sha256(_raw).hexdigest()[:16],
            "shape": list(_param.shape),
        }
    print("init_probe: " + json.dumps({
        "seed": SEED,
        "n_layer": DEPTH,
        "n_embd": MODEL_DIM,
        "n_head": N_HEAD,
        "vocab_size": vocab_size,
        "n_tensors": len(_tensor_hashes),
        "tensors": _tensor_hashes,
    }))
    sys.stdout.flush()
    sys.exit(0)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
sys.stdout.flush()

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

print("[train.py] torch.compile skipped (SDPA-only build, compiled SDPA is memory-inefficient)")
sys.stdout.flush()

print("[train.py] Initializing dataloader...")
sys.stdout.flush()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
print("[train.py] Dataloader created, prefetching first batch...")
sys.stdout.flush()
x, y, epoch = next(train_loader)  # prefetch first batch
print("[train.py] First batch prefetched successfully")
sys.stdout.flush()

print(f"Token budget: {TOKEN_BUDGET:,} tokens (safety cap {MAX_TRAIN_SECONDS}s)")
print(f"Gradient accumulation steps: {grad_accum_steps}")
sys.stdout.flush()

# Schedules (all based on progress = tokens_seen / TOKEN_BUDGET). Deliberately
# tokens, not wall-clock: a contended run used to advance its LR schedule on
# elapsed seconds, so it decayed the learning rate across fewer optimizer
# steps than an uncontended one -- the schedule itself varied with how busy
# the shared server was. Against a token budget, two runs of the same config
# follow the identical schedule regardless of load.

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"[train.py] Starting training loop...")
sys.stdout.flush()

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    tokens_seen = (step + 1) * TOTAL_BATCH_SIZE
    progress = min(tokens_seen / TOKEN_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging with visual progress bar
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TOKEN_BUDGET - tokens_seen)

    # Visual progress bar (ASCII blocks)
    if step % 5 == 0:
        bar_filled = int(pct_done / 5)
        bar = '[' + '=' * bar_filled + '-' * (20 - bar_filled) + ']'
        print(f"{bar} {pct_done:5.1f}% | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | tok/sec: {tok_per_sec:,} | mfu: {mfu:5.1f}% | remaining: {remaining/1e6:5.2f}M tok")

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Budget spent — but only stop after warmup steps so we don't count compilation.
    if step > 10 and tokens_seen >= TOKEN_BUDGET:
        break

    # Safety valve, NOT the objective: a pathologically slow config (or a
    # badly contended GPU) must not stall the campaign indefinitely. A run
    # that ends here has NOT seen its full token budget, so its val_bpb is
    # not comparable to a complete run -- budget_shortfall_pct below is what
    # lets the search exclude it rather than silently rank it.
    if step > 10 and total_training_time >= MAX_TRAIN_SECONDS:
        print(f"\n[train.py] WARNING: hit the {MAX_TRAIN_SECONDS}s safety cap after "
              f"{tokens_seen/1e6:.2f}M of {TOKEN_BUDGET/1e6:.2f}M tokens -- this run is "
              f"INCOMPLETE and its val_bpb is not comparable to a full-budget run.")
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE
# 0.0 for a run that consumed its whole budget; >0 means it was cut short by
# the safety cap and must be excluded from comparisons, not ranked.
budget_shortfall_pct = max(0.0, 100.0 * (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET)

# Final eval. evaluate_bpb (prepare.py) now prints its own progress bar
# (same reasoning as the training loop's: this is a ~10k-step loop with
# lazy, unprefetched dataloading, and under multi-GPU parallel dispatch
# several runs can hit it at once and contend hard enough to stall for
# minutes -- silent that long would trip the SSH read timeout in
# agents/remote_runner.py and drop a training run that actually succeeded).
print("[train.py] Starting validation eval...")
model.eval()
t_start_eval = time.time()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
eval_seconds = time.time() - t_start_eval

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
# WHERE THE TIME ACTUALLY GOES. Only training_seconds was ever reported, and
# it is 61-65% of a run -- so a third of every run was unaccounted for and any
# decision about the token budget was being made half-blind. It matters
# because these three scale with completely different things: startup is
# roughly fixed, training scales with TOKEN_BUDGET, and eval scales with
# prepare.EVAL_TOKENS -- which is 21.0M, i.e. 1.68x the entire training
# budget. Cutting TOKEN_BUDGET alone therefore cannot take a run below
# startup + eval, however far it is cut.
print(f"startup_seconds:  {startup_time:.1f}")
print(f"eval_seconds:     {eval_seconds:.1f}")
# Echoed back, like `seed`: val_bpb is only comparable between runs that saw
# the same amount of training and were scored on the same amount of
# validation, so the two budgets a run ACTUALLY used have to be recoverable
# from its row rather than assumed from whatever prepare.py says today.
print(f"token_budget:     {TOKEN_BUDGET}")
print(f"eval_tokens:      {_prepare.EVAL_TOKENS}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"budget_shortfall_pct: {budget_shortfall_pct:.2f}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
# Reported back, not just requested: results.tsv logs the seed train.py
# ACTUALLY ran with (same pattern as `depth`), so a run can never be attributed
# to an initialization it did not use -- e.g. if the hyperparams file failed to
# load and the default silently applied.
print(f"seed:             {SEED}")

# ---------------------------------------------------------------------------
# Held-out shard check (opt-in, off by default). The search loop's val_bpb
# is measured against one pinned shard across 100+ accept/reject decisions —
# a multiple-comparisons problem. This re-evaluates on a shard the search
# never sees, but only when explicitly requested (scripts/holdout_eval.py
# sets holdout_eval: true for a small number of final top-K candidates —
# never for every run, since it doubles eval cost for no benefit otherwise).
# evaluate_bpb itself (the official metric) is untouched by this.
# ---------------------------------------------------------------------------

try:
    _cfg_holdout = {}
    if os.path.exists(_hp_path):
        with open(_hp_path) as _f:
            _cfg_holdout = _yaml.safe_load(_f) or {}
    # (a) explicit opt-in, used by scripts/holdout_eval.py for a batch of
    #     final top-K candidates.
    _do_holdout = bool(_cfg_holdout.get("holdout_eval", False))
    # (b) continuous drift tracking: the orchestrator passes the campaign's
    #     best-so-far val_bpb, and holdout is evaluated only when THIS run
    #     beat it -- i.e. exactly on a new best. Deciding it here rather
    #     than in the orchestrator is what makes it exact: train.py fuses
    #     training and eval into one process with no checkpoint
    #     save/reload, so once a run ends its model is gone and a new best
    #     can never be re-evaluated after the fact. (token_xai_enabled
    #     works around that same constraint by approximating -- it
    #     fingerprints the NEXT run after a new best. Here we don't have
    #     to approximate, because val_bpb is already known at this point
    #     in the same process.)
    _threshold = _cfg_holdout.get("holdout_eval_if_below")
    if not _do_holdout and _threshold is not None:
        try:
            _do_holdout = float(val_bpb) < float(_threshold)
            if _do_holdout:
                print(f"[holdout_eval] New best ({val_bpb:.6f} < {float(_threshold):.6f}) "
                      f"-- evaluating the held-out shard too")
        except (TypeError, ValueError):
            _do_holdout = False
    if _do_holdout:
        from prepare import evaluate_bpb_holdout
        model.eval()
        with autocast_ctx:
            holdout_bpb = evaluate_bpb_holdout(model, tokenizer, DEVICE_BATCH_SIZE)
        print(f"holdout_val_bpb:  {holdout_bpb:.6f}")
except Exception as _e:
    print(f"[holdout_eval] Could not evaluate holdout shard: {_e}")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Real interpretable signal for Agent 2 (no mocking, no fabricated numbers).
#
# 1) Free scalars: resid_lambdas/x0_lambdas/ve_gate are already-trained
#    parameters sitting in memory — reading them costs zero extra GPU time.
# 2) Head ablation: costs real GPU time (k+1 extra cheap eval passes), so it
#    only runs here, inside the same process that already has the trained
#    model, tokenizer, and val dataloader live — Agent 2 never receives a
#    model object (train.py never checkpoints), so this is the only place
#    real weight-based analysis is possible at all.
# Both are wrapped defensively: a failure here must never invalidate an
# otherwise-successful training run.
# ---------------------------------------------------------------------------

try:
    resid_lambdas = model.resid_lambdas.detach().float().cpu().tolist()
    x0_lambdas = model.x0_lambdas.detach().float().cpu().tolist()
    ve_gate_norm = {
        str(i): torch.norm(block.attn.ve_gate.weight.detach().float()).item()
        for i, block in enumerate(model.transformer.h)
        if block.attn.ve_gate is not None
    }
    print("interpretable_scalars: " + json.dumps({
        "resid_lambdas": resid_lambdas,
        "x0_lambdas": x0_lambdas,
        "ve_gate_norm": ve_gate_norm,
    }))
except Exception as _e:
    print(f"[interpretable_scalars] Could not extract: {_e}")
sys.stdout.flush()

try:
    from agents.xai_methods.fast_methods import FastXAIMethods

    _ablation_k = 3
    if os.path.exists(_hp_path):
        try:
            with open(_hp_path) as _f:
                _ablation_hp = _yaml.safe_load(_f) or {}
            _ablation_k = int(_ablation_hp.get("ablation_k", _ablation_k))
        except Exception:
            pass
    _ablation_k = max(0, min(_ablation_k, 20))  # bound worst-case extra eval passes

    if _ablation_k > 0:
        _ABLATION_EVAL_STEPS = 8  # cheap, fixed-size — independent of the official EVAL_TOKENS metric

        @torch.no_grad()
        def _cheap_eval_fn(m):
            m.eval()
            token_bytes = get_token_bytes(device="cuda")
            loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
            total_nats, total_bytes = 0.0, 0
            with autocast_ctx:
                for _ in range(_ABLATION_EVAL_STEPS):
                    ax, ay, _epoch = next(loader)
                    loss_flat = m(ax, ay, reduction='none').view(-1)
                    y_flat = ay.view(-1)
                    nbytes = token_bytes[y_flat]
                    mask = nbytes > 0
                    total_nats += (loss_flat * mask).sum().item()
                    total_bytes += nbytes.sum().item()
            m.train()
            return total_nats / (math.log(2) * total_bytes)

        xai = FastXAIMethods()
        head_ablation_impacts = xai.top_k_ablation_study(model, _cheap_eval_fn, k=_ablation_k)
        print("head_ablation_impacts: " + json.dumps(head_ablation_impacts))
except Exception as _e:
    print(f"[head_ablation] Could not run ablation study: {_e}")
sys.stdout.flush()

try:
    _token_xai_hp = {}
    if os.path.exists(_hp_path):
        try:
            with open(_hp_path) as _f:
                _token_xai_hp = _yaml.safe_load(_f) or {}
        except Exception:
            pass

    if bool(_token_xai_hp.get("token_xai_enabled", False)):
        from agents.xai_methods.token_methods import compute_behavioral_fingerprint

        _token_xai_seq_len = min(int(_token_xai_hp.get("token_xai_seq_len", 384)), MAX_SEQ_LEN)
        _token_xai_n_batches = max(1, min(int(_token_xai_hp.get("token_xai_n_batches", 4)), 20))

        with autocast_ctx:
            fingerprint = compute_behavioral_fingerprint(
                model, tokenizer, DEVICE_BATCH_SIZE,
                seq_len=_token_xai_seq_len, n_batches=_token_xai_n_batches,
            )
        fingerprint["x0_lambda"] = x0_lambdas  # reuse the Tier 0 scalar, don't recompute
        print("token_fingerprint: " + json.dumps(fingerprint))
except Exception as _e:
    print(f"[token_fingerprint] Could not compute: {_e}")
sys.stdout.flush()
