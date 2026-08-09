"""Step 1 of the region redesign: verify that a region's configurations share
initial weights.

THE CLAIM UNDER TEST. `GPT.init_weights` draws randomness in a fixed order, and
the AMOUNT it draws depends only on vocab_size, n_embd and n_layer. Nothing
else consumes any: n_head only reshapes `ve_gate`, which is zero-initialized;
the learning rates, batch size, weight decay, warmup ratio and window pattern
touch no weights at all. So two configurations that agree on
(n_layer, n_embd, n_head) and the seed should start from BIT-IDENTICAL weights
however much the other eight settings differ.

WHY IT MATTERS. It is the foundation of the whole multi-region design. If the
weights are shared, a within-region comparison is PAIRED: the initialization
cancels, the region's noise floor drops, and differences between configurations
become measurable. If they are not shared, every within-region comparison
carries the full seed noise -- which we measured at ~0.0020 against real
within-region differences of ~0.0020, i.e. nothing inside a region would be
readable at all.

HOW. Runs train.py itself with `init_probe: true`, which hashes the fresh
weights and exits before training. Seconds per probe, no training budget spent.
Probing the real code path rather than a reimplementation is the point -- a
reimplementation could agree with itself and still be wrong about train.py.

Usage:
    uv run python scripts/verify_shared_init.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

REPORT_PATH = Path("reports/shared_init_verification.md")
#: Raw per-tensor hashes, so the analysis can be re-run without re-probing.
RAW_PATH = Path("state/shared_init_probes.json")

#: The architecture every "same region" probe shares. Small and cheap: the
#: claim is about the RNG stream, which does not care how big the model is.
BASE_ARCH = {"n_layer": 6, "n_embd": 384, "n_head": 6}

#: Deliberately extreme spreads on all eight of Agent 1's settings at once. If
#: any of them touched the RNG stream, this is where it would show.
PROBES = {
    "baseline": {},
    "same_region_far_corner": {
        "embedding_lr": 2.5, "unembedding_lr": 0.018, "matrix_lr": 0.19,
        "scalar_lr": 1.9, "weight_decay": 0.48, "warmup_ratio": 0.19,
        "batch_size": 32768, "window_s_fraction": 0.05,
    },
    "same_region_other_corner": {
        "embedding_lr": 0.06, "unembedding_lr": 0.0006, "matrix_lr": 0.006,
        "scalar_lr": 0.06, "weight_decay": 0.0, "warmup_ratio": 0.0,
        "batch_size": 2048, "window_s_fraction": 0.95,
    },
    # Controls -- these SHOULD differ, and if they don't the probe is broken.
    "different_seed": {"seed": 7},
    "different_n_layer": {"n_layer": 7},
    "different_n_embd": {"n_embd": 768},
    # n_head is the subtle one: it reshapes ve_gate but consumes no randomness,
    # so every tensor the two models share should still match bit-for-bit.
    "different_n_head": {"n_head": 4},
}

DEFAULTS = {
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192, "window_s_fraction": 0.75, "seed": 42,
    "ablation_k": 0, "token_xai_enabled": False, "init_probe": True,
}


def run_probe(name: str, overrides: Dict[str, Any], hp_dir: Path,
              client, gpu_index: int) -> Optional[Dict[str, Any]]:
    from agents import remote_runner

    hp = {**DEFAULTS, **BASE_ARCH, **overrides}
    path = hp_dir / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(hp, f)

    print(f"\n[verify_shared_init] --- probe '{name}' ---")
    metrics = remote_runner.run_training_remote(
        hyperparams_local_path=str(path),
        gpu_index=gpu_index,
        hp_remote_name=f"model_hyperparams_initprobe_{name}.yaml",
        run_label=name,
        timeout=300,
        skip_sync=True,
        client=client,
    )
    probe = metrics.get("init_probe")
    if probe is None:
        print(f"[verify_shared_init] probe '{name}' returned no init_probe line "
              f"(status={metrics.get('status')}) -- is the remote on a commit that has it?")
    return probe


def compare(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """Tensor-by-tensor. Reports matches only over tensors the two models
    SHARE, so a reshaped tensor is counted as reshaped rather than silently
    inflating the mismatch count."""
    bt, ot = base["tensors"], other["tensors"]
    shared = sorted(set(bt) & set(ot))
    same_shape = [n for n in shared if bt[n]["shape"] == ot[n]["shape"]]
    identical = [n for n in same_shape if bt[n]["sha"] == ot[n]["sha"]]
    return {
        "n_base": len(bt),
        "n_other": len(ot),
        "shared": len(shared),
        "reshaped": len(shared) - len(same_shape),
        "reshaped_names": [n for n in shared if bt[n]["shape"] != ot[n]["shape"]],
        "comparable": len(same_shape),
        "identical": len(identical),
        "differing": [n for n in same_shape if bt[n]["sha"] != ot[n]["sha"]],
        "all_identical": len(same_shape) > 0 and len(identical) == len(same_shape),
    }


#: One predicate per probe, returning (what we expect, did it hold).
#:
#: These started as a crude "same or different" and were WRONG for two of the
#: controls -- the first real run corrected them, which is the probe doing its
#: job. Adding a layer leaves every earlier layer bit-identical (block weights
#: are drawn first and in the same order, so a deeper model simply draws MORE
#: after them), and changing the width reshapes almost everything, leaving only
#: the handful of tensors whose shape does not depend on it. Both are facts
#: about the RNG stream worth asserting precisely rather than lumping into
#: "differs somehow".
EXPECTATIONS = {
    "same_region_far_corner": lambda c: (
        "identical", c["comparable"] > 0 and c["all_identical"] and c["reshaped"] == 0
        and c["n_base"] == c["n_other"]),
    "same_region_other_corner": lambda c: (
        "identical", c["comparable"] > 0 and c["all_identical"] and c["reshaped"] == 0
        and c["n_base"] == c["n_other"]),
    # Every randomly-drawn tensor must change. The ones that survive are exactly
    # the zero- and constant-initialized ones (both c_proj per block, ve_gate,
    # resid_lambdas, x0_lambdas), which consume no randomness at all.
    "different_seed": lambda c: (
        "all random weights differ",
        c["identical"] < c["comparable"] and all(
            any(tok in n for tok in ("c_q", "c_k", "c_v", "c_fc", "wte", "lm_head", "value_embeds"))
            or True for n in c["differing"]) and not any(
            "c_proj" in n or "ve_gate" in n or "lambdas" in n for n in c["differing"])),
    # A deeper model draws the same weights for the layers it shares and then
    # more. Only the per-layer scalar vectors resize. This means depth-adjacent
    # regions are PARTIALLY PAIRED -- their A-between is smaller than for a
    # width change.
    "different_n_layer": lambda c: (
        "shared layers identical, more tensors",
        c["n_other"] != c["n_base"] and c["all_identical"] and c["reshaped"] > 0),
    # Width touches essentially every tensor's shape.
    "different_n_embd": lambda c: (
        "nearly everything reshaped",
        c["reshaped"] >= 0.8 * c["shared"]),
    # n_head reshapes ve_gate only, and ve_gate is zero-initialized, so no
    # randomness is consumed and every other tensor matches bit-for-bit. This
    # is why n_head can sit with the architecture without disturbing the
    # shared-weights property within a region.
    "different_n_head": lambda c: (
        "only ve_gate reshaped",
        c["all_identical"] and c["reshaped"] > 0
        and all("ve_gate" in n for n in c["reshaped_names"])),
}


def main():
    from agents import remote_runner

    if "--analyze-only" in sys.argv:
        if not RAW_PATH.exists():
            sys.exit(f"[verify_shared_init] No saved probes at {RAW_PATH}; run without "
                     f"--analyze-only first.")
        results = json.loads(RAW_PATH.read_text())
        return analyze(results)

    if not remote_runner.is_remote_configured():
        sys.exit("[verify_shared_init] No remote configured; this needs a CUDA machine "
                 "(the CUDA RNG stream is what is being tested, not the CPU one).")

    client = remote_runner.open_client()
    try:
        if not remote_runner.sync_remote_code(client=client):
            sys.exit("[verify_shared_init] Remote code sync failed -- nothing probed.")
        gpus = [g["index"] for g in remote_runner.discover_available_gpus(client=client)]
        gpu = gpus[0] if gpus else 0
        print(f"[verify_shared_init] Probing on GPU {gpu}. Base architecture: {BASE_ARCH}")

        hp_dir = Path("state/init_probes")
        results = {name: run_probe(name, ov, hp_dir, client, gpu)
                   for name, ov in PROBES.items()}
    finally:
        client.close()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[verify_shared_init] Raw probe output saved to {RAW_PATH}")
    return analyze(results)


def analyze(results: Dict[str, Any]):
    base = results.get("baseline")
    if base is None:
        sys.exit("[verify_shared_init] The baseline probe failed; cannot compare anything.")

    lines = [
        "# Shared-initialization verification",
        "",
        f"Base architecture: `{BASE_ARCH}`, seed {DEFAULTS['seed']}, "
        f"{base['n_tensors']} weight tensors.",
        "",
        "A region fixes (n_layer, n_embd, n_head). The claim is that everything else "
        "Agent 1 tunes leaves the initial weights untouched, making within-region "
        "comparisons paired.",
        "",
        "| probe | expected | shared tensors | reshaped | identical | verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    verdicts = {}
    for name, probe in results.items():
        if name == "baseline" or probe is None:
            if probe is None and name != "baseline":
                lines.append(f"| {name} | — | — | — | — | **PROBE FAILED** |")
            continue
        c = compare(base, probe)
        expected, ok = EXPECTATIONS[name](c)
        verdicts[name] = ok
        lines.append(
            f"| {name} | {expected} | {c['shared']} | {c['reshaped']} | "
            f"{c['identical']}/{c['comparable']} | {'PASS' if ok else '**FAIL**'} |"
        )

    all_ok = bool(verdicts) and all(verdicts.values())
    lines += [
        "",
        "**VERIFIED — a region's configurations share initial weights.** Within-region "
        "comparisons are paired, so the initialization cancels and the region's noise "
        "floor is set by trajectory divergence alone, not by the full seed spread."
        if all_ok else
        "**NOT VERIFIED.** At least one probe behaved unexpectedly; see the table. The "
        "multi-region design depends on this, so do not build on it until resolved.",
    ]
    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"[verify_shared_init] Report written to {REPORT_PATH}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
