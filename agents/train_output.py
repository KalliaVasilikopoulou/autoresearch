"""The single definition of how train.py's final summary block is parsed.

train.py ends by printing a block of `label: value` lines (val_bpb, num_steps,
seed, ...) plus a few `label: {json blob}` lines carrying per-run interpretability
evidence. Two places consume that output:

  - agents/remote_runner.py::_parse_output           -- every real campaign run
  - Agent1TrainingSpecialist._parse_training_output  -- the local subprocess fallback

They used to hold near-identical private copies of the field map, the JSON key
set, and the line-scanning loop, kept in sync by hand. That is a silent-failure
shape rather than merely untidy: a field added to one copy is simply absent from
whichever path the other one serves, and nothing raises. It came within one edit
of happening -- the `seed:` field was first added only to Agent 1's map, which
would have produced a `seed` column in results.tsv that was correct for local
runs and blank for every real one, i.e. for the entire campaign.

This module is a leaf: it imports nothing from the package, so both callers can
import it at module scope without an import cycle (agent1 imports remote_runner
lazily inside train_model precisely because a module-level import there would be
one).

What is deliberately NOT shared is each caller's INITIAL metrics dict.
remote_runner seeds `status: "remote_ok"` and `training_time: None` because a
remote run has a transport status and may be reported on partial output; Agent 1
seeds only `val_bpb: inf` and lets its caller set the status. Those defaults are
statements about the two execution paths, not about the text format, so they
stay where they belong -- `parse` returns only what it actually found in the
output, and the callers merge it onto their own base.
"""

import json
from typing import Any, Callable, Dict, Tuple

#: `<label>:` as printed by train.py -> (key in the returned metrics, cast).
#: Matched against the lowercased first whitespace-separated token, so the
#: trailing colon is part of the key and `num_params_M:` is written lowercase.
FIELDS: Dict[str, Tuple[str, Callable[[str], Any]]] = {
    "val_bpb:": ("val_bpb", float),
    "training_seconds:": ("training_time", float),
    "total_seconds:": ("total_seconds", float),
    "peak_vram_mb:": ("peak_vram_mb", float),
    "mfu_percent:": ("mfu_percent", float),
    "total_tokens_m:": ("total_tokens_M", float),
    "num_steps:": ("num_steps", int),
    "num_params_m:": ("num_params_M", float),
    "depth:": ("depth", int),
    # The initial-weight seed train.py actually ran with (train.py's SEED),
    # as opposed to the one requested in model_hyperparams.yaml. The two
    # diverge exactly when that file failed to load.
    "seed:": ("seed", int),
    "holdout_val_bpb:": ("holdout_val_bpb", float),
    # >0 means the run hit the wall-clock safety cap before consuming its token
    # budget -- an incomplete measurement, not a worse configuration.
    "budget_shortfall_pct:": ("budget_shortfall_pct", float),
}

#: Lines like `head_ablation_impacts: {...}` carry real per-run evidence as a
#: JSON blob rather than a single scalar. Keyed generically so any future
#: `<name>: {json}` line train.py prints needs only an entry here.
JSON_KEYS = frozenset({
    "interpretable_scalars",
    "head_ablation_impacts",
    "hyperparam_clamps",
    "token_fingerprint",
})


def parse(stdout: str) -> Dict[str, Any]:
    """Extract every recognised metric from train.py's stdout.

    Returns ONLY what was actually found -- no defaults, no placeholders -- so a
    caller can distinguish "train.py did not report this" from "it reported
    something". Callers supply their own base dict and merge this onto it.

    Every malformed value is skipped rather than raised on: a summary block can
    be truncated mid-line when an SSH channel drops, and one unparseable field
    must not discard the val_bpb sitting three lines above it.
    """
    metrics: Dict[str, Any] = {}
    for line in stdout.splitlines():
        if ":" in line:
            prefix, _, rest = line.partition(":")
            key = prefix.strip()
            if key in JSON_KEYS:
                try:
                    metrics[key] = json.loads(rest.strip())
                except (json.JSONDecodeError, ValueError):
                    pass
                # Consumed either way: a JSON line is never also a scalar field,
                # so falling through would only risk a spurious match.
                continue

        parts = line.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key in FIELDS and len(parts) >= 2:
            dest, cast = FIELDS[key]
            try:
                metrics[dest] = cast(parts[1])
            except (ValueError, IndexError):
                pass
    return metrics
