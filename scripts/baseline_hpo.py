"""Standard HPO baselines, so "our method is better" can be measured.

The campaign has improved its champion elite 1.496748 -> 1.419925 over ~90
runs. Nobody has ever checked whether random search would have done the same in
90 runs, so the central claim -- that region-aware, noise-calibrated tuning is
more SAMPLE EFFICIENT than standard HPO -- is currently unsupported by any
measurement in this repo. This runs the comparison.

WHAT MAKES IT FAIR
  - Same 8 tunables, same SEARCH_SPACE bounds.
  - Same normalization: sampling is uniform in NORMALIZED space, so the LR
    groups and batch_size are sampled log-uniformly exactly as
    surrogate.sobol_cold_start and propose_via_ei do. Sampling those linearly
    would hand our method a win it did not earn -- a linear draw over
    [1e-4, 3] puts almost every sample above 0.1.
  - Same architecture. Measured: all 215 campaign runs at this budget used
    (15, 828, 6), so our method has been doing pure 8-parameter HPO all along
    and there is nothing to hold still.
  - Same token budget, same eval, one GPU, sequential.
  - Same results schema, so the analysis reads all three the same way.

WHAT IS DELIBERATELY NOT CONTROLLED
  Our method's runs came from a search that also opened, paused and migrated
  regions. That overhead is part of the method and is charged to it: the
  comparison is runs-spent vs best-found, which is what a practitioner pays.

METRICS (see analyze)
  - best-so-far after N runs, the headline
  - runs-to-threshold, which is the sample-efficiency claim stated directly
  - spread of the best-so-far curve, because a method that sometimes wins is
    not the same as one that usually does
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state import surrogate
from state.results_analysis import (
    TUNABLE_COLUMNS, at_current_budget, load_results,
)
from state.results_logger import log_result

REPORT_JSON = Path("state/baseline_hpo.json")
REPORT_MD = Path("reports/baseline_hpo.md")

#: The architecture every campaign run at this budget used. Baselines hold it
#: fixed for the same reason: this is a hyperparameter comparison.
ARCHITECTURE = {"n_layer": 15, "n_embd": 828, "n_head": 6}

#: One seed for every baseline run, matching how the campaign trains. Repeats
#: are how noise is handled here, not seed averaging -- see noise_audit.
SEED = 42

OK_STATUSES = {"remote_ok", "ok"}


def _search_space() -> Dict[str, Any]:
    from agents.agent1_training_specialist import SEARCH_SPACE

    return SEARCH_SPACE


def _hyperparams_from_unit(unit: Dict[str, float]) -> Dict[str, Any]:
    """A point in [0,1]^8 -> real hyperparameters, through the SAME
    normalization the surrogate uses."""
    bounds = _search_space()
    hp: Dict[str, Any] = dict(ARCHITECTURE)
    for col in TUNABLE_COLUMNS:
        hp[col] = surrogate.denormalize(col, float(unit[col]), bounds)
    hp["batch_size"] = int(round(hp["batch_size"]))
    hp["seed"] = SEED
    # Pure measurement runs: analysis costs GPU time and cannot change val_bpb.
    hp["ablation_k"] = 0
    hp["token_xai_enabled"] = False
    return hp


def already_done(results_path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not Path(results_path).exists():
        return out
    for row in load_results(results_path):
        if row.get("status") in OK_STATUSES and isinstance(row.get("val_bpb"), (int, float)) \
                and math.isfinite(row["val_bpb"]):
            out[str(row.get("run_id"))] = float(row["val_bpb"])
    return out


# ---------------------------------------------------------------------------
# The two baselines
# ---------------------------------------------------------------------------

def random_points(n: int, seed: int) -> List[Dict[str, float]]:
    """Uniform in normalized space. No adaptivity at all -- this is the bar any
    search has to clear to have earned its complexity."""
    rng = random.Random(seed)
    return [{col: rng.random() for col in TUNABLE_COLUMNS} for _ in range(n)]


def run_baseline(method: str, n_runs: int, results_path: str, timeout: int,
                 seed: int = 0) -> None:
    """Dispatch `n_runs` for `method`, skipping any already recorded."""
    from agents import remote_runner

    if not remote_runner.is_remote_configured():
        raise SystemExit("[baseline] no remote configured -- nothing to run")

    done = already_done(results_path)
    if method == "random":
        _run_fixed(random_points(n_runs, seed), method, n_runs, results_path, timeout, done)
    elif method == "tpe":
        _run_tpe(n_runs, results_path, timeout, done, seed)
    else:
        raise SystemExit(f"[baseline] unknown method {method!r}")


#: Per-run hyperparameter YAMLs, mirroring what size_sweep does. The remote
#: name must be unique per run or two dispatches would overwrite each other's
#: settings file on the server.
HP_DIR = Path("state/baseline_hyperparams")


def _dispatch(run_id: str, hp: Dict[str, Any], results_path: str,
              timeout: int) -> Optional[float]:
    """One training run, logged in the standard schema. None if it failed.

    run_training_remote takes a PATH to a hyperparams YAML, not a dict -- it
    uploads the file over SFTP and train.py reads it on the far side.
    """
    import yaml

    from agents import remote_runner

    gpus = remote_runner.discover_available_gpus()
    if not gpus:
        print(f"[baseline] {run_id}: no GPU available")
        return None
    HP_DIR.mkdir(parents=True, exist_ok=True)
    hp_path = HP_DIR / f"{run_id}.yaml"
    with open(hp_path, "w", encoding="utf-8") as f:
        yaml.dump(hp, f)
    result = remote_runner.run_training_remote(
        hyperparams_local_path=str(hp_path), gpu_index=gpus[0]["index"],
        hp_remote_name=f"model_hyperparams_{run_id}.yaml",
        run_label=f"GPU{gpus[0]['index']}", timeout=timeout)
    metrics = dict(result or {})
    val = metrics.get("val_bpb")
    metrics.setdefault("status", "remote_error")
    log_result(run_id, hp, metrics, results_path=results_path)
    ok = isinstance(val, (int, float)) and math.isfinite(val)
    print(f"[baseline] {run_id}: val_bpb={val if ok else 'FAILED'} "
          f"status={metrics.get('status')}")
    return float(val) if ok else None


def _run_fixed(points: List[Dict[str, float]], method: str, n_runs: int,
               results_path: str, timeout: int, done: Dict[str, float]) -> None:
    for i, unit in enumerate(points):
        run_id = f"{method}_{i:04d}"
        if run_id in done:
            print(f"[baseline] {run_id}: already measured ({done[run_id]:.6f})")
            continue
        _dispatch(run_id, _hyperparams_from_unit(unit), results_path, timeout)


def _run_tpe(n_runs: int, results_path: str, timeout: int,
             done: Dict[str, float], seed: int) -> None:
    """Optuna's TPE -- the standard adaptive baseline.

    Sampled in normalized space for the same reason random search is: TPE would
    otherwise be tuning a linear learning rate axis while our method tunes a log
    one, and the comparison would be about parameterization rather than method.
    Runs already measured are replayed into the study with tell() so a resumed
    sweep keeps its history instead of starting cold.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    for i in range(n_runs):
        run_id = f"tpe_{i:04d}"
        trial = study.ask({col: optuna.distributions.FloatDistribution(0.0, 1.0)
                           for col in TUNABLE_COLUMNS})
        if run_id in done:
            print(f"[baseline] {run_id}: already measured ({done[run_id]:.6f})")
            study.tell(trial, done[run_id])
            continue
        val = _dispatch(run_id, _hyperparams_from_unit(trial.params),
                        results_path, timeout)
        # A failed run is PRUNED rather than scored: telling TPE a sentinel
        # would teach it that a region of the space is terrible when all we
        # know is that a machine fell over.
        if val is None:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, val)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def best_so_far(values: List[float]) -> List[float]:
    out, best = [], math.inf
    for v in values:
        best = min(best, v)
        out.append(best)
    return out


def _ours(history_path: str = "results.tsv") -> List[float]:
    """Our method's runs at the budget in force, in order."""
    rows = [r for r in at_current_budget(load_results(history_path))
            if r.get("status") in OK_STATUSES
            and isinstance(r.get("val_bpb"), (int, float))
            and math.isfinite(r["val_bpb"])]
    rows.sort(key=lambda r: str(r.get("run_id")))
    return [float(r["val_bpb"]) for r in rows]


def _series(results_path: str, prefix: str) -> List[float]:
    rows = [r for r in load_results(results_path)
            if str(r.get("run_id", "")).startswith(prefix)
            and r.get("status") in OK_STATUSES
            and isinstance(r.get("val_bpb"), (int, float))]
    rows.sort(key=lambda r: str(r.get("run_id")))
    return [float(r["val_bpb"]) for r in rows]


def analyze(results_path: str, history_path: str = "results.tsv") -> Dict[str, Any]:
    from prepare import TOKEN_BUDGET

    series = {"ours": _ours(history_path)}
    for method in ("random", "tpe"):
        got = _series(results_path, f"{method}_")
        if got:
            series[method] = got

    n = min((len(v) for v in series.values() if v), default=0)
    report: Dict[str, Any] = {"token_budget": TOKEN_BUDGET, "compared_over_n_runs": n,
                              "architecture": ARCHITECTURE, "methods": {}}
    for name, values in series.items():
        if not values:
            continue
        curve = best_so_far(values)
        report["methods"][name] = {
            "n_runs": len(values),
            "best": min(values),
            "best_at_n": curve[n - 1] if n else None,
            "curve": curve,
        }

    # runs-to-threshold: the sample-efficiency claim, stated directly. The
    # threshold is the WORST of the methods' final bests, so every method
    # reaches it and the comparison is "how many runs did that cost".
    finals = [m["best"] for m in report["methods"].values()]
    if finals:
        threshold = max(finals)
        report["threshold"] = threshold
        for name, m in report["methods"].items():
            hit = next((i + 1 for i, b in enumerate(m["curve"]) if b <= threshold), None)
            m["runs_to_threshold"] = hit
    return report


def render(report: Dict[str, Any]) -> str:
    lines = ["# HPO baselines vs the region search", "",
             f"- token budget: {report['token_budget']:,}",
             f"- architecture held at {report['architecture']}",
             f"- compared over the first {report['compared_over_n_runs']} runs of each", ""]
    if report.get("threshold") is not None:
        lines += [f"- threshold = {report['threshold']:.6f} "
                  "(the worst final best, so every method reaches it)", ""]
    lines += ["| method | runs | best | best at n | runs to threshold |",
              "|---|---|---|---|---|"]
    for name, m in sorted(report["methods"].items(),
                          key=lambda kv: kv[1]["best"]):
        at_n = m.get("best_at_n")
        lines.append(f"| {name} | {m['n_runs']} | {m['best']:.6f} | "
                     f"{'-' if at_n is None else f'{at_n:.6f}'} | "
                     f"{m.get('runs_to_threshold') or '-'} |")
    lines += ["", "A method is more sample efficient if it reaches the threshold in "
                  "FEWER runs, not if it eventually finds a lower number.", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("random", "tpe"), default=None)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--results-path", default="state/baseline_hpo.tsv")
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if not args.analyze_only:
        if args.method is None:
            raise SystemExit("[baseline] --method is required unless --analyze-only")
        run_baseline(args.method, args.runs, args.results_path, args.timeout, args.seed)

    report = analyze(args.results_path)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render(report), encoding="utf-8")
    print(render(report))
    print(f"[baseline] wrote {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
