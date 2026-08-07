"""Tier 0 step 3 -- the decisive experiment: does the initial-weight seed
change WHICH configuration wins?

Every run in this campaign so far shared one hardcoded initialization
(train.py's `torch.manual_seed(42)`), so val_bpb has never measured
"how good is this config" -- it has measured "how good is this config when
started from seed 42". Those differ by a bias term b(S) = measured(S) - true(S).

A CONSTANT bias is harmless: the search only ever uses val_bpb to rank configs
against each other, so an offset shared by every config changes nothing. The
experiment therefore does not ask "how big is b(S)" but the only question that
has consequences:

    DOES b(S) VARY WITH S?

Design: N_CONFIGS configurations spanning the observed quality range x N seeds,
fully crossed. Because every config is run under the SAME set of seeds, this is
a paired design, and the pairing is what makes it cheap -- see the two
quantities reported:

  sigma_seed        spread of ONE config's val_bpb across seeds.
                    "How much does a single measurement move?"

  sigma_paired_diff spread of the GAP between two configs across seeds.
                    "How much does the COMPARISON move?"

Only the second one matters for a search:

  * sigma_paired_diff << sigma_seed  -> a seed shifts every config by nearly
    the same amount. b(S) is near-constant, rankings are stable, and freezing
    the seed is not merely acceptable but actively good (it removes a shared
    nuisance from every comparison). One seed per config is enough.

  * sigma_paired_diff ~ sqrt(2) * sigma_seed -> seed effects are independent
    per config. b(S) varies with S, the search has partly been ranking seeds,
    and configurations separated by less than the resolvable gap reported
    below cannot be told apart with one run each.

Also reported: the smallest val_bpb gap that is resolvable at k seeds per
config. That number is what agents_config.yaml's sigma-scaled thresholds
should be re-derived against (plan step 7) -- state/noise_floor.json's
sigma = 0.000797 cannot serve, because it was measured by repeating one config
with the seed, the data and the token budget ALL held fixed, so it captures
bf16/kernel nondeterminism and nothing statistical at all.

Usage:
    uv run python scripts/seed_variance.py --seeds 5
    uv run python scripts/seed_variance.py --history legacy_results.tsv/results.tsv.legacy-20260806191826
    uv run python scripts/seed_variance.py --analyze-only
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from state.results_analysis import HYPERPARAM_COLUMNS, load_results
from state.results_logger import log_result
from state.surrogate import INT_PARAMS

RUN_ID_PREFIX = "seedvar"
DEFAULT_RESULTS_PATH = "state/seed_variance.tsv"
REPORT_JSON_PATH = Path("state/seed_variance.json")

# Seed 42 goes first on purpose: it is the campaign's historical seed, so its
# row is directly comparable to that config's original results.tsv entry and
# doubles as a check that the plumbing reproduces what it should.
SEED_POOL = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)

#: Statuses whose val_bpb is a real, complete measurement.
OK_STATUSES = {"remote_ok", "ok"}


# ---------------------------------------------------------------------------
# Choosing the configurations to test
# ---------------------------------------------------------------------------

def _usable(row: Dict[str, Any]) -> bool:
    """A row is usable as a test configuration only if it completed normally
    AND consumed its whole token budget. A truncated run (budget_shortfall_pct
    > 0) measured less training than a full one, so its val_bpb is not on the
    same scale as the numbers this experiment is about to produce."""
    if row.get("status") not in OK_STATUSES:
        return False
    val_bpb = row.get("val_bpb")
    if not isinstance(val_bpb, (int, float)) or val_bpb != val_bpb or val_bpb in (float("inf"), float("-inf")):
        return False
    shortfall = row.get("budget_shortfall_pct")
    if isinstance(shortfall, (int, float)) and shortfall > 0.0:
        return False
    return all(col in row for col in HYPERPARAM_COLUMNS)


def _row_to_hyperparams(row: Dict[str, Any]) -> Dict[str, Any]:
    hp: Dict[str, Any] = {}
    for col in HYPERPARAM_COLUMNS:
        value = float(row[col])
        hp[col] = int(round(value)) if col in INT_PARAMS else value
    # Both are pure post-training overhead here: neither changes val_bpb, and
    # this experiment is dispatching 3x more runs than usual. ablation_k=0
    # skips train.py's head-ablation passes; token_xai_enabled=False skips the
    # behavioral fingerprint, which roughly doubles wall-clock on its own.
    hp["ablation_k"] = 0
    hp["token_xai_enabled"] = False
    return hp


def select_configs(rows: List[Dict[str, Any]], n_configs: int) -> List[Dict[str, Any]]:
    """The two best runs, plus configurations spread across the rest of the range.

    The two best are mandatory, and that is the crux of the design. The search's
    real decisions all happen at the frontier, between configurations separated
    by very little -- in this campaign's history the gap between the best and
    second-best run is 0.0058 val_bpb. Whether a seed can reorder THAT pair is
    the question with consequences. An experiment made only of far-apart
    configurations would return "stable" under any noise level and would have
    answered a question nobody asked.

    The wider picks are the control in the other direction: they are far enough
    apart that they MUST resolve, so a flip there would indict the experiment
    rather than the search, and a clean separation there proves the method has
    the power to detect a real difference when one exists.

    The extreme tail is skipped (picks stop at the 75th percentile): the worst
    runs are pathological configurations whose gap to everything else resolves
    trivially and carries no information.
    """
    usable = sorted((r for r in rows if _usable(r)), key=lambda r: r["val_bpb"])
    if len(usable) < n_configs:
        raise SystemExit(
            f"[seed_variance] Need at least {n_configs} complete runs to pick test "
            f"configurations from, found {len(usable)}.\n"
            f"                Pass --history with a file that has them, e.g. an archive "
            f"under legacy_results.tsv/, or --config-json to specify them by hand."
        )
    if n_configs == 1:
        return [usable[0]]

    picks = [0, 1]  # the frontier pair -- the comparison the search actually makes
    remaining = n_configs - 2
    if remaining > 0:
        last = int(0.75 * (len(usable) - 1))
        step = (last - 1) / remaining
        picks.extend(round(1 + (i + 1) * step) for i in range(remaining))

    # Dedupe (a short history can collapse picks onto the same index), then top
    # up from whatever is left so the caller always gets the count it asked for
    # rather than silently running a smaller experiment.
    ordered = list(dict.fromkeys(picks))
    for i in range(len(usable)):
        if len(ordered) >= n_configs:
            break
        if i not in ordered:
            ordered.append(i)
    return [usable[i] for i in ordered[:n_configs]]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _write_hp(hp: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(hp, f)


def _run_block(
    agent1: Agent1TrainingSpecialist,
    jobs: List[Tuple[int, int, Dict[str, Any], str]],
    hp_dir: Path,
    results_path: str,
) -> List[Dict[str, Any]]:
    """Train every configuration for ONE seed, concurrently when the remote
    server has the GPUs for it.

    Blocking by seed rather than by config is what keeps a mid-experiment
    interruption interpretable: stopping after block 3 leaves 3 complete seeds
    for every configuration (a smaller but valid experiment) instead of 3
    configurations finished and the rest untouched (no experiment at all).
    """
    from agents import remote_runner

    records: List[Dict[str, Any]] = []

    if not remote_runner.is_remote_configured():
        for config_idx, seed, hp, run_id in jobs:
            print(f"\n[seed_variance] --- {run_id} (config {config_idx}, seed {seed}) ---")
            metrics = agent1.train_model(hp, dry_run=False, iteration=config_idx)
            log_result(run_id, hp, metrics, results_path=results_path)
            records.append({"config_idx": config_idx, "seed": seed, "run_id": run_id, **metrics})
        return records

    try:
        client = remote_runner.open_client()
    except Exception as e:
        raise SystemExit(f"[seed_variance] Could not reach the remote server: {e}")

    try:
        remote_runner.kill_stale_training_processes(client=client)
        gpus = [g["index"] for g in remote_runner.discover_available_gpus(client=client)]
        if not remote_runner.sync_remote_code(client=client):
            raise SystemExit("[seed_variance] Remote code sync failed -- nothing dispatched.")

        if len(gpus) < 2:
            gpu_for = {job[0]: (gpus[0] if gpus else None) for job in jobs}
            parallel = False
        else:
            # Rotate the config -> GPU mapping by seed so no configuration is
            # ever pinned to one device for the whole experiment. GPU identity
            # contributes its own (small) offset; leaving it confounded with
            # configuration would put that offset straight into the very
            # comparison this script exists to measure.
            seed_index = jobs[0][1]
            offset = SEED_POOL.index(seed_index) if seed_index in SEED_POOL else 0
            gpu_for = {job[0]: gpus[(i + offset) % len(gpus)] for i, job in enumerate(jobs)}
            parallel = True

        if not parallel:
            for config_idx, seed, hp, run_id in jobs:
                hp_path = hp_dir / f"{run_id}.yaml"
                _write_hp(hp, hp_path)
                metrics = remote_runner.run_training_remote(
                    hyperparams_local_path=str(hp_path),
                    gpu_index=gpu_for[config_idx],
                    hp_remote_name=f"model_hyperparams_{run_id}.yaml",
                    run_label=run_id,
                    timeout=agent1.training_budget + 120,
                    skip_sync=True,
                    client=client,
                )
                log_result(run_id, hp, metrics, results_path=results_path)
                records.append({"config_idx": config_idx, "seed": seed, "run_id": run_id, **metrics})
            return records

        from agents.live_progress import MultiGpuProgressDisplay

        labels = [f"GPU{gpu_for[job[0]]}" for job in jobs]
        with MultiGpuProgressDisplay(labels) as display:
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                futures = {}
                for config_idx, seed, hp, run_id in jobs:
                    hp_path = hp_dir / f"{run_id}.yaml"
                    _write_hp(hp, hp_path)
                    future = executor.submit(
                        remote_runner.run_training_remote,
                        hyperparams_local_path=str(hp_path),
                        gpu_index=gpu_for[config_idx],
                        hp_remote_name=f"model_hyperparams_{run_id}.yaml",
                        run_label=f"GPU{gpu_for[config_idx]}",
                        timeout=agent1.training_budget + 120,
                        skip_sync=True,
                        display=display,
                        client=client,
                    )
                    futures[future] = (config_idx, seed, hp, run_id)

                for future in as_completed(futures):
                    config_idx, seed, hp, run_id = futures[future]
                    try:
                        metrics = future.result()
                    except Exception as e:
                        display.print_line(f"[seed_variance] {run_id} failed: {e}")
                        metrics = {"val_bpb": float("inf"), "status": "remote_error", "error": str(e)}
                    display.print_line(
                        f"[seed_variance] {run_id}: val_bpb={metrics.get('val_bpb')} "
                        f"status={metrics.get('status')}"
                    )
                    log_result(run_id, hp, metrics, results_path=results_path)
                    records.append({"config_idx": config_idx, "seed": seed, "run_id": run_id, **metrics})
        return records
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(measurements: Dict[int, Dict[int, float]], seeds: List[int]) -> Dict[str, Any]:
    """measurements: {config_idx: {seed: val_bpb}} -- complete cells only.

    Returns the per-config spread, the paired-difference spread for every pair,
    and a verdict. Nothing here fabricates a number from an incomplete cell: a
    pair is only compared on the seeds where BOTH configs produced a result.
    """
    per_config: Dict[int, Dict[str, Any]] = {}
    within_variances: List[float] = []
    for config_idx, by_seed in sorted(measurements.items()):
        values = [by_seed[s] for s in seeds if s in by_seed]
        entry: Dict[str, Any] = {
            "n": len(values),
            "values_by_seed": {str(s): by_seed[s] for s in seeds if s in by_seed},
            "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
        }
        if entry["std"] is not None:
            within_variances.append(entry["std"] ** 2)
        per_config[config_idx] = entry

    sigma_seed = (sum(within_variances) / len(within_variances)) ** 0.5 if within_variances else None

    pairs: List[Dict[str, Any]] = []
    for a, b in combinations(sorted(measurements), 2):
        shared = [s for s in seeds if s in measurements[a] and s in measurements[b]]
        diffs = [measurements[a][s] - measurements[b][s] for s in shared]
        if len(diffs) < 2:
            continue
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs)
        a_wins = sum(1 for d in diffs if d < 0)
        pairs.append({
            "config_a": a,
            "config_b": b,
            "n_seeds": len(diffs),
            "mean_gap": mean_diff,
            "std_of_gap": std_diff,
            # How often the winner is the same one. A ranking that flips is the
            # failure mode; a consistent winner with a tiny gap is not.
            "a_wins": a_wins,
            "b_wins": len(diffs) - a_wins,
            "ranking_consistent": a_wins in (0, len(diffs)),
            # Paired t-style statistic. Above ~2 the gap is real at these
            # seeds; below it, the two configs are indistinguishable no matter
            # what their means say.
            "separation": abs(mean_diff) / (std_diff / len(diffs) ** 0.5) if std_diff > 0 else float("inf"),
        })

    # Does the seed move every config together, or each one independently?
    # sqrt(2)*sigma_seed is what std_of_gap equals when the effects are fully
    # independent; near 0 means they move as one and the comparison is immune.
    independent_ref = (2 ** 0.5) * sigma_seed if sigma_seed else None
    gap_stds = [p["std_of_gap"] for p in pairs]
    mean_gap_std = statistics.mean(gap_stds) if gap_stds else None
    shared_fraction = (
        max(0.0, 1.0 - mean_gap_std / independent_ref)
        if mean_gap_std is not None and independent_ref else None
    )

    n_seeds = len(seeds)
    resolvable = {
        str(k): 2.0 * mean_gap_std / k ** 0.5
        for k in (1, 2, 3, 5)
    } if mean_gap_std else {}

    all_consistent = bool(pairs) and all(p["ranking_consistent"] for p in pairs)
    verdict = (
        "RANKINGS STABLE -- every pair kept the same winner across all seeds. "
        "Freezing the seed for screening is sound; keep one seed per config."
        if all_consistent else
        "RANKINGS FLIP -- at least one pair changed winner depending on the seed. "
        "Single-seed val_bpb cannot separate configs at that gap size; multi-seed "
        "evaluation is required for frontier decisions (plan tier 2)."
    )

    return {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "per_config": per_config,
        "sigma_seed": sigma_seed,
        "mean_std_of_pairwise_gap": mean_gap_std,
        "independent_reference_sqrt2_sigma": independent_ref,
        "shared_fraction_of_seed_effect": shared_fraction,
        "resolvable_gap_at_k_seeds": resolvable,
        "pairs": pairs,
        "all_pairs_consistent": all_consistent,
        "verdict": verdict,
    }


def _load_noise_floor_sigma() -> Optional[float]:
    path = Path("state/noise_floor.json")
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["std"])
    except (OSError, ValueError, KeyError):
        return None


def render(report: Dict[str, Any], configs: List[Dict[str, Any]]) -> str:
    lines = ["", "=" * 72, "[seed_variance] RESULT", "=" * 72, ""]

    lines.append("Per-configuration val_bpb across seeds:")
    for idx, entry in sorted(report["per_config"].items()):
        hp = configs[idx]
        std = f"{entry['std']:.6f}" if entry["std"] is not None else "n/a"
        mean = f"{entry['mean']:.6f}" if entry["mean"] is not None else "n/a"
        lines.append(
            f"  config {idx}: mean={mean} std={std} n={entry['n']}  "
            f"(n_layer={hp.get('n_layer')}, n_embd={hp.get('n_embd')}, n_head={hp.get('n_head')})"
        )
        lines.append(f"      by seed: " + ", ".join(
            f"{s}={v:.6f}" for s, v in entry["values_by_seed"].items()))

    sigma_seed = report["sigma_seed"]
    gap_std = report["mean_std_of_pairwise_gap"]
    lines += ["", "Two spreads (only the second one constrains a search):"]
    if sigma_seed is not None:
        lines.append(f"  sigma_seed            = {sigma_seed:.6f}   how much ONE config moves with the seed")
    if gap_std is not None:
        lines.append(f"  sigma_paired_diff     = {gap_std:.6f}   how much a COMPARISON moves with the seed")
        lines.append(f"  independent reference = {report['independent_reference_sqrt2_sigma']:.6f}   "
                     f"(sqrt(2)*sigma_seed; what sigma_paired_diff equals if seeds hit each config independently)")
        if report["shared_fraction_of_seed_effect"] is not None:
            lines.append(f"  -> {report['shared_fraction_of_seed_effect']:.0%} of the seed effect is SHARED across "
                         f"configs (cancels in a comparison)")

    floor = _load_noise_floor_sigma()
    if floor and sigma_seed:
        lines += ["", "Against the currently-configured yardsticks:"]
        lines.append(f"  state/noise_floor.json sigma = {floor:.6f}  -> sigma_seed is {sigma_seed / floor:.1f}x larger")
        lines.append(f"  agents_config sigma_region   = 0.002800  -> sigma_seed is {sigma_seed / 0.0028:.1f}x that")
        lines.append("  Every sigma-scaled threshold in agents_config.yaml was calibrated on the")
        lines.append("  noise_floor sigma, which held the seed FIXED and so measured no seed effect at all.")

    if report["resolvable_gap_at_k_seeds"]:
        lines += ["", "Smallest val_bpb gap resolvable (~2 standard errors) at k seeds per config:"]
        for k, gap in report["resolvable_gap_at_k_seeds"].items():
            lines.append(f"  k={k}: {gap:.6f}")

    lines += ["", "Pairwise comparisons:"]
    for p in report["pairs"]:
        flag = "consistent" if p["ranking_consistent"] else "*** FLIPPED ***"
        lines.append(
            f"  config {p['config_a']} vs {p['config_b']}: mean_gap={p['mean_gap']:+.6f} "
            f"std_of_gap={p['std_of_gap']:.6f} separation={p['separation']:.1f} "
            f"wins {p['a_wins']}-{p['b_wins']} [{flag}]"
        )

    lines += ["", "VERDICT: " + report["verdict"], "=" * 72, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5,
                        help=f"How many seeds per configuration (from {SEED_POOL}). Default 5.")
    parser.add_argument("--configs", type=int, default=3,
                        help="How many configurations to test, spread across the observed quality range.")
    parser.add_argument("--history", default="results.tsv",
                        help="Where to pick the test configurations from. An archive under "
                             "legacy_results.tsv/ works too -- load_results matches rows by "
                             "field count, not by header.")
    parser.add_argument("--config-json", default=None,
                        help="Path to a JSON list of hyperparameter dicts, bypassing history selection.")
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH,
                        help=f"Where the runs are logged. Default {DEFAULT_RESULTS_PATH} -- deliberately "
                             f"NOT results.tsv: this dispatches the same few configurations many times, "
                             f"and those duplicates would over-weight a handful of points in the "
                             f"surrogate's training set. Pass results.tsv to override.")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip training; re-analyze runs already in --results-path.")
    parser.add_argument("--agents-config", default="agents_config.yaml")
    args = parser.parse_args()

    agent1 = Agent1TrainingSpecialist(config_path=args.agents_config)
    seeds = list(SEED_POOL[:args.seeds])

    # --- configurations ---
    if args.config_json:
        configs = json.loads(Path(args.config_json).read_text())
    elif args.analyze_only:
        configs = []
    else:
        rows = load_results(args.history)
        picked = select_configs(rows, args.configs)
        configs = [_row_to_hyperparams(r) for r in picked]
        print(f"[seed_variance] Test configurations, picked from {args.history} "
              f"({len(rows)} historical row(s)):")
        for i, (row, hp) in enumerate(zip(picked, configs)):
            print(f"  config {i}: historical val_bpb={row['val_bpb']:.6f} (run {row.get('run_id')}) "
                  f"n_layer={hp['n_layer']} n_embd={hp['n_embd']} n_head={hp['n_head']} "
                  f"matrix_lr={hp['matrix_lr']:.4g}")

    # --- dispatch ---
    if not args.analyze_only:
        total = len(configs) * len(seeds)
        print(f"\n[seed_variance] Dispatching {total} run(s): "
              f"{len(configs)} config(s) x {len(seeds)} seed(s) {seeds}")
        hp_dir = Path("state/seed_variance_hyperparams")
        started = time.time()
        for seed in seeds:
            jobs = []
            for config_idx, base in enumerate(configs):
                hp = dict(base)
                hp["seed"] = seed
                jobs.append((config_idx, seed, hp, f"{RUN_ID_PREFIX}_c{config_idx}_s{seed}"))
            print(f"\n[seed_variance] === seed {seed} ({len(jobs)} run(s)) ===")
            _run_block(agent1, jobs, hp_dir, args.results_path)
        print(f"\n[seed_variance] All runs finished in {(time.time() - started) / 60:.1f} min")

    # --- analysis (always from the logged rows, so --analyze-only and a
    #     completed dispatch take exactly the same path) ---
    rows = load_results(args.results_path)
    measurements: Dict[int, Dict[int, float]] = {}
    seen_configs: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id.startswith(f"{RUN_ID_PREFIX}_c"):
            continue
        if row.get("status") not in OK_STATUSES:
            continue
        val_bpb = row.get("val_bpb")
        if not isinstance(val_bpb, (int, float)):
            continue
        try:
            config_idx = int(run_id.split("_c")[1].split("_s")[0])
        except (IndexError, ValueError):
            continue
        seed = int(row["seed"]) if isinstance(row.get("seed"), (int, float)) else None
        if seed is None:
            continue
        measurements.setdefault(config_idx, {})[seed] = float(val_bpb)
        seen_configs.setdefault(config_idx, {c: row.get(c) for c in HYPERPARAM_COLUMNS})

    if not measurements:
        raise SystemExit(f"[seed_variance] No usable {RUN_ID_PREFIX}_* rows in {args.results_path}.")

    observed_seeds = sorted({s for by_seed in measurements.values() for s in by_seed})
    report = analyze(measurements, observed_seeds)
    if not configs:
        configs = [seen_configs.get(i, {}) for i in range(max(seen_configs) + 1)]
    report["configs"] = {str(i): hp for i, hp in enumerate(configs)}
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["results_path"] = args.results_path

    print(render(report, configs))

    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[seed_variance] Written to {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
