"""Check whether the search loop is fitting noise in its pinned validation
shard (see prepare.py's HOLDOUT_SHARD / evaluate_bpb_holdout).

Re-trains the top-K configs (by historical val_bpb) from results.tsv with
holdout_eval enabled, so each retrain reports both val_bpb (the shard the
search always compares against) and holdout_val_bpb (a shard the search
never sees) from the *same* trained model. If the config that ranks best on
val_bpb isn't also best on holdout_val_bpb, the search has been measuring
its own selection bias, not real improvement.

EACH CANDIDATE IS RUN ON SEVERAL SEEDS (--seeds, default 3). A single run per
candidate would rank them by one draw from a distribution whose spread is
~0.0020 at the frontier, while the gaps being ranked are ~0.0012
(scripts/seed_variance.py) -- i.e. the ranking would be mostly initialization
luck. That is the very failure this script exists to detect, so doing it here
would just move the bias rather than measure it. Candidates are ranked by MEAN
holdout_val_bpb across seeds, and the per-candidate spread is reported so a
tie can be recognised as a tie.

Cost: K * seeds full training runs. Run this once at the end of a search
campaign, not per-iteration -- the search loop itself never uses
holdout_val_bpb to decide anything (see prepare.py's _document_batches:
HOLDOUT_SHARD is excluded from training and never touched by evaluate_bpb).

Usage:
    uv run python scripts/holdout_eval.py --top-k 5 --seeds 3
    uv run python scripts/holdout_eval.py --drift-only     # no GPU, reads history
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from state.results_analysis import (
    HYPERPARAM_COLUMNS,
    holdout_drift,
    load_results,
    spearman,
)
from state.results_logger import log_result

RUN_ID_PREFIX = "holdout_check"
REPORT_PATH = Path("reports/holdout_eval_report.md")

# Seed 42 first: the campaign's historical seed, so its row is directly
# comparable to what history already recorded for the same config.
SEED_POOL = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def render_drift(drift) -> str:
    """The continuous half of step 8: has the validation shard drifted away
    from the holdout shard as the campaign made more decisions against it?"""
    if drift is None:
        return ("## Validation-vs-holdout drift\n\n"
                "Not enough history: fewer than two runs carry both `val_bpb` and "
                "`holdout_val_bpb`. The orchestrator scores the holdout shard only when a "
                "run sets a new best (`holdout_on_new_best`), so this fills in slowly -- "
                "and note that makes it a sample of WINNERS, which is why the top-K "
                "re-check above exists as well.\n")

    lines = [
        "## Validation-vs-holdout drift",
        "",
        f"{drift['n']} run(s) carry both numbers. The gap is "
        f"`holdout_val_bpb - val_bpb`; a constant gap is harmless (the shards are "
        f"different text, so one is simply harder), a GROWING gap is the search "
        f"fitting the validation shard's quirks.",
        "",
        "| run_id | val_bpb | holdout_val_bpb | gap |",
        "|---|---:|---:|---:|",
    ]
    for r in drift["runs"]:
        lines.append(f"| {r['run_id']} | {r['val_bpb']:.6f} | "
                     f"{r['holdout_val_bpb']:.6f} | {r['gap']:+.6f} |")
    lines += [
        "",
        f"- mean gap: **{drift['mean_gap']:+.6f}** (spread {drift['std_gap']:.6f})",
    ]
    if drift["growth"] is not None:
        lines.append(f"- earlier half {drift['early_mean_gap']:+.6f} -> "
                     f"later half {drift['late_mean_gap']:+.6f} "
                     f"(change {drift['growth']:+.6f})")
    lines.append("")
    lines.append(
        "**DRIFT DETECTED**: the gap grew by more than its own spread, which is what "
        "progressive overfitting of the validation shard looks like."
        if drift["growing"] else
        "No drift detected: the gap is a roughly constant offset, which is expected and "
        "harmless -- it reflects the two shards being different text, not selection bias."
    )
    return "\n".join(lines) + "\n"


def check_holdout_shard_available():
    """Is the pinned holdout shard actually present where training runs?

    Returns True/False, or None when it can't be determined (paramiko
    missing, remote not configured -- i.e. training runs locally).

    This exists because of a real, expensive failure: train.py wraps its
    holdout evaluation in a try/except that prints and continues, so a
    missing shard_06541.parquet doesn't fail a run -- it just silently omits
    holdout_val_bpb. The first real attempt at this check burned 7 full
    training runs before anyone noticed every one of them had come back
    without the number, and the cause turned out to be that the shard had
    never been downloaded at all (prepare.py gained the HOLDOUT_SHARD line
    after this machine's data was prepared). A two-second test beats
    discovering that 50 minutes in.
    """
    try:
        import paramiko  # noqa: F401
        from agents.remote_runner import _connect_with_retry, _load_cfg, is_remote_configured
        from prepare import HOLDOUT_SHARD
    except ImportError:
        return None
    if not is_remote_configured():
        return None

    cfg = _load_cfg()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        _connect_with_retry(client, cfg, timeout=30)
        path = f"~/.cache/autoresearch/data/shard_{HOLDOUT_SHARD:05d}.parquet"
        _, stdout, _ = client.exec_command(f"test -f {path} && echo PRESENT || echo ABSENT", timeout=30)
        return stdout.read().decode("utf-8", errors="replace").strip().endswith("PRESENT")
    except Exception as e:
        print(f"[holdout_eval] Could not verify the holdout shard remotely ({e}) -- continuing anyway")
        return None
    finally:
        client.close()


def _dedupe_top_k(rows, top_k):
    """Top-K distinct configs by ascending val_bpb (dedupe by hyperparams so
    repeated noise-floor runs of one config don't crowd out K distinct ones).
    """
    finite = [r for r in rows if "val_bpb" in r]
    finite.sort(key=lambda r: r["val_bpb"])
    seen = set()
    candidates = []
    for row in finite:
        key = tuple(row.get(col) for col in HYPERPARAM_COLUMNS)
        if key in seen or any(v is None for v in key):
            continue
        seen.add(key)
        candidates.append(row)
        if len(candidates) >= top_k:
            break
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Runs per candidate, each with a different initial-weight "
                             "seed. Default 3. One would rank candidates by a single "
                             "draw whose spread (~0.0020) exceeds the gaps being ranked "
                             "(~0.0012), i.e. by luck -- see scripts/seed_variance.py.")
    parser.add_argument("--config", default="agents_config.yaml")
    parser.add_argument("--results-path", default="results.tsv")
    parser.add_argument("--drift-only", action="store_true",
                        help="Skip all training; just report validation-vs-holdout drift "
                             "from runs already in --results-path. Costs no GPU time.")
    args = parser.parse_args()

    rows = load_results(args.results_path)

    if args.drift_only:
        report = "# Holdout Evaluation Report (drift only)\n\n" + render_drift(holdout_drift(rows))
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        print(f"[holdout_eval] Report written to {REPORT_PATH}")
        return

    candidates = _dedupe_top_k(rows, args.top_k)
    if len(candidates) < 2:
        print(f"[holdout_eval] Only {len(candidates)} distinct real config(s) in "
              f"{args.results_path} — need at least 2 to compare. Run a couple of "
              f"real, distinct search iterations first.")
        sys.exit(1)

    # Pre-flight before committing K training runs (~5 min each).
    present = check_holdout_shard_available()
    if present is False:
        from prepare import HOLDOUT_SHARD
        print(f"[holdout_eval] ABORTING: shard_{HOLDOUT_SHARD:05d}.parquet is not present on the "
              f"training machine, so every run would come back without a holdout_val_bpb "
              f"(train.py logs the failure and continues rather than erroring).\n"
              f"[holdout_eval] Fix it with:  python -c \"from prepare import "
              f"download_single_shard, HOLDOUT_SHARD; download_single_shard(HOLDOUT_SHARD)\"")
        sys.exit(1)
    if present:
        print("[holdout_eval] Pre-flight OK: holdout shard is present on the training machine.")

    seeds = list(SEED_POOL[: max(1, args.seeds)])
    print(f"[holdout_eval] Re-checking top-{len(candidates)} configs on the holdout shard, "
          f"{len(seeds)} seed(s) each ({len(candidates) * len(seeds)} runs)...")
    agent1 = Agent1TrainingSpecialist(config_path=args.config)

    results = []
    for i, cand in enumerate(candidates):
        base = {col: cand[col] for col in HYPERPARAM_COLUMNS if col in cand}
        base["holdout_eval"] = True
        original_run_id = cand.get("run_id", f"unknown_{i}")
        val_scores, holdout_scores = [], []

        for seed in seeds:
            hp = dict(base)
            hp["seed"] = seed
            run_id = f"{RUN_ID_PREFIX}_{i:04d}_s{seed}"
            print(f"\n[holdout_eval] --- {run_id} (was {original_run_id}, "
                  f"historical val_bpb={cand['val_bpb']:.6f}) ---")

            agent1.current_hyperparams = dict(hp)
            metrics = agent1.train_model(hp, dry_run=False, iteration=i)
            log_result(run_id, hp, metrics, results_path=args.results_path)

            val = metrics.get("val_bpb")
            hold = metrics.get("holdout_val_bpb")
            if isinstance(val, (int, float)):
                val_scores.append(float(val))
            if isinstance(hold, (int, float)):
                holdout_scores.append(float(hold))

        if not holdout_scores:
            print(f"[holdout_eval] {original_run_id}: no seed produced a holdout score -- omitted")
            continue

        results.append({
            "run_id": f"{RUN_ID_PREFIX}_{i:04d}",
            "original_run_id": original_run_id,
            "n_seeds": len(holdout_scores),
            # Ranked on the MEAN across seeds, never the best of them: best-of-N
            # over noisy draws is exactly the statistic that manufactured the
            # inflated frontier this script is checking.
            "val_bpb": statistics.mean(val_scores) if val_scores else float("inf"),
            "holdout_val_bpb": statistics.mean(holdout_scores),
            "holdout_std": statistics.stdev(holdout_scores) if len(holdout_scores) > 1 else None,
        })

    scored = [r for r in results if r.get("holdout_val_bpb") is not None]
    if len(scored) < 2:
        print("\n[holdout_eval] Fewer than 2 runs produced holdout_val_bpb — "
              "check that train.py's holdout eval ran (see its stdout above for errors).")
        sys.exit(1)

    by_val = sorted(scored, key=lambda r: r["val_bpb"])
    by_holdout = sorted(scored, key=lambda r: r["holdout_val_bpb"])
    val_rank = {r["run_id"]: i + 1 for i, r in enumerate(by_val)}
    holdout_rank = {r["run_id"]: i + 1 for i, r in enumerate(by_holdout)}

    rho = spearman(
        [val_rank[r["run_id"]] for r in scored],
        [holdout_rank[r["run_id"]] for r in scored],
    )
    bias_detected = by_val[0]["run_id"] != by_holdout[0]["run_id"]

    best = by_holdout[0]
    lines = [
        "# Holdout Evaluation Report",
        "",
        f"Re-trained top-{len(scored)} configs with `holdout_eval: true`, "
        f"{len(seeds)} seed(s) each. Every row is the MEAN across seeds -- never the "
        "best of them, since best-of-N over noisy draws is the statistic that inflates "
        "a frontier in the first place. `val_bpb` and `holdout_val_bpb` always come "
        "from the same trained model.",
        "",
        "## Campaign result, ranked by holdout",
        "",
        "This ordering -- not the val_bpb one -- is the campaign's answer. The "
        "validation shard chose these candidates, so it cannot also be the impartial "
        "judge of them.",
        "",
        "| rank | run_id | (was) | holdout_val_bpb | +- | val_bpb | gap | val_rank |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(by_holdout, start=1):
        spread = f"{r['holdout_std']:.6f}" if r.get("holdout_std") is not None else "n/a"
        lines.append(
            f"| {rank} | {r['run_id']} | {r['original_run_id']} | "
            f"{r['holdout_val_bpb']:.6f} | {spread} | {r['val_bpb']:.6f} | "
            f"{r['holdout_val_bpb'] - r['val_bpb']:+.6f} | {val_rank[r['run_id']]} |"
        )

    lines += ["", f"**Winner on holdout: `{best['original_run_id']}`** "
                  f"(holdout {best['holdout_val_bpb']:.6f}).", ""]
    if best.get("holdout_std") is not None:
        rivals = [r for r in by_holdout[1:]
                  if abs(r["holdout_val_bpb"] - best["holdout_val_bpb"]) <= 2 * best["holdout_std"]]
        if rivals:
            names = ", ".join(f"`{r['original_run_id']}`" for r in rivals)
            lines += [f"Not separable from {names} -- within 2 standard deviations of the "
                      f"winner's own seed spread. Treat these as tied rather than ranked.", ""]

    lines += [
        f"Spearman(val_rank, holdout_rank) = {rho:.4f}",
        "",
        "**SELECTION BIAS DETECTED**: best-on-val config is not best-on-holdout."
        if bias_detected else
        "No selection bias detected: best-on-val config is also best-on-holdout.",
        "",
        render_drift(holdout_drift(load_results(args.results_path))),
    ]
    report = "\n".join(lines) + "\n"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n{report}")
    print(f"[holdout_eval] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
