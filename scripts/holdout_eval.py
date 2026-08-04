"""Check whether the search loop is fitting noise in its pinned validation
shard (see prepare.py's HOLDOUT_SHARD / evaluate_bpb_holdout).

Re-trains the top-K configs (by historical val_bpb) from results.tsv with
holdout_eval enabled, so each retrain reports both val_bpb (the shard the
search always compares against) and holdout_val_bpb (a shard the search
never sees) from the *same* trained model. If the config that ranks best on
val_bpb isn't also best on holdout_val_bpb, the search has been measuring
its own selection bias, not real improvement.

Cost: K full training runs. Run this once at the end of a search campaign,
not per-iteration -- the search loop itself never uses holdout_val_bpb to
decide anything (see prepare.py's _document_batches: HOLDOUT_SHARD is
excluded from training and never touched by evaluate_bpb).

Usage:
    uv run python scripts/holdout_eval.py --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from state.results_analysis import HYPERPARAM_COLUMNS, load_results, spearman
from state.results_logger import log_result

RUN_ID_PREFIX = "holdout_check"
REPORT_PATH = Path("reports/holdout_eval_report.md")


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
    parser.add_argument("--config", default="agents_config.yaml")
    parser.add_argument("--results-path", default="results.tsv")
    args = parser.parse_args()

    rows = load_results(args.results_path)
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

    print(f"[holdout_eval] Re-checking top-{len(candidates)} configs on the holdout shard...")
    agent1 = Agent1TrainingSpecialist(config_path=args.config)

    results = []
    for i, cand in enumerate(candidates):
        hp = {col: cand[col] for col in HYPERPARAM_COLUMNS if col in cand}
        hp["holdout_eval"] = True
        original_run_id = cand.get("run_id", f"unknown_{i}")
        run_id = f"{RUN_ID_PREFIX}_{i:04d}"
        print(f"\n[holdout_eval] --- {run_id} (was {original_run_id}, "
              f"historical val_bpb={cand['val_bpb']:.6f}) ---")

        agent1.current_hyperparams = dict(hp)
        metrics = agent1.train_model(hp, dry_run=False, iteration=i)
        log_result(run_id, hp, metrics, results_path=args.results_path)

        results.append({
            "run_id": run_id,
            "original_run_id": original_run_id,
            "val_bpb": metrics.get("val_bpb", float("inf")),
            "holdout_val_bpb": metrics.get("holdout_val_bpb"),
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

    lines = [
        "# Holdout Evaluation Report",
        "",
        f"Re-trained top-{len(scored)} configs with `holdout_eval: true`; each row's "
        "`val_bpb`/`holdout_val_bpb` come from the same trained model.",
        "",
        "| run_id | (was) | val_bpb | holdout_val_bpb | delta | val_rank | holdout_rank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in scored:
        delta = r["holdout_val_bpb"] - r["val_bpb"]
        lines.append(
            f"| {r['run_id']} | {r['original_run_id']} | {r['val_bpb']:.6f} | "
            f"{r['holdout_val_bpb']:.6f} | {delta:+.6f} | {val_rank[r['run_id']]} | "
            f"{holdout_rank[r['run_id']]} |"
        )
    lines += [
        "",
        f"Spearman(val_rank, holdout_rank) = {rho:.4f}",
        "",
        "**SELECTION BIAS DETECTED**: best-on-val config is not best-on-holdout."
        if bias_detected else
        "No selection bias detected: best-on-val config is also best-on-holdout.",
    ]
    report = "\n".join(lines) + "\n"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n{report}")
    print(f"[holdout_eval] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
