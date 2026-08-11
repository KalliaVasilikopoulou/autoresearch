"""Is architecture ONE SMOOTH HILL in size? Six runs up a fixed-shape ladder.

WHY THIS MATTERS. Agent 4 currently treats the three architecture parameters as
a space to EXPLORE: it opens regions, fences them, judges them, retires them.
That machinery exists because a rugged landscape has to be sampled -- you cannot
walk downhill on a surface full of holes. If val_bpb is instead a smooth
function of model size, none of that is needed on this axis: the correct
behaviour is to walk, one step at a time, in the direction that improved. So
this experiment is not asking "which architecture is best". It is asking
WHAT KIND OF SURFACE ARCHITECTURE IS, which decides how much machinery the
architecture axis deserves at all.

SIZE, NOT SHAPE. "Bigger" is ambiguous -- a model can grow by getting deeper,
wider, or by adding heads, and those are different moves. This sweep separates
the two questions by holding SHAPE fixed and moving only SIZE:

    head_dim = n_embd / n_head   is frozen at the anchor's value
    n_embd / n_layer             is held ~constant (integer n_layer permitting)

so every rung is the same model drawn at a different scale. Scaling n_head with
n_embd (rather than freezing n_head and letting head_dim shrink) is the
controlled choice: a head with fewer channels is a different kind of head, and
that would smuggle a shape change into a size sweep. SHAPE AT FIXED SIZE
REMAINS UNANSWERED and needs its own sweep.

THE TUNABLES ARE FROZEN, AND THAT IS A REAL LIMIT. All eight tunables stay at
the anchor's values across a ~190x span of size, so a rung can look bad either
because that size is wrong or because the anchor's learning rates are wrong for
that size. Re-tuning per rung would cost a campaign, not six runs. The limit is
tolerable because of WHAT IS BEING ASKED: a learning-rate mismatch that grows
as you move away from the anchor bends the curve, it does not make it jagged.
Smoothness therefore survives the confound; the exact position of any minimum
does not, and is reported as such.

READING THE RESULT. Three questions, in the order that matters:

  readable     do the steps between rungs clear the noise? If they do not, no
               hill-climber can tell which way to go whatever the shape is.
  one direction  at most one real change of direction. "Real" means both steps
               either side clear the noise; a direction change below the noise
               floor IS the noise. This is form-free -- it makes no assumption
               about what the curve looks like -- and it is exactly the
               property hill-climbing needs.
  predictable  does a scaling law fit down to the noise floor? A bonus, not a
               requirement: it says whether Agent 4 could EXTRAPOLATE and skip
               rungs rather than only step. It deliberately does not get a vote
               in the verdict -- a badly fitting curve can mean the surface is
               rough OR just that the chosen formula is the wrong shape, and
               those two are not distinguishable from six points.

The fitted form is the scaling law L(N) = L_inf + c * N^-alpha, not a
polynomial. A quadratic in log(size) misfits a saturating curve badly enough to
call a textbook-clean descent "rough" -- which is a statement about the formula,
not about the surface.

Noise here is the spread of a comparison BETWEEN two architectures, which
scripts/region_geometry.py measured as ~0.00199 (an upper bound -- that
estimate was anchor-dominated). Using an upper bound is the conservative
direction: it can only make this experiment call a real wiggle "noise", never
invent smoothness that is not there.

WHY THESE RUNS ARE NOT LOGGED TO results.tsv. They are deliberately un-tuned
configurations, most of them far from anything the campaign would propose.
Feeding them to the surrogate would teach it that small models are bad -- when
what they actually are is bad AT THE ANCHOR'S LEARNING RATES. They go to their
own file.

Usage:
    .venv\\Scripts\\python.exe scripts/size_sweep.py --dry-run     # plan only
    .venv\\Scripts\\python.exe scripts/size_sweep.py
    .venv\\Scripts\\python.exe scripts/size_sweep.py --analyze-only
"""

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from state import surrogate
from state.results_analysis import (
    ARCHITECTURE_COLUMNS,
    HYPERPARAM_COLUMNS,
    TUNABLE_COLUMNS,
    load_results,
)
from state.results_logger import log_result

RUN_ID_PREFIX = "size"
DEFAULT_RESULTS_PATH = "state/size_sweep.tsv"
REPORT_JSON = Path("state/size_sweep.json")
REPORT_MD = Path("reports/size_sweep.md")

#: One seed for every rung. Differences between rungs are then pure
#: configuration differences (plus whatever the seed interacts with), and 42
#: keeps them comparable to the campaign's history.
SEED = 42

N_RUNGS = 6

#: Search-space ceilings, from agent1's ARCH_SAFE_RANGES. Imported lazily in
#: _search_space() so this module stays importable without the agent stack.
OK_STATUSES = {"remote_ok", "ok"}

#: Fallback only. The real value is read from state/region_geometry.json; see
#: `architecture_noise()`. Never let a threshold silently run on a guess -- the
#: saturation rule learned that lesson the hard way (a 7.5x-too-large default
#: would have retired every region on sight).
FALLBACK_SIGMA = 0.00197

#: A step counts as a real change of direction only if BOTH adjacent steps
#: clear this many sigma. Below that, a direction change is the noise.
REAL_STEP_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Building the ladder
# ---------------------------------------------------------------------------

def _search_space() -> Dict[str, Tuple[float, float]]:
    from agents.agent1_training_specialist import SEARCH_SPACE
    return SEARCH_SPACE


def pick_anchor(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The campaign's best complete run -- the same anchor step 4 used.

    The frontier is the only place the answer is actionable: a hill measured in
    a bad part of the space describes a neighbourhood the search never visits.
    """
    usable = [r for r in rows
              if r.get("status") in OK_STATUSES
              and isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])
              and not (isinstance(r.get("budget_shortfall_pct"), (int, float))
                       and r["budget_shortfall_pct"] > 0)
              and all(c in r for c in HYPERPARAM_COLUMNS)]
    if not usable:
        raise SystemExit("[size_sweep] No complete runs to anchor on.")
    best = min(usable, key=lambda r: r["val_bpb"])
    hp = {c: float(best[c]) for c in HYPERPARAM_COLUMNS}
    for c in ("n_layer", "n_embd", "n_head", "batch_size"):
        hp[c] = int(round(hp[c]))
    hp["_from_run"] = best.get("run_id")
    hp["_historical_val_bpb"] = best["val_bpb"]
    return hp


def non_embedding_params(n_layer: int, n_embd: int) -> int:
    """12 * n_layer * n_embd^2 -- exact for this architecture, not a rule of
    thumb: attention contributes q+k+v+proj = 4*n_embd^2 per layer (n_kv_head
    equals n_head here, so k and v are full width) and the 4x MLP contributes
    another 8.

    Embeddings are excluded on purpose. The vocab table is ~53k x n_embd, which
    at the narrow end of this ladder is an order of magnitude larger than the
    transformer itself -- including it would compress the whole x-axis into
    almost nothing and hide the very curvature being measured.
    """
    return 12 * int(n_layer) * int(n_embd) ** 2


def _rung_head_counts(anchor_head: int, n_rungs: int) -> List[int]:
    """Which n_head values to place rungs at, from 1 up to the anchor's.

    The ladder climbs TO the anchor rather than past it because the anchor is
    the frontier: the question is what the surface looks like on the approach
    to the best known point. (Going past it is blocked anyway -- see
    build_ladder, where the n_embd ceiling bites.)
    """
    if anchor_head <= n_rungs:
        return list(range(1, anchor_head + 1))
    # Log-spaced, so each rung is a similar MULTIPLE of the last. Size grows
    # like h^3 along this ladder, so equal steps in h would bunch every rung
    # into the top of the range.
    lo, hi = math.log(1), math.log(anchor_head)
    picks = {max(1, int(round(math.exp(lo + (hi - lo) * i / (n_rungs - 1)))))
             for i in range(n_rungs)}
    return sorted(picks)


def build_ladder(anchor: Dict[str, Any], n_rungs: int = N_RUNGS) -> List[Dict[str, Any]]:
    """The rungs: same shape, different size, everything else the anchor's."""
    bounds = _search_space()
    embd_lo, embd_hi = bounds["n_embd"]
    layer_lo, layer_hi = bounds["n_layer"]

    anchor_head = int(anchor["n_head"])
    head_dim = int(anchor["n_embd"]) // anchor_head
    #: layers per head-group, so n_embd/n_layer stays put as both grow
    aspect = int(anchor["n_layer"]) / anchor_head

    rungs = []
    for h in _rung_head_counts(anchor_head, n_rungs):
        n_embd = head_dim * h
        # Round half UP rather than to even: Python's round() sends 10.5 to 10
        # and 3.5 to 4, which would put a visible kink in the aspect ratio for
        # no reason other than the rounding rule.
        n_layer = int(math.floor(aspect * h + 0.5))
        n_layer = int(min(max(n_layer, max(1, layer_lo)), layer_hi))
        if not (embd_lo <= n_embd <= embd_hi):
            continue
        hp = dict(anchor)
        hp["n_head"], hp["n_embd"], hp["n_layer"] = h, n_embd, n_layer
        # Belt and braces: train.py re-snaps n_embd itself, and a rung that
        # trains at a different width than the one recorded here would be
        # analysed against the wrong x-axis.
        hp["n_embd"] = surrogate.snap_n_embd(n_embd, h)
        rungs.append({
            "label": f"h{h}",
            "n_head": h, "n_embd": hp["n_embd"], "n_layer": n_layer,
            "params": non_embedding_params(n_layer, hp["n_embd"]),
            "aspect": hp["n_embd"] / n_layer,
            "hyperparams": hp,
        })
    return rungs


# ---------------------------------------------------------------------------
# Reusing measurements that already exist
# ---------------------------------------------------------------------------

def _same_config(row: Dict[str, Any], hp: Dict[str, Any]) -> bool:
    """Every hyperparameter equal, and the seed equal to this sweep's.

    Used to avoid re-buying a rung the campaign or an earlier experiment has
    already paid for. Equality has to hold on ALL of them: a row that matches
    the architecture but not the learning rates is a different measurement,
    and quietly reusing it would put a differently-tuned point on the curve.
    """
    seed = row.get("seed")
    if not isinstance(seed, (int, float)) or int(seed) != SEED:
        return False
    for c in HYPERPARAM_COLUMNS:
        want, got = hp.get(c), row.get(c)
        if not isinstance(got, (int, float)):
            return False
        if c in ARCHITECTURE_COLUMNS or c == "batch_size":
            if int(round(got)) != int(round(float(want))):
                return False
        elif not math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-12):
            return False
    return True


def import_existing(rungs: List[Dict[str, Any]], results_path: str,
                    sources: List[str]) -> int:
    """Copy any already-measured rung in from another experiment's results.

    The top rung IS the anchor configuration, and step 4 already ran exactly it
    at seed 42 with the same analysis switches off -- that is the single most
    expensive run in the ladder, and re-buying it would be paying twice for the
    same number.
    """
    have = already_done(results_path)
    imported = 0
    for src in sources:
        if not Path(src).exists():
            continue
        rows = load_results(src)
        for rung in rungs:
            run_id = f"{RUN_ID_PREFIX}_{rung['label']}"
            if run_id in have:
                continue
            for row in rows:
                if row.get("status") not in OK_STATUSES or not _same_config(row, rung["hyperparams"]):
                    continue
                val = row.get("val_bpb")
                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    continue
                metrics = {k: row.get(k) for k in
                           ("val_bpb", "training_time", "peak_vram_mb", "mfu_percent",
                            "num_params_M", "num_steps", "status", "budget_shortfall_pct")}
                metrics["status"] = "remote_ok"
                print(f"[size_sweep] {run_id}: reusing {row.get('run_id')} from {src} "
                      f"(val_bpb={val:.6f}) -- identical config and seed.")
                log_result(run_id, _hyperparams_for(rung), metrics, results_path=results_path)
                have.add(run_id)
                imported += 1
                break
    return imported


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _hyperparams_for(rung: Dict[str, Any]) -> Dict[str, Any]:
    hp = {k: v for k, v in rung["hyperparams"].items() if not k.startswith("_")}
    hp["seed"] = SEED
    # Pure measurement runs: post-training analysis costs GPU time and cannot
    # change val_bpb.
    hp["ablation_k"] = 0
    hp["token_xai_enabled"] = False
    return hp


def already_done(results_path: str) -> set:
    """run_ids already carrying a real measurement, so a re-run does only what
    is missing."""
    done = set()
    for row in load_results(results_path):
        if row.get("status") in OK_STATUSES and isinstance(row.get("val_bpb"), (int, float)) \
                and math.isfinite(row["val_bpb"]):
            done.add(str(row.get("run_id")))
    return done


def run_all(rungs: List[Dict[str, Any]], results_path: str, timeout: int) -> None:
    from agents import remote_runner
    from agents.live_progress import MultiGpuProgressDisplay

    if not remote_runner.is_remote_configured():
        raise SystemExit("[size_sweep] No remote configured.")

    done = already_done(results_path)
    pending = [r for r in rungs if f"{RUN_ID_PREFIX}_{r['label']}" not in done]
    if len(pending) < len(rungs):
        print(f"[size_sweep] Resuming: {len(rungs) - len(pending)} rung(s) already "
              f"measured, {len(pending)} to run.")
    if not pending:
        print("[size_sweep] Nothing to run.")
        return

    hp_dir = Path("state/size_sweep_hyperparams")
    hp_dir.mkdir(parents=True, exist_ok=True)

    # One connection PER WAVE, and stop if a whole wave dies -- when the SSH
    # session drops, train.py dies with it and every concurrent run in that
    # wave dies at the same instant. Step 4 lost 13 of 39 cells learning this.
    wave_no = 0
    start = 0
    while start < len(pending):
        wave_no += 1
        try:
            client = remote_runner.open_client()
        except Exception as e:
            print(f"[size_sweep] Could not reach the server for wave {wave_no}: {e}")
            break
        try:
            if not remote_runner.sync_remote_code(client=client):
                print(f"[size_sweep] Sync failed on wave {wave_no} -- stopping so the "
                      f"remaining rungs stay re-runnable rather than becoming inf rows.")
                break
            gpus = [g["index"] for g in remote_runner.discover_available_gpus(client=client)]
            if not gpus:
                print("[size_sweep] No free GPUs -- stopping; re-run to continue.")
                break

            wave = pending[start:start + len(gpus)]
            wave_statuses: List[Optional[str]] = []
            labels = [f"GPU{gpus[i]}" for i in range(len(wave))]
            print(f"\n[size_sweep] === wave {wave_no} ({len(wave)} run(s), "
                  f"{start}/{len(pending)} done) ===")
            with MultiGpuProgressDisplay(labels) as display:
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    futures = {}
                    for i, rung in enumerate(wave):
                        run_id = f"{RUN_ID_PREFIX}_{rung['label']}"
                        hp = _hyperparams_for(rung)
                        path = hp_dir / f"{run_id}.yaml"
                        with open(path, "w", encoding="utf-8") as f:
                            yaml.dump(hp, f)
                        fut = pool.submit(
                            remote_runner.run_training_remote,
                            hyperparams_local_path=str(path), gpu_index=gpus[i],
                            hp_remote_name=f"model_hyperparams_{run_id}.yaml",
                            run_label=f"GPU{gpus[i]}", timeout=timeout,
                            skip_sync=True, display=display, client=client)
                        futures[fut] = (run_id, hp)
                    for fut in as_completed(futures):
                        run_id, hp = futures[fut]
                        try:
                            metrics = fut.result()
                        except Exception as e:
                            display.print_line(f"[size_sweep] {run_id} failed: {e}")
                            metrics = {"val_bpb": float("inf"), "status": "remote_error",
                                       "error": str(e)}
                        display.print_line(f"[size_sweep] {run_id}: "
                                           f"val_bpb={metrics.get('val_bpb')} "
                                           f"status={metrics.get('status')}")
                        log_result(run_id, hp, metrics, results_path=results_path)
                        wave_statuses.append(metrics.get("status"))
            start += len(wave)

            if wave_statuses and all(s == "remote_error" for s in wave_statuses):
                print(f"\n[size_sweep] Every run in wave {wave_no} failed. Stopping rather "
                      f"than spending the remaining {len(pending) - start} rung(s) against "
                      f"the same fault -- re-run to resume where this left off.")
                break
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def architecture_noise(state_dir: Path = Path("state")) -> Tuple[float, str]:
    """How much a comparison BETWEEN two architectures moves, and where the
    number came from.

    Prefers the paired-difference spread measured in step 4. That estimate was
    anchor-dominated, so it OVERSTATES the noise -- which is the safe direction
    here: an overstated noise floor can only make this experiment dismiss a
    real wiggle as noise, never manufacture smoothness.
    """
    geom = state_dir / "region_geometry.json"
    if geom.exists():
        try:
            data = json.loads(geom.read_text(encoding="utf-8"))
            stds = [e["std_of_gap"] for e in (data.get("a_between") or {}).values()
                    if isinstance(e.get("std_of_gap"), (int, float))]
            if stds:
                return max(stds), "region_geometry.json (A-between, paired difference)"
            if isinstance(data.get("a_within"), (int, float)):
                return data["a_within"], "region_geometry.json (A-within -- understates)"
        except (ValueError, KeyError):
            pass
    seedv = state_dir / "seed_variance.json"
    if seedv.exists():
        try:
            data = json.loads(seedv.read_text(encoding="utf-8"))
            for key in ("sigma_seed", "pooled_std", "sigma"):
                if isinstance(data.get(key), (int, float)):
                    return data[key], f"seed_variance.json ({key})"
        except ValueError:
            pass
    return FALLBACK_SIGMA, "fallback constant (nothing measured on disk)"


def fit_scaling_law(params: List[float], ys: List[float]) -> Dict[str, Any]:
    """Fit L(N) = L_inf + c * N^-alpha -- the standard scaling-law form.

    Three parameters, the same budget a quadratic would spend, but the right
    SHAPE: loss falls steeply at small N and flattens toward an irreducible
    floor. A quadratic in log(N) cannot do that over a 190x span, and its
    misfit shows up as residuals that look exactly like roughness.

    Fitted without scipy by exploiting the structure: fix alpha and the other
    two parameters fall out of ordinary least squares, so a 1-D scan over alpha
    is enough. That also makes the fit deterministic -- no initial guess, no
    convergence to worry about.
    """
    import numpy as np

    n = len(params)
    if n < 4:  # 3 parameters need at least one residual degree of freedom
        return {"insufficient_data": True, "n": n}

    N = np.asarray(params, dtype=float)
    y = np.asarray(ys, dtype=float)
    best = None
    for alpha in np.linspace(0.01, 2.0, 400):
        A = np.column_stack([np.ones(n), N ** (-alpha)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_res = float(resid @ resid)
        if best is None or ss_res < best[0]:
            best = (ss_res, float(alpha), float(coef[0]), float(coef[1]), resid)

    ss_res, alpha, l_inf, c, resid = best
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": n, "alpha": alpha, "l_inf": l_inf, "c": c,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        # Divided by the residual degrees of freedom, not by n: with six points
        # and three parameters a plain RMS flatters the fit badly.
        "residual_rms": math.sqrt(ss_res / max(1, n - 3)),
        "residuals": [float(r) for r in resid],
    }


def history_by_size(history_path: str, n_bins: int = 5) -> List[Dict[str, Any]]:
    """The campaign's own runs, binned by size -- a free cross-check on the one
    weakness this sweep cannot design away.

    Every rung ran at the ANCHOR's learning rates, so a rung could look bad
    because its size is wrong or because those rates are wrong for it. The
    campaign's runs have no such property: each carries whatever tunables the
    search proposed for it, so they are wrong in every direction rather than in
    one. If the best-per-size trend agrees with the ladder, the ladder's shape
    is not an artefact of freezing the tunables.

    Both best and median are reported because best-of-n falls as n grows purely
    by selection, and the bins do not hold equal numbers of runs.
    """
    rows = [r for r in load_results(history_path)
            if r.get("status") in OK_STATUSES
            and isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])
            and all(c in r for c in ARCHITECTURE_COLUMNS)
            and not (isinstance(r.get("budget_shortfall_pct"), (int, float))
                     and r["budget_shortfall_pct"] > 0)]
    if len(rows) < n_bins * 2:
        return []

    for r in rows:
        r["_params"] = non_embedding_params(int(r["n_layer"]), int(r["n_embd"]))
    rows.sort(key=lambda r: r["_params"])

    lo, hi = math.log10(rows[0]["_params"]), math.log10(rows[-1]["_params"])
    width = (hi - lo) / n_bins
    out = []
    for i in range(n_bins):
        edge_lo, edge_hi = lo + i * width, lo + (i + 1) * width
        last = i == n_bins - 1
        members = [r for r in rows
                   # Half-open bins so no run is counted twice, except the top
                   # bin, which has to close so the largest run lands somewhere.
                   if edge_lo <= math.log10(r["_params"]) < edge_hi
                   or (last and math.isclose(math.log10(r["_params"]), edge_hi))]
        if not members:
            continue
        vals = [r["val_bpb"] for r in members]
        out.append({
            "params_lo": 10 ** edge_lo, "params_hi": 10 ** edge_hi,
            "n": len(members), "best": min(vals), "median": statistics.median(vals),
        })
    return out


def analyze(results_path: str, history_path: str = "results.tsv") -> Dict[str, Any]:
    by_label: Dict[str, Dict[str, Any]] = {}
    for row in load_results(results_path):
        run_id = str(row.get("run_id", ""))
        if not run_id.startswith(f"{RUN_ID_PREFIX}_") or row.get("status") not in OK_STATUSES:
            continue
        val = row.get("val_bpb")
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        n_layer, n_embd = int(row["n_layer"]), int(row["n_embd"])
        by_label[run_id[len(RUN_ID_PREFIX) + 1:]] = {
            "n_layer": n_layer, "n_embd": n_embd, "n_head": int(row["n_head"]),
            "params": non_embedding_params(n_layer, n_embd),
            "total_params_M": row.get("num_params_M"),
            "val_bpb": float(val),
            "training_time": row.get("training_time"),
            "budget_shortfall_pct": row.get("budget_shortfall_pct"),
        }

    rungs = sorted(by_label.values(), key=lambda r: r["params"])
    sigma, sigma_source = architecture_noise()

    # --- steps between neighbouring rungs ---
    for i, r in enumerate(rungs):
        if i == 0:
            r["delta"] = r["delta_sigma"] = None
        else:
            d = r["val_bpb"] - rungs[i - 1]["val_bpb"]
            r["delta"] = d
            r["delta_sigma"] = d / sigma if sigma else None
        r["size_ratio"] = (r["params"] / rungs[0]["params"]) if rungs else None

    # --- readable: do the steps clear the noise at all? ---
    # This comes first because it gates everything else. A hill whose slope is
    # below the resolution is not climbable no matter how smooth it is.
    deltas = [r["delta"] for r in rungs[1:]]
    threshold = REAL_STEP_SIGMA * sigma
    real = [d for d in deltas if abs(d) >= threshold]
    below_noise = len(deltas) - len(real)

    # --- one direction: how many REAL changes of direction? ---
    # A hill has at most one. Only real steps get a vote: two rungs that differ
    # by less than the noise have no trustworthy direction, so a flip between
    # them says nothing about the surface. Form-free by construction -- it
    # assumes nothing about what the curve looks like, which is why it, and not
    # the fit below, decides the verdict.
    sign_changes = sum(1 for a, b in zip(real, real[1:]) if a * b < 0)

    # --- predictable: does a scaling law explain the rungs? ---
    law = fit_scaling_law([float(r["params"]) for r in rungs],
                          [r["val_bpb"] for r in rungs])
    predictable = (not law.get("insufficient_data")
                   and law["residual_rms"] <= threshold)

    # --- where is the best size, and is the ladder still descending at the top? ---
    best = min(rungs, key=lambda r: r["val_bpb"]) if rungs else None
    last_step = rungs[-1]["delta"] if len(rungs) > 1 else None
    still_falling = bool(last_step is not None and last_step <= -threshold)

    readable = bool(real)
    unimodal = sign_changes <= 1

    # If the ladder is still descending at its top, the obvious next question is
    # how much is left and what it costs. The fitted law answers both, so long
    # as it is read as an extrapolation from six points and not as a promise.
    headroom = None
    if still_falling and not law.get("insufficient_data") and rungs:
        top = rungs[-1]
        # TWO DIFFERENT GAPS, kept apart on purpose. The measured gap is how far
        # the best run actually sits above the floor; the fitted gap is what the
        # law says at that size. They differ by the fit's residual there, and
        # every projection below has to run on the fitted one -- mixing them
        # would quietly attribute the residual to a change in size.
        measured_gap = top["val_bpb"] - law["l_inf"]
        fitted_gap = law["c"] * top["params"] ** -law["alpha"]
        # Size needed to close half of what is left: the gap halves when
        # N^-alpha halves, i.e. at 2^(1/alpha) times the size.
        multiple = 2 ** (1.0 / law["alpha"]) if law["alpha"] > 0 else None
        headroom = {
            "measured_gap": measured_gap,
            "measured_gap_in_sigma": measured_gap / sigma if sigma else None,
            "fitted_gap": fitted_gap,
            "size_multiple_to_halve_it": multiple,
            "params_to_halve_it": top["params"] * multiple if multiple else None,
            # What one more step of the ladder would buy, if the box allowed it.
            "next_doubling_gain": fitted_gap * (1 - 2 ** -law["alpha"]),
        }

    # The cross-check: does the campaign's own history, whose runs were NOT all
    # tuned at the anchor's rates, trend the same way?
    bins = history_by_size(history_path)
    bests = [b["best"] for b in bins]
    history_agrees = None
    if len(bests) >= 3:
        history_turns = sum(1 for a, b, c in zip(bests, bests[1:], bests[2:])
                            if (b - a) * (c - b) < 0)
        history_agrees = history_turns <= 1

    return {
        "history_bins": bins, "history_agrees": history_agrees,
        "rungs": rungs, "sigma": sigma, "sigma_source": sigma_source,
        "real_step_sigma": REAL_STEP_SIGMA, "step_threshold": threshold,
        "steps_below_noise": below_noise, "n_steps": len(deltas),
        "real_sign_changes": sign_changes,
        "scaling_law": law,
        "best_rung": best["params"] if best else None,
        "best_is_largest": bool(best and rungs and best is rungs[-1]),
        "best_is_smallest": bool(best and rungs and best is rungs[0]),
        "last_step": last_step, "still_falling_at_the_top": still_falling,
        "headroom": headroom,
        "readable": readable, "unimodal": unimodal, "predictable": predictable,
        "verdict": _verdict(readable, unimodal, sign_changes, below_noise, len(deltas)),
    }


def _verdict(readable: bool, unimodal: bool, sign_changes: int,
             below_noise: int, n_steps: int) -> str:
    if not readable:
        return ("UNREADABLE -- no step between rungs clears the noise. Size does "
                "not matter over this range, or the noise estimate is too "
                "pessimistic to tell.")
    if unimodal:
        base = ("ONE HILL -- at most one change of direction, so Agent 4 can walk "
                "downhill on size instead of exploring it.")
        if below_noise:
            base += (f" ({below_noise} of {n_steps} steps were below the noise, "
                     f"so those stretches are flat rather than sloped.)")
        return base
    return (f"NOT A HILL -- {sign_changes} real changes of direction. A single "
            f"step's direction does not point at the optimum, so size keeps its "
            f"share of the exploration machinery.")


def render(report: Dict[str, Any]) -> str:
    rungs = report["rungs"]
    sigma = report["sigma"]
    lines = ["# Is architecture one smooth hill?", "",
             "Six models of the same shape at different sizes -- `head_dim` and "
             "`n_embd/n_layer` frozen, every tunable frozen at the anchor's value, one "
             "seed. The question is not which size wins but WHAT KIND OF SURFACE size "
             "is, because that decides whether Agent 4 should explore this axis or "
             "just walk down it.", "",
             f"Noise used: **{sigma:.6f}** bpb, from {report['sigma_source']}. "
             f"A step counts as real at {report['real_step_sigma']:.0f}x that.", ""]

    if not rungs:
        return "\n".join(lines + ["No completed rungs yet.", ""]) + "\n"

    lines += ["## The ladder", "",
              "| n_layer | n_embd | n_head | params (non-emb) | x smallest | val_bpb | "
              "step | step / noise |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rungs:
        step = f"{r['delta']:+.6f}" if r["delta"] is not None else "--"
        step_s = f"{r['delta_sigma']:+.1f}" if r["delta_sigma"] is not None else "--"
        lines.append(f"| {r['n_layer']} | {r['n_embd']} | {r['n_head']} | "
                     f"{r['params'] / 1e6:.2f}M | {r['size_ratio']:.0f}x | "
                     f"{r['val_bpb']:.6f} | {step} | {step_s} |")

    lines += ["", "## Is it one hill?", "",
              f"- Steps big enough to have a trustworthy direction: "
              f"**{report['n_steps'] - report['steps_below_noise']} of "
              f"{report['n_steps']}** (a step counts at "
              f"{report['step_threshold']:.6f} bpb).",
              f"- Real changes of direction among those: "
              f"**{report['real_sign_changes']}** -- a hill has at most one.",
              "",
              "> This is the criterion that decides the verdict, and it assumes "
              "nothing about the shape of the curve. Hill-climbing needs exactly one "
              "thing: that the direction which improved keeps pointing at the "
              "optimum. That is what a single change of direction means."]

    law = report["scaling_law"]
    if not law.get("insufficient_data"):
        fit_reads = ("at the noise floor, so the law describes the rungs completely"
                     if report["predictable"] else
                     "above the noise floor, so the law is a summary, not a predictor")
        lines += ["", "## Could Agent 4 predict instead of stepping?", "",
                  "Fitting the scaling law `val_bpb = L_inf + c * N^-alpha`:", "",
                  f"- `alpha` = {law['alpha']:.3f}, `L_inf` = {law['l_inf']:.4f} "
                  f"(the floor this shape approaches as size grows), R^2 = "
                  f"{law['r2']:.4f}.",
                  f"- Residual RMS = {law['residual_rms']:.6f}, "
                  f"{law['residual_rms'] / sigma:.1f}x the noise -- {fit_reads}.",
                  "",
                  "> This does NOT get a vote in the verdict. A poor fit can mean the "
                  "surface is rough, or just that this formula is the wrong shape for "
                  "it, and six points cannot separate those. It is here to answer a "
                  "different question -- whether sizes could be chosen by "
                  "extrapolation rather than one step at a time.",
                  "",
                  "> Three parameters against six points leaves three degrees of "
                  "freedom. Evidence, not proof."]

    bins = report.get("history_bins") or []
    if bins:
        lines += ["", "## Cross-check: the campaign's own runs", "",
                  "Every rung above ran at the anchor's learning rates, so a rung could "
                  "look bad because its size is wrong OR because those rates are wrong "
                  "for it. The campaign's runs do not share that flaw -- each carries "
                  "whatever tunables the search proposed for it, so they are mistuned in "
                  "every direction rather than in one. Agreement here means the ladder's "
                  "shape is not an artefact of the freeze.", "",
                  "| size range (non-emb) | runs | best val_bpb | median |",
                  "|---|---:|---:|---:|"]
        for b in bins:
            lines.append(f"| {b['params_lo'] / 1e6:.1f}M - {b['params_hi'] / 1e6:.1f}M | "
                         f"{b['n']} | {b['best']:.4f} | {b['median']:.4f} |")
        agrees = report.get("history_agrees")
        if agrees is not None:
            lines += ["", ("> The history trends the same way as the ladder."
                           if agrees else
                           "> The history does NOT trend the same way as the ladder -- "
                           "worth resolving before acting on either.")]
        lines += ["", "> Read the `best` column with care: best-of-n falls as n grows "
                  "purely by selection, and the bins do not hold equal numbers of runs. "
                  "The median is the fairer column and the noisier one."]

    lines += ["", "## Where is the best size?", ""]
    if report["best_is_largest"]:
        lines += ["The **largest rung tested wins**, so the minimum is at or beyond the "
                  "edge of the ladder. The ladder stops where it does because "
                  "`n_embd` is capped at 1024 in `ARCH_SAFE_RANGES` -- a search-space "
                  "setting, not a hardware limit (train.py allows 8192). If the curve "
                  "is still falling at the cap, the binding constraint on this campaign "
                  "is the box, not the search.", ""]
    elif report["best_is_smallest"]:
        lines += ["The **smallest rung tested wins** -- unexpected at a fixed token "
                  "budget, and worth checking against the anchor's learning rates "
                  "before believing it.", ""]
    else:
        lines += ["The best rung is in the **interior** of the ladder: a real optimum "
                  "in size, not a boundary effect.", ""]
    if report["last_step"] is not None:
        lines += [f"The final step, {rungs[-2]['params'] / 1e6:.0f}M -> "
                  f"{rungs[-1]['params'] / 1e6:.0f}M, is "
                  f"**{report['last_step']:+.6f}** bpb -- "
                  + ("still falling, so growth had not paid out by the top of the "
                     "ladder." if report["still_falling_at_the_top"] else
                     "inside the noise, so size had stopped paying by the top of the "
                     "ladder."),
                  "",
                  "> That last step is the one Agent 4 would actually take next, which "
                  "makes it the most directly actionable number here.", ""]

    h = report.get("headroom")
    if h:
        lines += [f"How much is left: the best run sits **{h['measured_gap']:.4f}** bpb "
                  f"above the floor the fitted curve approaches "
                  f"({h['measured_gap_in_sigma']:.0f}x the noise -- a real amount, not a "
                  f"rounding error).", "",
                  f"What it costs: another **{h['next_doubling_gain']:.4f}** bpb for the "
                  f"next doubling of size, and **{h['size_multiple_to_halve_it']:.0f}x** "
                  f"the current size to close half of it. Growing is worth doing and "
                  f"will not get far on its own.", "",
                  f"> Those projections run on the law's own gap at the top rung "
                  f"({h['fitted_gap']:.4f}), not on the measured one above -- the two "
                  f"differ by the fit's residual there, and charging that residual to a "
                  f"change in size would be wrong.", "",
                  "> This is the one number here that leans entirely on the fitted law, "
                  "so it inherits every caveat above -- six points, three parameters, "
                  "and an extrapolation past the largest model actually measured.", ""]

    lines += ["> Where the best size SITS is the part of this experiment the frozen "
              "tunables damage most. Every rung ran at the anchor's learning rates, and "
              "a rate tuned for the anchor suits a 190x smaller model less well -- "
              "which drags the apparent optimum toward the anchor. The SHAPE of the "
              "curve survives that; its position does not.", ""]

    lines += ["## Verdict", "", f"**{report['verdict']}**", "",
              "What it changes for Agent 4:", ""]
    if report["unimodal"] and report["readable"]:
        lines += ["- Size does not need regions. One step in the direction that improved, "
                  "repeated, gets there.",
                  "- The exploration machinery (open / fence / judge / retire) is still "
                  "needed for the eight tunables, and for SHAPE at fixed size, which "
                  "this sweep deliberately did not test."]
    elif not report["readable"]:
        lines += ["- Nothing, yet. Before changing how Agent 4 treats size, establish "
                  "that size moves val_bpb at all over this range -- seed replicates on "
                  "the two end rungs would settle it."]
    else:
        lines += ["- Size keeps its share of the exploration machinery: a single step's "
                  "direction is not reliable enough to follow.",
                  "- Before concluding the surface is genuinely rugged, rule out the "
                  "frozen tunables: re-run the two rungs on either side of a turn with "
                  "learning rates re-tuned for their own size."]
    lines += ["", "Not answered here: **shape at fixed size** (deep-and-narrow versus "
              "shallow-and-wide), which needs its own sweep along the constant-size "
              "contour.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default="results.tsv")
    ap.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    ap.add_argument("--timeout", type=int, default=2200)
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; dispatch nothing.")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-reuse", action="store_true",
                    help="Re-run every rung even if an identical measurement exists.")
    args = ap.parse_args()

    if not args.analyze_only:
        anchor = pick_anchor(load_results(args.history))
        print(f"[size_sweep] Anchor: {anchor.get('_from_run')} "
              f"(val_bpb {anchor.get('_historical_val_bpb'):.6f})")
        print(f"[size_sweep]   " + ", ".join(f"{c}={anchor[c]}" for c in ARCHITECTURE_COLUMNS)
              + f", head_dim={int(anchor['n_embd']) // int(anchor['n_head'])}")
        print(f"[size_sweep]   tunables frozen at the anchor's values: "
              + ", ".join(TUNABLE_COLUMNS))

        rungs = build_ladder(anchor)
        print(f"\n[size_sweep] {len(rungs)} rung(s), seed {SEED}:")
        print(f"[size_sweep]   {'label':>6} {'n_layer':>7} {'n_embd':>6} {'n_head':>6} "
              f"{'params':>10} {'n_embd/n_layer':>14}")
        for r in rungs:
            print(f"[size_sweep]   {r['label']:>6} {r['n_layer']:7d} {r['n_embd']:6d} "
                  f"{r['n_head']:6d} {r['params'] / 1e6:9.2f}M {r['aspect']:14.1f}")
        span = rungs[-1]["params"] / rungs[0]["params"] if len(rungs) > 1 else 1
        print(f"[size_sweep]   size span: {span:.0f}x")

        if args.dry_run:
            print("\n[size_sweep] --dry-run: nothing dispatched.")
            return

        if not args.no_reuse:
            n = import_existing(rungs, args.results_path,
                                sources=["state/region_geometry.tsv", args.history])
            if n:
                print(f"[size_sweep] Reused {n} existing measurement(s).")
        run_all(rungs, args.results_path, args.timeout)

    report = analyze(args.results_path, history_path=args.history)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text = render(report)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"[size_sweep] Written to {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
