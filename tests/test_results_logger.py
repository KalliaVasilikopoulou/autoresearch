"""Synthetic-data tests for state/results_logger.py's "device" and
"window_s_fraction" columns -- confirm they round-trip through
log_result/load_results, and that the legacy-schema rename guard still
fires correctly each time COLUMNS grows.

window_s_fraction (dev/checks.txt follow-up: the "search never narrows"
investigation) was in state/results_analysis.py's HYPERPARAM_COLUMNS --
proposed/tuned as a real search dimension -- but was never actually written
to results.tsv, so search_planner.propose_next()'s n_usable count (rows
with every HYPERPARAM_COLUMNS field present) was 0 for every historical
row, forever, and the surrogate's cold-start check never passed.
"""

from state.results_analysis import HYPERPARAM_COLUMNS, load_results
from state.results_logger import COLUMNS, log_result


def test_window_s_fraction_is_in_the_schema():
    """Was `COLUMNS[-1] == "window_s_fraction"`. Position was never the point --
    the bug being guarded is that this column was MISSING entirely -- and
    pinning it to last made the assertion fail the moment any later column
    (seed) was appended, for a reason unrelated to what it tests."""
    assert "window_s_fraction" in COLUMNS


def test_every_hyperparameter_column_is_actually_logged():
    """Regression guard for the actual bug: every field
    state/results_analysis.py's HYPERPARAM_COLUMNS lists as a real search
    dimension must be a real results.tsv column, or search_planner's
    n_usable count silently stays at 0 forever regardless of how much data
    accumulates."""
    for param in HYPERPARAM_COLUMNS:
        assert param in COLUMNS, f"{param} is tuned (HYPERPARAM_COLUMNS) but never logged to results.tsv"


def test_log_result_writes_window_s_fraction_from_hyperparams(tmp_path):
    path = tmp_path / "results.tsv"
    log_result("run_0000", {"n_layer": 8, "window_s_fraction": 0.75}, {"val_bpb": 1.1, "status": "remote_ok"},
               results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert rows[0]["window_s_fraction"] == 0.75


def test_log_result_writes_device_from_metrics(tmp_path):
    path = tmp_path / "results.tsv"
    log_result("run_0000", {"n_layer": 8}, {"val_bpb": 1.1, "status": "remote_ok", "device": 3},
               results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert rows[0]["device"] == "3"


def test_log_result_leaves_device_blank_when_metrics_has_none(tmp_path):
    path = tmp_path / "results.tsv"
    # status "ok" (not "dry_run"/"simulated") -- load_results drops synthetic
    # statuses (see state/results_analysis.py::SYNTHETIC_STATUSES), and this
    # test's actual point is the blank "device" field, not status handling.
    log_result("run_0000", {"n_layer": 8}, {"val_bpb": 1.1, "status": "ok"},
               results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert "device" not in rows[0]  # blank fields are omitted by _coerce_row, not fabricated as ""


def test_previous_schema_is_never_silently_appended_to(tmp_path):
    """Was `..._still_triggers_rename_after_device_column_added`, asserting the
    file got parked. The guarantee it protects is that a mismatched header is
    never appended to blindly (that once left 59 rows under one header and 23
    under another, unparseable as a table) -- and an append-only change now
    satisfies that by MIGRATING instead of parking, which keeps the history.
    Parking is still asserted for headers that can't be migrated, below."""
    from state.results_logger import SUPERSEDED_SCHEMAS

    path = tmp_path / "results.tsv"
    # The most recent REAL superseded layout, not COLUMNS[:-1]: a schema change
    # can append more than one column at a time, and then COLUMNS[:-1] is a
    # width that never existed and is correctly parked rather than migrated.
    old_header = "\t".join(SUPERSEDED_SCHEMAS[0])
    path.write_text(old_header + "\n" + "2024-01-01T00:00:00\trun_0000\t8\n")

    log_result("run_0001", {"n_layer": 8}, {"val_bpb": 1.0, "status": "ok"}, results_path=str(path))

    assert not (tmp_path / "legacy_results.tsv").exists()
    assert path.read_text().splitlines()[0] == "\t".join(COLUMNS)
    rows = load_results(str(path))
    assert [r["run_id"] for r in rows] == ["run_0000", "run_0001"], "old row kept, new row appended"


def test_a_superseded_schema_is_parked_where_git_ignores_it(tmp_path):
    """The migration used to write results.tsv.legacy-* next to results.tsv,
    which matched no ignore rule (.gitignore lists "results.tsv", not
    "results.tsv.legacy-*") and showed up as untracked junk -- guaranteed for
    anyone carrying an older file across the region_id schema change."""
    from state.results_logger import log_result

    results = tmp_path / "results.tsv"
    results.write_text("an\told\theader\n1\t2\t3\n", encoding="utf-8")

    log_result("run_0000", {"n_layer": 8}, {"val_bpb": 1.3, "status": "remote_ok"},
               results_path=str(results))

    parked = list((tmp_path / "legacy_results.tsv").glob("results.tsv.legacy-*"))
    assert len(parked) == 1, "the old file is kept, not deleted"
    assert "an\told\theader" in parked[0].read_text(encoding="utf-8")
    assert not list(tmp_path.glob("results.tsv.legacy-*")), "and not left in the repo root"
    assert results.exists() and "region_id" in results.read_text(encoding="utf-8")


# --- seed: a recorded nuisance variable, not a search dimension -------------


def test_seed_is_logged_but_is_not_a_search_dimension():
    """The whole point of the column. If `seed` ever appears in
    HYPERPARAM_COLUMNS it becomes a surrogate feature and a coordinate of the
    region space, i.e. something the search optimizes -- which would select the
    luckiest initialization rather than averaging over initializations."""
    assert "seed" in COLUMNS
    assert "seed" not in HYPERPARAM_COLUMNS


def test_log_result_prefers_the_seed_train_py_reported_over_the_one_requested(tmp_path):
    """metrics beats hyperparams, same rule as `depth`. They diverge exactly
    when the hyperparams file failed to load and train.py silently fell back to
    its default -- the case worth catching, since the run would otherwise be
    attributed to an initialization it never used."""
    path = tmp_path / "results.tsv"
    log_result("run_0000", {"n_layer": 8, "seed": 7},
               {"val_bpb": 1.1, "status": "remote_ok", "seed": 42},
               results_path=str(path))

    rows = load_results(str(path))
    assert rows[0]["seed"] == 42


def test_log_result_falls_back_to_the_requested_seed(tmp_path):
    """Older train.py builds don't echo `seed:` back, so metrics has no entry."""
    path = tmp_path / "results.tsv"
    log_result("run_0000", {"n_layer": 8, "seed": 7}, {"val_bpb": 1.1, "status": "remote_ok"},
               results_path=str(path))

    rows = load_results(str(path))
    assert rows[0]["seed"] == 7


def test_a_superseded_append_only_schema_is_migrated_in_place_not_parked(tmp_path):
    """Parking loses the history from every consumer -- they all read
    results.tsv, none read the archive. That cost was real: 32 runs went into
    legacy_results.tsv/ this way and the campaign restarted its Sobol cold start
    from zero. An append-only schema change can be migrated instead."""
    from state.results_logger import PRE_SEED_COLUMNS

    path = tmp_path / "results.tsv"
    old_row = ["2026-08-06T18:57:44", "run_0029"] + [""] * (len(PRE_SEED_COLUMNS) - 2)
    old_row[PRE_SEED_COLUMNS.index("val_bpb")] = "1.319072"
    old_row[PRE_SEED_COLUMNS.index("status")] = "remote_ok"
    path.write_text("\t".join(PRE_SEED_COLUMNS) + "\n" + "\t".join(old_row) + "\n", encoding="utf-8")

    log_result("run_0030", {"n_layer": 8, "seed": 3},
               {"val_bpb": 1.2, "status": "remote_ok", "seed": 3}, results_path=str(path))

    assert not (tmp_path / "legacy_results.tsv").exists(), "nothing should have been parked"
    rows = load_results(str(path))
    assert [r["run_id"] for r in rows] == ["run_0029", "run_0030"], "the old row survived"
    assert rows[0]["val_bpb"] == 1.319072
    assert "seed" not in rows[0], "a pre-seed run reports no seed, never a fabricated default"
    assert rows[1]["seed"] == 3


def test_an_unrecognized_header_is_still_parked_rather_than_guessed_at(tmp_path):
    """Only an append is migratable. A reordered or truncated layout cannot be
    padded into the current one, and padding it anyway would file values under
    the wrong headers."""
    path = tmp_path / "results.tsv"
    path.write_text("something\tcompletely\tdifferent\n1\t2\t3\n", encoding="utf-8")

    log_result("run_0000", {"n_layer": 8}, {"val_bpb": 1.3, "status": "remote_ok"},
               results_path=str(path))

    assert len(list((tmp_path / "legacy_results.tsv").glob("results.tsv.legacy-*"))) == 1


def test_rows_written_before_the_seed_column_still_load(tmp_path):
    """Rows are matched by FIELD COUNT, so every superseded width needs a
    frozen tuple in results_analysis or its rows silently stop parsing the
    moment COLUMNS grows -- indistinguishable from having no history at all.
    Real stakes: 32 archived runs under legacy_results.tsv/ are in exactly this
    shape."""
    from state.results_analysis import PRE_SEED_COLUMNS

    assert PRE_SEED_COLUMNS == tuple(
        c for c in COLUMNS if c not in ("seed", "startup_seconds", "eval_seconds"))

    path = tmp_path / "archived.tsv"
    values = {"timestamp": "2026-08-06T18:57:44", "run_id": "run_0029", "n_layer": "19",
              "n_embd": "832", "n_head": "13", "val_bpb": "1.319072", "status": "remote_ok",
              "region_id": "r0005", "window_s_fraction": "0.368"}
    path.write_text(
        "\t".join(PRE_SEED_COLUMNS) + "\n"
        + "\t".join(values.get(c, "") for c in PRE_SEED_COLUMNS) + "\n",
        encoding="utf-8",
    )

    rows = load_results(str(path))
    assert len(rows) == 1, "a pre-seed row must not be silently skipped"
    assert rows[0]["run_id"] == "run_0029"
    assert rows[0]["val_bpb"] == 1.319072
    assert rows[0]["window_s_fraction"] == 0.368
    assert "seed" not in rows[0]  # honestly absent, never fabricated as 42


def test_every_superseded_width_still_parses(tmp_path):
    """THE GUARD THAT WAS MISSING. load_results tells rows apart by FIELD
    COUNT, so a superseded layout that is frozen in results_logger but absent
    from results_analysis's dispatch does not error -- its rows are skipped,
    which is indistinguishable from the file being empty.

    Adding startup_seconds/eval_seconds without updating that dispatch took
    results.tsv, region_geometry.tsv, seed_variance.tsv and size_sweep.tsv to
    zero rows each in a single edit. In-place migration does not cover it:
    that only runs when a file is WRITTEN, and the experiment files are only
    ever read.
    """
    from state.results_analysis import load_results
    from state.results_logger import SUPERSEDED_SCHEMAS

    for i, schema in enumerate(SUPERSEDED_SCHEMAS):
        values = {"timestamp": "2026-08-06T18:57:44", "run_id": f"run_{i:04d}",
                  "n_layer": "19", "n_embd": "832", "n_head": "13",
                  "val_bpb": "1.319072", "status": "remote_ok"}
        path = tmp_path / f"schema_{len(schema)}.tsv"
        path.write_text("\t".join(schema) + "\n"
                        + "\t".join(values.get(c, "") for c in schema) + "\n",
                        encoding="utf-8")

        rows = load_results(str(path))
        assert len(rows) == 1, (
            f"a {len(schema)}-column row was skipped: that width is frozen in "
            f"SUPERSEDED_SCHEMAS but missing from load_results' dispatch")
        assert rows[0]["val_bpb"] == 1.319072


def test_the_real_results_files_on_disk_still_parse():
    """The end-to-end version of the above, against the actual measurements
    this project has made. Cheap, and it fails loudly the moment a schema
    change orphans them."""
    import os

    from state.results_analysis import load_results

    for name in ("results.tsv", "state/region_geometry.tsv",
                 "state/seed_variance.tsv", "state/size_sweep.tsv"):
        if not os.path.exists(name):
            continue
        assert load_results(name), f"{name} parsed to zero rows"
