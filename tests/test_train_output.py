"""Tests for agents/train_output.py -- the one definition of how train.py's
final summary block is parsed.

These behaviours existed before the extraction but were never covered: they
lived in two hand-synced private copies that nothing exercised directly. Now
that there is a single definition, this is where they are pinned down.
"""

import math

from agents import train_output


FULL_SUMMARY = """
[train.py] Starting validation eval...
[====================] 100.0% | eval step 400/400
---
val_bpb:          1.302689
training_seconds: 291.4
total_seconds:    412.7
peak_vram_mb:     647.6
mfu_percent:      2.54
total_tokens_M:   12.5
num_steps:        1526
budget_shortfall_pct: 0.00
num_params_M:     239.6
depth:            8
seed:             7
interpretable_scalars: {"resid_lambdas": [1.0, 0.9], "x0_lambdas": [0.1]}
head_ablation_impacts: {"0.1": 0.004}
token_fingerprint: {"induction_score": 0.12}
"""


def test_parses_every_scalar_field_with_the_right_type():
    m = train_output.parse(FULL_SUMMARY)

    assert m["val_bpb"] == 1.302689
    assert m["training_time"] == 291.4
    assert m["total_seconds"] == 412.7
    assert m["peak_vram_mb"] == 647.6
    assert m["mfu_percent"] == 2.54
    assert m["total_tokens_M"] == 12.5
    assert m["num_steps"] == 1526 and isinstance(m["num_steps"], int)
    assert m["budget_shortfall_pct"] == 0.0
    assert m["num_params_M"] == 239.6
    assert m["depth"] == 8 and isinstance(m["depth"], int)
    assert m["seed"] == 7 and isinstance(m["seed"], int)


def test_parses_json_evidence_blobs():
    m = train_output.parse(FULL_SUMMARY)

    assert m["interpretable_scalars"]["resid_lambdas"] == [1.0, 0.9]
    assert m["head_ablation_impacts"] == {"0.1": 0.004}
    assert m["token_fingerprint"]["induction_score"] == 0.12


def test_returns_only_what_was_found_and_invents_no_defaults():
    """Callers merge this onto their own base dict, so a key that is absent has
    to stay absent -- that is what lets them distinguish "train.py did not
    report this" from "it reported something"."""
    m = train_output.parse("val_bpb:  1.25\n")

    assert m == {"val_bpb": 1.25}
    assert "status" not in m and "holdout_val_bpb" not in m


def test_a_malformed_value_is_skipped_without_discarding_the_rest():
    """A summary block can be truncated mid-line when an SSH channel drops. One
    unparseable field must not take the val_bpb above it down with it."""
    m = train_output.parse(
        "val_bpb:          1.302689\n"
        "num_steps:        not-a-number\n"
        "depth:            8\n"
    )

    assert m["val_bpb"] == 1.302689
    assert m["depth"] == 8
    assert "num_steps" not in m


def test_a_label_with_no_value_is_skipped():
    """train.py prints `holdout_val_bpb:` with nothing after it when the holdout
    eval did not run."""
    m = train_output.parse("val_bpb: 1.25\nholdout_val_bpb:\n")

    assert m["val_bpb"] == 1.25
    assert "holdout_val_bpb" not in m


def test_a_malformed_json_blob_is_skipped_rather_than_stored_as_text():
    m = train_output.parse("hyperparam_clamps: not json at all\nval_bpb: 1.25\n")

    assert "hyperparam_clamps" not in m
    assert m["val_bpb"] == 1.25


def test_progress_and_log_lines_are_ignored():
    m = train_output.parse(
        "[========------------]  40.0% | loss: 3.2 | lrm: 1.00 | tok/sec: 9,001\n"
        "[hyperparams] DEPTH=8 N_HEAD=4 SEED=7\n"
        "Scaling AdamW LRs by 1/sqrt(512/768) = 1.224745\n"
    )

    assert m == {}, "only the summary block's own labels may be picked up"


def test_the_last_occurrence_of_a_field_wins():
    """train.py prints the summary once, but a partial-output retry can leave
    two blocks in the captured stream; the later one is the more complete."""
    m = train_output.parse("val_bpb: 9.99\nval_bpb: 1.25\n")

    assert m["val_bpb"] == 1.25


def test_empty_output_parses_to_nothing_rather_than_raising():
    assert train_output.parse("") == {}


def test_fields_and_train_py_summary_labels_match_exactly():
    """Guard in both directions against the producer and the parser drifting.
    Reads train.py's source rather than a copy of the list, so a summary line
    added there without a FIELDS entry (silently unparsed for every run) and a
    FIELDS entry for a line train.py no longer prints (dead config that reads
    as live) both fail here."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "train.py").read_text(encoding="utf-8")
    # Lines of the shape: print(f"some_label:   {value}")
    printed = {label.lower() for label in re.findall(r'print\(f"([a-zA-Z_]+):\s+\{', src)}
    known = {k.rstrip(":").lower() for k in train_output.FIELDS}

    # Without this the whole test passes vacuously the moment train.py's print
    # style changes and the regex stops matching anything.
    assert len(printed) >= 10, f"regex matched only {printed} -- it has stopped tracking train.py"
    assert printed == known, (
        f"train.py prints {sorted(printed - known)} that nothing parses; "
        f"FIELDS has {sorted(known - printed)} that train.py never prints"
    )


def test_json_evidence_keys_match_what_train_py_emits():
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "train.py").read_text(encoding="utf-8")
    emitted = set(re.findall(r'print\("([a-z_]+): " \+ json\.dumps', src))

    assert len(emitted) >= 4, f"regex matched only {emitted} -- it has stopped tracking train.py"
    assert emitted == set(train_output.JSON_KEYS), (
        f"train.py emits {sorted(emitted - set(train_output.JSON_KEYS))} that nothing parses; "
        f"JSON_KEYS has {sorted(set(train_output.JSON_KEYS) - emitted)} that train.py never emits"
    )
