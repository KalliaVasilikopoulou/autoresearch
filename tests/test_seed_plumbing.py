"""End-to-end guard for the seed reaching train.py and coming back recorded.

The seed is only useful as evidence if the value in results.tsv is the one the
run ACTUALLY initialized with. train.py echoes `seed:` in its summary block for
exactly that reason, and there are two near-identical parsers that have to pick
it up -- agents/remote_runner.py's (every real run) and
Agent1TrainingSpecialist's (the local-subprocess fallback). A field added to
only one of them is silently wrong for whichever path is not covered.
"""

import re
from pathlib import Path

from agents import remote_runner
from agents.agent1_training_specialist import Agent1TrainingSpecialist, DEFAULT_SEED
from state.results_analysis import load_results
from state.results_logger import log_result

TRAIN_PY = Path(__file__).resolve().parent.parent / "train.py"

_SUMMARY = """
[hyperparams] DEPTH=8 N_HEAD=4 SEED=7
---
val_bpb:          1.302689
training_seconds: 291.4
num_steps:        1526
depth:            8
budget_shortfall_pct: 0.00
seed:             7
"""


def test_remote_parser_reads_the_seed_train_py_reported():
    metrics = remote_runner._parse_output(_SUMMARY)
    assert metrics["seed"] == 7


def test_local_subprocess_parser_reads_the_seed_too(tmp_path):
    """Both parsers, one behaviour -- they are near-copies of each other and
    have drifted before."""
    specialist = Agent1TrainingSpecialist(config_path=str(tmp_path / "missing.yaml"),
                                          root_dir=str(tmp_path))
    metrics = specialist._parse_training_output(_SUMMARY)
    assert metrics["seed"] == 7


def test_there_is_only_one_field_map_to_keep_in_sync():
    """Replaces a test that compared the two copies' `seed:` entries. Equality
    of two definitions is a weaker guarantee than there being one: it passes for
    every field it happens to check and says nothing about the next field added
    to only one side. The private copies must be gone, not merely equal."""
    from agents import train_output
    from agents.agent1_training_specialist import Agent1TrainingSpecialist as A1

    assert not hasattr(remote_runner, "_OUTPUT_FIELDS")
    assert not hasattr(remote_runner, "_JSON_OUTPUT_KEYS")
    assert not hasattr(A1, "_TRAIN_OUTPUT_FIELDS")
    assert not hasattr(A1, "_JSON_OUTPUT_KEYS")
    assert train_output.FIELDS["seed:"] == ("seed", int)


def test_both_call_sites_extract_identical_metrics(tmp_path):
    """The behavioural guarantee, independent of how the sharing is arranged:
    the same stdout yields the same measurements on both paths. Only the
    path-specific defaults may differ."""
    specialist = Agent1TrainingSpecialist(config_path=str(tmp_path / "missing.yaml"),
                                          root_dir=str(tmp_path))
    local = specialist._parse_training_output(_SUMMARY)
    remote = remote_runner._parse_output(_SUMMARY)

    assert remote["status"] == "remote_ok", "remote keeps its transport status default"
    assert "status" not in local, "the local path lets train_model set the status"
    assert {k: v for k, v in remote.items() if k != "status"} == local


def test_train_py_reads_the_seed_from_hyperparams_instead_of_hardcoding_it():
    """train.py can't be imported here (it needs CUDA and runs at import), so
    this asserts on the source: the seed must come from the hyperparams file,
    and `manual_seed` must no longer take a literal."""
    src = TRAIN_PY.read_text(encoding="utf-8")

    assert 'SEED = _clamp("seed"' in src, "train.py must read `seed` from model_hyperparams.yaml"
    assert "torch.manual_seed(SEED)" in src
    assert "torch.cuda.manual_seed(SEED)" in src
    assert not re.search(r"manual_seed\(\s*\d+\s*\)", src), \
        "no hardcoded seed literal may remain -- that is the bug being fixed"
    assert re.search(r'print\(f"seed:\s+\{SEED\}"\)', src), \
        "train.py must echo the seed it used so it can be logged, not just requested"


def test_agent1_default_hyperparams_carry_the_seed_but_never_search_it():
    from agents.agent1_training_specialist import SEARCH_SPACE

    specialist = Agent1TrainingSpecialist(config_path="does-not-exist.yaml")
    defaults = specialist._default_hyperparams()

    assert defaults["seed"] == DEFAULT_SEED
    assert "seed" not in SEARCH_SPACE, \
        "a seed in SEARCH_SPACE would make the search optimize for the luckiest init"


def test_seed_survives_the_whole_path_from_proposal_to_results_tsv(tmp_path):
    """The one that matters: propose -> train.py reports -> parsed -> logged."""
    hyperparams = {"n_layer": 8, "n_embd": 512, "seed": 7}
    metrics = remote_runner._parse_output(_SUMMARY)
    metrics["status"] = "remote_ok"

    results = tmp_path / "results.tsv"
    log_result("seedvar_c0_s7", hyperparams, metrics, results_path=str(results))

    rows = load_results(str(results))
    assert rows[0]["seed"] == 7
    assert rows[0]["val_bpb"] == 1.302689
