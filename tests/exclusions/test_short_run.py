"""Tests for network_qa.exclusions.short_run — flag functional runs too short
to be usable (aborted / prematurely-ended scans) so lev1 excludes them.

Mirrors the monolith's `scripts/check_tr.sh` heuristic (per-task expected
length = the statistical MODE of dim4 across that task's runs), and the
qa_runs.py short-run flagger's multi-echo dedup. This is a header-only read:
no BIDS data is modified. Fixtures build tiny synthetic BOLD NIfTIs with
nibabel so dim4 is fully controllable.
"""
import numpy as np
import nibabel as nib
from argparse import Namespace
from pathlib import Path

from network_qa.exclusions.short_run import (
    SHORT_RUN_FRACTION,
    ShortRunGenerator,
    _scan_short_runs,
)


def _write_bold(bids_dir: Path, subject: str, session: str, task: str, run: str,
                n_vols: int, echo: int | None = None) -> Path:
    """Write a synthetic `*_bold.nii.gz` with `n_vols` volumes. `subject`/`session`
    are already BIDS-prefixed (e.g. `sub-s01`, `ses-01`)."""
    func_dir = bids_dir / subject / session / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    parts = [subject, session, f"task-{task}", f"run-{run}"]
    if echo is not None:
        parts.append(f"echo-{echo}")
    name = "_".join(parts) + "_bold.nii.gz"
    path = func_dir / name
    img = nib.Nifti1Image(np.zeros((2, 2, 2, n_vols), dtype=np.int16), np.eye(4))
    nib.save(img, str(path))
    return path


def test_generator_attributes():
    g = ShortRunGenerator()
    assert g.name == "short_run"
    assert g.description


def test_registered_as_short_run():
    from network_qa.exclusions.base import get_generator
    import network_qa.exclusions.short_run  # noqa: F401 (trigger registration)
    gen = get_generator("short_run")
    assert gen is not None
    assert gen.name == "short_run"


def test_mode_computed_and_run_at_mode_not_flagged(tmp_path):
    """The per-task expected length is the mode of dim4; a run AT the mode is
    never flagged."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(tmp_path, "sub-s02", "ses-01", "stroop", "1", 100)

    entries = _scan_short_runs(tmp_path, SHORT_RUN_FRACTION)
    assert entries == []


def test_run_below_half_is_flagged(tmp_path):
    """A run shorter than 50% of the task mode IS flagged; entry shape matches
    the exclusions contract (source/action/reason/metrics)."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(tmp_path, "sub-s02", "ses-01", "stroop", "1", 40)   # aborted

    entries = _scan_short_runs(tmp_path, SHORT_RUN_FRACTION)
    assert len(entries) == 1
    e = entries[0]
    assert e["subject"] == "sub-s02"
    assert e["session"] == "ses-01"
    assert e["task"] == "task-stroop"
    assert e["run"] == "run-1"
    assert e["source"] == "short-run"
    assert e["action"] == "exclude"
    assert e["reason"] == "40/100 TRs (40% of task mode, <50% cutoff)"
    assert e["metrics"] == {"dim4": 40, "expected": 100, "fraction": 40 / 100}


def test_run_exactly_at_fraction_is_kept(tmp_path):
    """Strict `<`: a run at EXACTLY the fraction of the mode is kept."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(tmp_path, "sub-s02", "ses-01", "stroop", "1", 50)   # exactly half

    entries = _scan_short_runs(tmp_path, SHORT_RUN_FRACTION)
    assert entries == []


def test_multi_echo_run_dedups_to_one_entry(tmp_path):
    """All 3 echoes of one short run collapse to ONE logical run -> ONE entry."""
    for echo in (1, 2, 3):
        _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100, echo=echo)
        _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100, echo=echo)
        _write_bold(tmp_path, "sub-s02", "ses-01", "stroop", "1", 30, echo=echo)  # abort

    entries = _scan_short_runs(tmp_path, SHORT_RUN_FRACTION)
    assert len(entries) == 1
    assert entries[0]["subject"] == "sub-s02"
    assert entries[0]["run"] == "run-1"


def test_subjects_filter_drops_out_of_sample_subject(tmp_path):
    """A short run in a subject not in the dataset subjects_file is not emitted
    (and does not even contribute to the mode)."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(tmp_path, "sub-s99", "ses-01", "stroop", "1", 30)   # aborted, out of sample

    entries = _scan_short_runs(tmp_path, SHORT_RUN_FRACTION, subjects={"sub-s01"})
    assert entries == []


def test_two_tasks_each_judged_against_own_mode(tmp_path):
    """Same dim4 gets opposite verdicts under different task modes: a 60-vol run
    is kept under a mode-100 task but flagged under a mode-200 task."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "taskA", "1", 100)
    _write_bold(tmp_path, "sub-s02", "ses-01", "taskA", "1", 100)
    _write_bold(tmp_path, "sub-s03", "ses-01", "taskA", "1", 60)    # 60 vs mode 100 -> keep

    _write_bold(tmp_path, "sub-s01", "ses-01", "taskB", "1", 200)
    _write_bold(tmp_path, "sub-s02", "ses-01", "taskB", "1", 200)
    _write_bold(tmp_path, "sub-s03", "ses-01", "taskB", "1", 60)    # 60 vs mode 200 -> drop

    entries = _scan_short_runs(tmp_path, SHORT_RUN_FRACTION)
    assert len(entries) == 1
    e = entries[0]
    assert e["task"] == "task-taskB"
    assert e["subject"] == "sub-s03"
    assert e["metrics"]["expected"] == 200


def test_empty_or_missing_dir_no_crash(tmp_path):
    assert _scan_short_runs(tmp_path, SHORT_RUN_FRACTION) == []
    assert _scan_short_runs(tmp_path / "does_not_exist", SHORT_RUN_FRACTION) == []


def test_unreadable_file_is_skipped_no_crash(tmp_path):
    """A corrupt/unreadable BOLD file is skipped (no crash, no entry)."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    func_dir = tmp_path / "sub-s02" / "ses-01" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    (func_dir / "sub-s02_ses-01_task-stroop_run-1_bold.nii.gz").write_bytes(b"not a nifti")

    entries = _scan_short_runs(tmp_path, SHORT_RUN_FRACTION)
    assert entries == []


def test_short_run_fraction_override_via_generate(tmp_path):
    """--short-run-fraction changes the cutoff: a 60/100 run is kept at 0.5 but
    flagged at 0.7."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(tmp_path, "sub-s02", "ses-01", "stroop", "1", 60)

    g = ShortRunGenerator()
    config = {"bids_dir": str(tmp_path)}

    kept = g.generate("test", config, Namespace(short_run_fraction=0.5))
    assert kept == []

    flagged = g.generate("test", config, Namespace(short_run_fraction=0.7))
    assert len(flagged) == 1
    assert flagged[0]["source"] == "short-run"
    assert flagged[0]["reason"] == "60/100 TRs (60% of task mode, <70% cutoff)"


def test_generate_reads_bids_dir_from_config_and_default_fraction(tmp_path):
    """generate() with no short_run_fraction attr falls back to SHORT_RUN_FRACTION
    and reads bids_dir from dataset_config, mirroring behavioral.py."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(tmp_path, "sub-s02", "ses-01", "stroop", "1", 20)

    g = ShortRunGenerator()
    entries = g.generate("test", {"bids_dir": str(tmp_path)}, Namespace())
    assert len(entries) == 1
    assert entries[0]["subject"] == "sub-s02"


def test_generate_honors_subjects_file(tmp_path):
    """generate() resolves the subjects filter through dataset_config, exactly
    like behavioral.py."""
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(tmp_path, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(tmp_path, "sub-s99", "ses-01", "stroop", "1", 20)

    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("s01\n")

    g = ShortRunGenerator()
    config = {"bids_dir": str(tmp_path), "subjects_file": str(subjects_file)}
    entries = g.generate("test", config, Namespace())
    assert entries == []
