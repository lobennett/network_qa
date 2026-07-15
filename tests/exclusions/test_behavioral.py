"""Tests for network_qa.exclusions.behavioral — ported from
neuro_workflow/tests/exclusions/test_behavioral.py and
neuro_workflow/tests/events/test_behavioral_exclusion_generator.py
(two separate monolith test files covering the same generator),
import-repointed."""
import json
from argparse import Namespace

import pytest
from pathlib import Path
from unittest.mock import patch

from network_qa.exclusions.behavioral import (
    NONMONOTONIC_EXCLUDE_FRACTION,
    BehavioralGenerator,
    _scan_nonmonotonic_exclusions,
)


def test_generator_attributes():
    g = BehavioralGenerator()
    assert g.name == "behavioral"
    assert g.description


def test_generate_returns_empty(tmp_path):
    g = BehavioralGenerator()
    config = {"bids_dir": str(tmp_path)}
    args = Namespace()
    entries = g.generate("test", config, args)
    assert entries == []


class TestBehavioralGenerator:
    def test_registered_as_behavioral(self):
        from network_qa.exclusions.base import get_generator
        import network_qa.exclusions.behavioral  # noqa: F401 (trigger registration)
        gen = get_generator("behavioral")
        assert gen is not None
        assert gen.name == "behavioral"

    def test_generate_returns_list(self, tmp_path):
        from network_qa.exclusions.base import get_generator
        import network_qa.exclusions.behavioral  # noqa: F401
        gen = get_generator("behavioral")
        # Create minimal sourcedata structure
        beh_dir = tmp_path / "sourcedata" / "sub-s01" / "ses-01" / "beh"
        beh_dir.mkdir(parents=True)
        # Also need bids_dir with func dir
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        func_dir.mkdir(parents=True)

        from argparse import Namespace
        args = Namespace(behavioral_dir=str(tmp_path / "sourcedata"))
        config = {"bids_dir": str(tmp_path)}
        result = gen.generate("test", config, args)
        assert isinstance(result, list)

    def test_exclusion_entry_format(self):
        """Entries must have required fields for the exclusions system."""
        entry = {
            "subject": "sub-s01",
            "session": "ses-01",
            "task": "stopSignal",
            "run": "run-1",
            "action": "exclude",
            "source": "behavioral-qc",
            "reason": "test reason",
        }
        required = {"subject", "session", "task", "run", "action", "reason"}
        assert required.issubset(entry.keys())


def _write_sidecar(func_dir: Path, subject: str, session: str, task: str, run: str,
                    expected: int, retained: int, fraction: float) -> Path:
    func_dir.mkdir(parents=True, exist_ok=True)
    name = f"{subject}_{session}_task-{task}_run-{run}_events.json"
    path = func_dir / name
    path.write_text(json.dumps({
        "NTestTrialsExpected": expected,
        "NTestTrialsRetained": retained,
        "FractionTestTrialsDropped": fraction,
    }))
    return path


class TestNonmonotonicExclusionRule:
    """The >50%-test-trials-dropped policy ported from neuro_workflow.events.qc
    (NONMONOTONIC_EXCLUDE_FRACTION = 0.5, strict `>` comparison). This is the
    decision half of the truncation/decision split: network_events computes +
    writes the _events.json sidecar (truncation), network_qa reads it and
    decides whether to exclude the scan (this module)."""

    def test_excludes_run_over_threshold(self, tmp_path):
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s01", "ses-01", "stopSignal", "1",
                       expected=10, retained=3, fraction=0.7)

        entries = _scan_nonmonotonic_exclusions(tmp_path, NONMONOTONIC_EXCLUDE_FRACTION)

        assert len(entries) == 1
        e = entries[0]
        assert e["subject"] == "sub-s01"
        assert e["session"] == "ses-01"
        assert e["task"] == "task-stopSignal"
        assert e["run"] == "run-1"
        assert e["action"] == "exclude"
        assert e["source"] == "behavioral-qc"
        assert "7/10" in e["reason"]
        assert "non-monotonic onset truncation" in e["reason"]
        assert e["metrics"] == {
            "NTestTrialsExpected": 10,
            "NTestTrialsRetained": 3,
            "FractionTestTrialsDropped": 0.7,
        }

    def test_keeps_run_under_threshold(self, tmp_path):
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s01", "ses-01", "stopSignal", "1",
                       expected=10, retained=7, fraction=0.3)

        entries = _scan_nonmonotonic_exclusions(tmp_path, NONMONOTONIC_EXCLUDE_FRACTION)
        assert entries == []

    def test_keeps_run_zero_dropped(self, tmp_path):
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s01", "ses-01", "stopSignal", "1",
                       expected=10, retained=10, fraction=0.0)

        entries = _scan_nonmonotonic_exclusions(tmp_path, NONMONOTONIC_EXCLUDE_FRACTION)
        assert entries == []

    def test_threshold_boundary_exactly_half_is_kept(self, tmp_path):
        """The monolith uses strict `>` (events/qc.py:
        `tstats["fraction_test_dropped"] > NONMONOTONIC_EXCLUDE_FRACTION`) --
        a run dropping EXACTLY the threshold fraction is NOT excluded."""
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s01", "ses-01", "stopSignal", "1",
                       expected=10, retained=5, fraction=0.5)

        entries = _scan_nonmonotonic_exclusions(tmp_path, NONMONOTONIC_EXCLUDE_FRACTION)
        assert entries == []

    def test_just_over_threshold_is_excluded(self, tmp_path):
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s01", "ses-01", "stopSignal", "1",
                       expected=100, retained=49, fraction=0.51)

        entries = _scan_nonmonotonic_exclusions(tmp_path, NONMONOTONIC_EXCLUDE_FRACTION)
        assert len(entries) == 1

    def test_missing_sidecar_no_crash_no_entry(self, tmp_path):
        """A run with an _events.tsv but no _events.json sidecar (e.g. an
        older network_events version, or a run whose events generation
        failed) is treated as 0 dropped: no crash, no entry."""
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        func_dir.mkdir(parents=True)
        (func_dir / "sub-s01_ses-01_task-stopSignal_run-1_events.tsv").write_text("onset\tduration\n")

        entries = _scan_nonmonotonic_exclusions(tmp_path, NONMONOTONIC_EXCLUDE_FRACTION)
        assert entries == []

    def test_subject_filter_excludes_out_of_sample_subject(self, tmp_path):
        func_dir = tmp_path / "sub-s99" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s99", "ses-01", "stopSignal", "1",
                       expected=10, retained=1, fraction=0.9)

        entries = _scan_nonmonotonic_exclusions(tmp_path, NONMONOTONIC_EXCLUDE_FRACTION,
                                                 subjects={"sub-s01"})
        assert entries == []

    def test_custom_threshold_via_generator_cli_arg(self, tmp_path):
        """generate() reads args.nonmonotonic_exclude_fraction, not just the
        module default."""
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s01", "ses-01", "stopSignal", "1",
                       expected=10, retained=8, fraction=0.2)

        g = BehavioralGenerator()
        config = {"bids_dir": str(tmp_path)}
        args = Namespace(nonmonotonic_exclude_fraction=0.1)
        entries = g.generate("test", config, args)
        assert len(entries) == 1
        assert entries[0]["source"] == "behavioral-qc"

    def test_generate_wires_nonmonotonic_entries_through(self, tmp_path):
        func_dir = tmp_path / "sub-s01" / "ses-01" / "func"
        _write_sidecar(func_dir, "sub-s01", "ses-01", "stopSignal", "1",
                       expected=10, retained=2, fraction=0.8)

        g = BehavioralGenerator()
        config = {"bids_dir": str(tmp_path)}
        args = Namespace()
        entries = g.generate("test", config, args)
        assert len(entries) == 1
        assert entries[0]["task"] == "task-stopSignal"
