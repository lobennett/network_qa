"""Tests for network_qa.exclusions.qa_decisions — ported from
neuro_workflow/tests/exclusions/test_qa_decisions.py, import-repointed.

Adapted: the original `test_generator_output_flows_through_compile` drove
neuro_workflow's `core.exclusions` sources-dir/EXCLUSIONS_DIR persistence
layer. That layer doesn't exist in network_qa; the equivalent
generator-output-flows-through-compile integration coverage lives in
tests/exclusions/test_compile.py (Task 6), which exercises the real
qa_decisions generator through network_qa.compile.compile_exclusions.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest


def test_qa_decisions_generator_importable():
    """The generator module imports and exposes QADecisionsGenerator."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    assert QADecisionsGenerator.name == "qa_decisions"


def _write_tsv(path: Path, rows: list[dict]) -> None:
    """Write a minimal qa decisions TSV (subject, session, task, run, action, reason)."""
    fieldnames = ["subject", "session", "task", "run", "action", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_args(tsv_path: Path) -> "object":
    """Minimal Namespace stand-in for args (only attributes the generator reads)."""
    from argparse import Namespace
    return Namespace(decisions_tsv=tsv_path)


def test_generator_has_cli_arg_for_decisions_tsv():
    """The generator declares --decisions-tsv on its parser."""
    from argparse import ArgumentParser
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator

    parser = ArgumentParser()
    QADecisionsGenerator().add_cli_args(parser)
    args = parser.parse_args(["--decisions-tsv", "/tmp/whatever.tsv"])
    assert str(args.decisions_tsv) == "/tmp/whatever.tsv"


def test_scan_level_exclude_emits_one_entry(tmp_path):
    """A single scan-level action=exclude row -> one entry, BIDS-prefixed."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [
        {"subject": "sub-s03", "session": "ses-02", "task": "task-cuedTS",
         "run": "run-1", "action": "exclude", "reason": "noisy task data"},
    ])

    entries = QADecisionsGenerator().generate("discovery", {}, _make_args(tsv))

    assert len(entries) == 1
    e = entries[0]
    assert e == {
        "subject": "sub-s03",
        "session": "ses-02",
        "task": "task-cuedTS",
        "run": "run-1",
        "source": "qa_decisions",
        "action": "exclude",
        "reason": "qa_decisions: noisy task data (scan-level)",
    }


def test_pass_and_review_rows_skipped(tmp_path, capsys):
    """Mixed actions: only `exclude` produces entries; summary line counts the others."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [
        {"subject": "sub-s03", "session": "ses-02", "task": "task-cuedTS",
         "run": "run-1", "action": "exclude", "reason": "noisy"},
        {"subject": "sub-s10", "session": "ses-01", "task": "task-flanker",
         "run": "run-1", "action": "review", "reason": "borderline RT"},
        {"subject": "sub-s19", "session": "ses-03", "task": "task-goNogo",
         "run": "run-1", "action": "pass", "reason": "looks fine"},
    ])

    entries = QADecisionsGenerator().generate("discovery", {}, _make_args(tsv))
    captured = capsys.readouterr()

    assert len(entries) == 1
    assert entries[0]["subject"] == "sub-s03"
    assert "1 excluded" in captured.out
    assert "1 review-skipped" in captured.out
    assert "1 pass-skipped" in captured.out


def _make_fake_bids(tmp_path, subject: str, scans: list[tuple[str, str, str]]) -> Path:
    """Build a minimal BIDS-like dir with empty BOLD files for the given scans.

    Each scan tuple is (session, task, run), e.g. ('ses-02', 'cuedTS', '1').
    Returns the BIDS dir root.
    """
    bids = tmp_path / "bids"
    for session, task, run in scans:
        func = bids / subject / session / "func"
        func.mkdir(parents=True, exist_ok=True)
        fname = f"{subject}_{session}_task-{task}_run-{run}_bold.nii.gz"
        (func / fname).write_bytes(b"")
    return bids


def test_subject_level_exclude_expands_via_bids_glob(tmp_path, capsys):
    """A subject-level exclude row -> one entry per matched BOLD file in BIDS."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    bids_dir = _make_fake_bids(tmp_path, "sub-s03", [
        ("ses-01", "flanker", "1"),
        ("ses-02", "cuedTS", "1"),
        ("ses-02", "stopSignal", "1"),
    ])
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [
        {"subject": "sub-s03", "session": "-", "task": "-", "run": "-",
         "action": "exclude", "reason": "dropped from cohort"},
    ])

    config = {"bids_dir": str(bids_dir)}
    entries = QADecisionsGenerator().generate("discovery", config, _make_args(tsv))
    captured = capsys.readouterr()

    assert len(entries) == 3
    assert {e["subject"] for e in entries} == {"sub-s03"}
    for e in entries:
        assert "(subject-level)" in e["reason"]
        assert "dropped from cohort" in e["reason"]
        assert e["action"] == "exclude"
        assert e["source"] == "qa_decisions"
        assert e["task"].startswith("task-")
        assert e["run"].startswith("run-")
        assert e["session"].startswith("ses-")
    sessions_tasks = [(e["session"], e["task"], e["run"]) for e in entries]
    assert sessions_tasks == sorted(sessions_tasks)
    assert "0 scan-level" in captured.out
    assert "3 expanded from 1 subject-level" in captured.out


def test_subject_level_with_no_bids_files_emits_zero(tmp_path, capsys):
    """Subject-level row for a sub with no BOLD scans in BIDS -> 0 entries, no error."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [
        {"subject": "sub-s99", "session": "-", "task": "-", "run": "-",
         "action": "exclude", "reason": "missing data"},
    ])

    config = {"bids_dir": str(bids_dir)}
    entries = QADecisionsGenerator().generate("discovery", config, _make_args(tsv))
    captured = capsys.readouterr()

    assert entries == []
    assert "0 expanded from 1 subject-level" in captured.out


def test_subject_filter_drops_non_member_scan_level(tmp_path):
    """Scan-level rows whose subject isn't in subjects_file are dropped."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [
        {"subject": "sub-s03", "session": "ses-02", "task": "task-cuedTS",
         "run": "run-1", "action": "exclude", "reason": "in dataset"},
        {"subject": "sub-s1035", "session": "ses-02", "task": "task-flanker",
         "run": "run-1", "action": "exclude", "reason": "out of dataset"},
    ])
    subjects_path = tmp_path / "subjects_discovery.txt"
    subjects_path.write_text("s03\ns10\n")

    config = {"subjects_file": str(subjects_path)}
    entries = QADecisionsGenerator().generate("discovery", config, _make_args(tsv))

    assert len(entries) == 1
    assert entries[0]["subject"] == "sub-s03"


def test_subject_filter_drops_subject_level_before_glob(tmp_path):
    """Subject-level row for non-member subject is dropped before BIDS glob fires.

    Even with a populated BIDS dir for the non-member subject, no entries are
    emitted; the only entries come from the in-roster subject.
    """
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    bids_dir = _make_fake_bids(tmp_path, "sub-s03", [
        ("ses-01", "flanker", "1"),
    ])
    # Out-of-dataset subject also has BIDS files — these should be ignored.
    _make_fake_bids(tmp_path, "sub-s1035", [
        ("ses-01", "flanker", "1"),
        ("ses-02", "cuedTS", "1"),
    ])
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [
        {"subject": "sub-s03", "session": "-", "task": "-", "run": "-",
         "action": "exclude", "reason": "in dataset"},
        {"subject": "sub-s1035", "session": "-", "task": "-", "run": "-",
         "action": "exclude", "reason": "out of dataset"},
    ])
    subjects_path = tmp_path / "subjects_discovery.txt"
    subjects_path.write_text("s03\ns10\n")

    config = {"bids_dir": str(bids_dir), "subjects_file": str(subjects_path)}
    entries = QADecisionsGenerator().generate("discovery", config, _make_args(tsv))

    assert len(entries) == 1
    assert entries[0]["subject"] == "sub-s03"
    assert entries[0]["task"] == "task-flanker"


def test_missing_tsv_raises_file_not_found_error(tmp_path):
    """Bogus TSV path -> FileNotFoundError with the path in the message."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    bogus = tmp_path / "does_not_exist.tsv"
    with pytest.raises(FileNotFoundError, match=str(bogus)):
        QADecisionsGenerator().generate("discovery", {}, _make_args(bogus))


def test_empty_tsv_returns_empty_list(tmp_path):
    """TSV with header only returns []."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [])
    entries = QADecisionsGenerator().generate("discovery", {}, _make_args(tsv))
    assert entries == []


def test_invalid_action_propagates_value_error(tmp_path):
    """Unknown action value (e.g. 'maybe') propagates ValueError from load_decisions."""
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator
    tsv = tmp_path / "decisions.tsv"
    _write_tsv(tsv, [
        {"subject": "sub-s03", "session": "ses-02", "task": "task-cuedTS",
         "run": "run-1", "action": "maybe", "reason": "uh"},
    ])
    with pytest.raises(ValueError, match="invalid action"):
        QADecisionsGenerator().generate("discovery", {}, _make_args(tsv))
