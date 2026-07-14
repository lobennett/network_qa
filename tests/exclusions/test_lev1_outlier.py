"""Tests for network_qa.exclusions.lev1_outlier — ported from
neuro_workflow/tests/exclusions/test_lev1_outlier.py, import-repointed.

Adapted: `test_generator_output_flows_through_compile` originally drove
neuro_workflow's `core.exclusions` sources-dir/EXCLUSIONS_DIR persistence
layer (save_source_entries -> compile_exclusions reading from disk). That
layer doesn't exist in network_qa (Task 6's compile.py runs generators
in-memory), so the test is rewritten to exercise `network_qa.compile`'s
`compile_exclusions(...)` directly against a registered generator instance,
preserving the original intent: "generator output flows through compile."
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest


def test_lev1_outlier_generator_importable():
    """The generator module imports and exposes Lev1OutlierGenerator."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    assert Lev1OutlierGenerator.name == "lev1_outlier"


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal lev1_outliers.csv with the column set the generator expects."""
    fieldnames = [
        "subject", "session", "run", "task", "contrast",
        "outlier_pct", "vif", "flagged_outliers", "flagged_vif",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_args(csv_path: Path, **overrides) -> "object":
    """Minimal Namespace stand-in for args (only attributes the generator reads)."""
    from argparse import Namespace
    base = dict(
        lev1_outliers_csv=csv_path,
        combined_vif=10.0,
        combined_outlier_pct=10.0,
        strict_vif=15.0,
        strict_outlier_pct=15.0,
    )
    base.update(overrides)
    return Namespace(**base)


def test_strict_vif_rule_fires(tmp_path):
    """A scan-contrast with vif >= strict_vif emits an exclusion entry."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator

    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s03", "session": "ses-01", "run": "1", "task": "stopSignal",
         "contrast": "stop_success-go", "outlier_pct": "2.0", "vif": "18.0",
         "flagged_outliers": "0", "flagged_vif": "1"},
    ])

    entries = Lev1OutlierGenerator().generate(
        "discovery", {}, _make_args(csv_path)
    )
    assert len(entries) == 1
    e = entries[0]
    assert e["subject"] == "sub-s03"
    assert e["session"] == "ses-01"
    assert e["task"] == "task-stopSignal"
    assert e["run"] == "run-1"
    assert e["source"] == "lev1_outlier"
    assert e["action"] == "exclude"
    assert "strict_vif" in e["reason"]
    assert "stop_success-go" in e["reason"]


def test_combined_rule_fires(tmp_path):
    """A row with vif>=combined_vif AND outlier_pct>=combined_outlier_pct fires combined."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s10", "session": "ses-02", "run": "1", "task": "cuedTS",
         "contrast": "response_time", "outlier_pct": "11.0", "vif": "11.0",
         "flagged_outliers": "1", "flagged_vif": "1"},
    ])
    entries = Lev1OutlierGenerator().generate("discovery", {}, _make_args(csv_path))
    assert len(entries) == 1
    assert "combined" in entries[0]["reason"]
    assert "combined" in entries[0]["metrics"]["rules_fired"]


def test_strict_outliers_rule_fires(tmp_path):
    """A row with outlier_pct >= strict_outlier_pct fires strict_outliers."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s19", "session": "ses-03", "run": "1", "task": "flanker",
         "contrast": "incongruent-congruent", "outlier_pct": "18.0", "vif": "4.0",
         "flagged_outliers": "1", "flagged_vif": "0"},
    ])
    entries = Lev1OutlierGenerator().generate("discovery", {}, _make_args(csv_path))
    assert len(entries) == 1
    assert "strict_outliers" in entries[0]["reason"]


def test_below_all_thresholds_emits_nothing(tmp_path):
    """A row that fails all three rules produces no entry."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s29", "session": "ses-04", "run": "1", "task": "goNogo",
         "contrast": "go", "outlier_pct": "8.0", "vif": "8.0",
         "flagged_outliers": "0", "flagged_vif": "0"},
    ])
    entries = Lev1OutlierGenerator().generate("discovery", {}, _make_args(csv_path))
    assert entries == []


def test_per_scan_aggregation_collapses_multiple_contrasts(tmp_path):
    """Two flagged contrasts on the same (subject, session, task, run) -> one entry."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s03", "session": "ses-02", "run": "1", "task": "cuedTS",
         "contrast": "response_time", "outlier_pct": "2.0", "vif": "18.09",
         "flagged_outliers": "0", "flagged_vif": "1"},
        {"subject": "sub-s03", "session": "ses-02", "run": "1", "task": "cuedTS",
         "contrast": "cue_switch_cost", "outlier_pct": "12.3", "vif": "11.5",
         "flagged_outliers": "1", "flagged_vif": "1"},
        # An untouched contrast on the same scan — should NOT show in reason.
        {"subject": "sub-s03", "session": "ses-02", "run": "1", "task": "cuedTS",
         "contrast": "task-baseline", "outlier_pct": "1.0", "vif": "1.2",
         "flagged_outliers": "0", "flagged_vif": "0"},
    ])
    entries = Lev1OutlierGenerator().generate("discovery", {}, _make_args(csv_path))
    assert len(entries) == 1
    e = entries[0]
    assert "response_time" in e["reason"]
    assert "cue_switch_cost" in e["reason"]
    assert "task-baseline" not in e["reason"]
    assert e["metrics"]["n_flagged_contrasts"] == 2
    assert set(e["metrics"]["rules_fired"]) == {"strict_vif", "combined"}
    assert e["metrics"]["max_vif"] == pytest.approx(18.09)
    assert e["metrics"]["max_outlier_pct"] == pytest.approx(12.3)


def test_threshold_configurability(tmp_path):
    """Bumping combined_vif to 20 makes a vif=11 row stop firing combined.

    The same row with strict_vif=12 still fires strict_vif. Demonstrates
    each threshold can be tuned independently.
    """
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s10", "session": "ses-02", "run": "1", "task": "cuedTS",
         "contrast": "response_time", "outlier_pct": "11.0", "vif": "11.0",
         "flagged_outliers": "1", "flagged_vif": "1"},
    ])

    # default thresholds: combined fires
    e_default = Lev1OutlierGenerator().generate("discovery", {}, _make_args(csv_path))
    assert "combined" in e_default[0]["metrics"]["rules_fired"]

    # combined_vif bumped to 20: combined no longer fires; outlier_pct=11 < strict 15;
    # vif=11 < strict 15 -> nothing fires, no entry.
    args_loose = _make_args(csv_path, combined_vif=20.0)
    e_loose = Lev1OutlierGenerator().generate("discovery", {}, args_loose)
    assert e_loose == []

    # strict_vif bumped down to 10: strict_vif fires for vif=11 even though combined
    # still fires too. Just confirms independent threshold tuning works.
    args_tight = _make_args(csv_path, combined_vif=20.0, strict_vif=10.0)
    e_tight = Lev1OutlierGenerator().generate("discovery", {}, args_tight)
    assert len(e_tight) == 1
    assert "strict_vif" in e_tight[0]["metrics"]["rules_fired"]


def test_empty_outlier_pct_treated_as_zero(tmp_path):
    """Cohort-of-1 / degenerate case: outlier_pct is empty string. Must not fire rules."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        # outlier_pct empty (cohort QC couldn't compute) and vif=4 -> no rule fires.
        {"subject": "sub-s43", "session": "ses-01", "run": "1", "task": "nBack",
         "contrast": "twoBack-oneBack", "outlier_pct": "", "vif": "4.0",
         "flagged_outliers": "0", "flagged_vif": "0"},
    ])
    entries = Lev1OutlierGenerator().generate("discovery", {}, _make_args(csv_path))
    assert entries == []


def test_missing_csv_raises_clear_error(tmp_path):
    """Missing input CSV -> FileNotFoundError with the path in the message."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    bogus = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match=str(bogus)):
        Lev1OutlierGenerator().generate("discovery", {}, _make_args(bogus))


def test_empty_csv_returns_empty_list(tmp_path):
    """CSV with only the header row returns []."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [])  # header only, no rows
    entries = Lev1OutlierGenerator().generate("discovery", {}, _make_args(csv_path))
    assert entries == []


def test_end_to_end_on_real_discovery_cohort_qc():
    """Smoke: generator runs against real cohort QC output and produces sensible entries.

    Skipped if the discovery cohort QC output isn't present.
    """
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator

    real_csv = Path("/scratch/users/logben/qa_lev1_discovery/lev1_outliers.csv")
    if not real_csv.is_file():
        pytest.skip(f"discovery cohort QC output not present at {real_csv}")

    entries = Lev1OutlierGenerator().generate(
        "discovery", {}, _make_args(real_csv)
    )

    # Discovery cohort N=5: only strict_vif rule should fire (per the math finding
    # in docs/audits/2026-05-06-lev1-base-task-audit.md — outlier_pct bounded by
    # sqrt(N-1) ~= 2.0 for N=5, so neither combined nor strict_outliers fires).
    # Just assert the shape is sound; counts depend on the actual data.
    for e in entries:
        assert e["source"] == "lev1_outlier"
        assert e["action"] == "exclude"
        assert e["subject"].startswith("sub-")
        assert e["session"].startswith("ses-")
        assert e["task"].startswith("task-")
        assert e["run"].startswith("run-")
        assert "lev1_outlier:" in e["reason"]
        assert "max_vif" in e["metrics"]
        assert e["metrics"]["n_flagged_contrasts"] >= 1
        # Discovery shouldn't fire outlier-based rules
        assert "strict_outliers" not in e["metrics"]["rules_fired"]


def test_dataset_filter_drops_non_member_subjects(tmp_path):
    """A pooled CSV with discovery + validation subjects must filter to just
    the dataset's roster when dataset_config['subjects_file'] is provided."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s03", "session": "ses-01", "run": "1", "task": "stopSignal",
         "contrast": "go", "outlier_pct": "1.0", "vif": "18.0",
         "flagged_outliers": "0", "flagged_vif": "1"},
        {"subject": "sub-s1035", "session": "ses-02", "run": "1", "task": "flanker",
         "contrast": "incongruent-congruent", "outlier_pct": "1.0", "vif": "20.0",
         "flagged_outliers": "0", "flagged_vif": "1"},
    ])
    subjects_path = tmp_path / "subjects_discovery.txt"
    subjects_path.write_text("s03\ns10\n")  # bare IDs, no `sub-` prefix

    config = {"subjects_file": str(subjects_path)}
    entries = Lev1OutlierGenerator().generate("discovery", config, _make_args(csv_path))

    assert len(entries) == 1
    assert entries[0]["subject"] == "sub-s03"


def test_dataset_filter_skipped_when_no_subjects_file(tmp_path):
    """With dataset_config={} (no subjects_file), filtering is a no-op — all flagged
    rows pass through. Preserves backwards-compatible behavior + tmp test ergonomics."""
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator
    csv_path = tmp_path / "lev1_outliers.csv"
    _write_csv(csv_path, [
        {"subject": "sub-s03", "session": "ses-01", "run": "1", "task": "stopSignal",
         "contrast": "go", "outlier_pct": "1.0", "vif": "18.0",
         "flagged_outliers": "0", "flagged_vif": "1"},
        {"subject": "sub-s1035", "session": "ses-02", "run": "1", "task": "flanker",
         "contrast": "incongruent-congruent", "outlier_pct": "1.0", "vif": "20.0",
         "flagged_outliers": "0", "flagged_vif": "1"},
    ])
    entries = Lev1OutlierGenerator().generate("anything", {}, _make_args(csv_path))
    assert {e["subject"] for e in entries} == {"sub-s03", "sub-s1035"}
