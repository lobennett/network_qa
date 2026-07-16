"""Tests for network_qa.compile — run registered generators -> merge/dedupe ->
provenance-stamped lockfile.

Lockfile shape verified against neuro_workflow's `core/exclusions.py` (the
downstream reader for the OLD monolith). That module actually splits the
concept across two files: a meta-only `data/exclusions/<ds>_lock.json`
(dataset/compiled_at/compiled_at_code_sha/compiled_path/n_total_entries/
n_overrides/sources -- no embedded entries) that POINTS AT a separate
bare-list `compiled_exclusions.json` holding the real per-scan entries
(read by `load_compiled_exclusions` / consumed by lev1's `--exclusions-file`).
There is no single self-contained `{_meta, exclusions}` file in the monolith.

network_qa consolidates this into ONE self-contained wrapped lockfile
(`{"_meta": ..., "exclusions": [...]}`) since there's no disk-based
sources/overrides layer here (Task 6 runs generators in-memory). This keeps
the same *read contract* the monolith already uses elsewhere (its own
`_read_source_file` tolerates both a wrapped `{_meta, entries}` shape and a
bare list) -- `load_lockfile` here mirrors that tolerance for `{_meta,
exclusions}` vs. bare list, so any future consumer (network_glm) can read
either shape without knowing which one was written.

The dedup key (subject, session, task, run, source) is a network_qa-native
safety net not present in the monolith (which never dedupes -- it just
concatenates every sources/*.json). It's non-breaking: the monolith's own
`is_excluded`/`get_trim_info` only care about (subject, session, task, run)
having *any* matching exclude/trim entry, and duplicate identical entries
from the same generator have never been a real production requirement.
"""
from argparse import Namespace
from network_qa import compile as nqc
from network_qa.exclusions.base import register_generator


class _FakeGen:
    name = "fake"; description = "test"
    def add_cli_args(self, p): pass
    def generate(self, ds, cfg, args):
        return [{"subject": "s03", "session": "05", "task": "task-rest",
                 "run": "run-1", "source": "fake", "action": "exclude", "reason": "x"}]


def test_compile_wraps_with_meta_and_dedupes(tmp_path):
    register_generator(_FakeGen())
    lock = nqc.compile_exclusions("discovery", {}, Namespace(), generator_names=["fake"])
    assert "_meta" in lock and "exclusions" in lock
    assert lock["_meta"]["dataset"] == "discovery"
    assert len(lock["exclusions"]) == 1
    out = tmp_path / "lock.json"
    nqc.write_lockfile(lock, out)
    assert out.is_file()
    import json
    assert json.loads(out.read_text())["exclusions"][0]["source"] == "fake"


class _DupeGen:
    name = "dupe"; description = "emits an exact duplicate entry twice"
    def add_cli_args(self, p): pass
    def generate(self, ds, cfg, args):
        entry = {"subject": "s10", "session": "01", "task": "task-flanker",
                  "run": "run-1", "source": "dupe", "action": "exclude", "reason": "y"}
        return [entry, dict(entry)]


def test_compile_dedupes_identical_entries_from_same_source(tmp_path):
    register_generator(_DupeGen())
    lock = nqc.compile_exclusions("discovery", {}, Namespace(), generator_names=["dupe"])
    assert len(lock["exclusions"]) == 1


def test_compile_merges_multiple_generators(tmp_path):
    register_generator(_FakeGen())
    register_generator(_DupeGen())
    lock = nqc.compile_exclusions(
        "discovery", {}, Namespace(), generator_names=["fake", "dupe"],
    )
    sources = {e["source"] for e in lock["exclusions"]}
    assert sources == {"fake", "dupe"}
    assert lock["_meta"]["generators"] == ["fake", "dupe"]
    assert lock["_meta"]["n_exclusions"] == len(lock["exclusions"]) == 2


def test_load_lockfile_accepts_wrapped_format(tmp_path):
    lock = {"_meta": {"dataset": "discovery"}, "exclusions": [
        {"subject": "s03", "session": "05", "task": "task-rest", "run": "run-1",
         "source": "fake", "action": "exclude", "reason": "x"},
    ]}
    out = tmp_path / "lock.json"
    nqc.write_lockfile(lock, out)
    loaded = nqc.load_lockfile(out)
    assert loaded == lock["exclusions"]


def test_load_lockfile_accepts_bare_list_back_compat(tmp_path):
    """A bare list (no _meta wrapper) is also readable -- mirrors the
    monolith's own tolerant `_read_source_file` behavior."""
    import json
    bare = [
        {"subject": "s10", "session": "02", "task": "task-cuedTS", "run": "run-1",
         "source": "qa_decisions", "action": "exclude", "reason": "z"},
    ]
    out = tmp_path / "bare_lock.json"
    out.write_text(json.dumps(bare))
    loaded = nqc.load_lockfile(out)
    assert loaded == bare


def test_compile_with_qa_decisions_generator_flows_through(tmp_path):
    """Integration coverage for the real qa_decisions generator (the monolith
    test this replaces originally drove neuro_workflow's core.exclusions
    disk-persistence layer; network_qa's compile is in-memory)."""
    import csv
    from network_qa.exclusions.qa_decisions import QADecisionsGenerator

    register_generator(QADecisionsGenerator())

    tsv = tmp_path / "decisions.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["subject", "session", "task", "run", "action", "reason"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerow({"subject": "sub-s03", "session": "ses-02", "task": "task-cuedTS",
                    "run": "run-1", "action": "exclude", "reason": "noisy"})

    args = Namespace(decisions_tsv=tsv)
    lock = nqc.compile_exclusions(
        "discovery", {}, args, generator_names=["qa_decisions"],
    )
    compiled = lock["exclusions"]
    assert len(compiled) == 1
    e = compiled[0]
    assert e["source"] == "qa_decisions"
    assert e["subject"] == "sub-s03"
    assert e["task"] == "task-cuedTS"
    assert e["action"] == "exclude"
    assert "noisy" in e["reason"]


def test_compile_with_behavioral_nonmonotonic_generator_flows_through(tmp_path):
    """Integration coverage for the behavioral generator's non-monotonic
    trial-retention rule: a truncation-QC sidecar (sourcedata/events_qc/...)
    over threshold flows through compile into the wrapped, provenance-stamped
    lockfile."""
    import json
    from network_qa.exclusions.behavioral import BehavioralGenerator

    register_generator(BehavioralGenerator())

    sidecar_dir = tmp_path / "sourcedata" / "events_qc" / "sub-s03" / "ses-02"
    sidecar_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "sub-s03_ses-02_task-cuedTS_run-1_desc-truncation.json"
    sidecar.write_text(json.dumps({
        "NTestTrialsExpected": 20,
        "NTestTrialsRetained": 5,
        "FractionTestTrialsDropped": 0.75,
    }))

    args = Namespace(nonmonotonic_exclude_fraction=0.5)
    lock = nqc.compile_exclusions(
        "discovery", {"bids_dir": str(tmp_path)}, args, generator_names=["behavioral"],
    )
    compiled = lock["exclusions"]
    assert len(compiled) == 1
    e = compiled[0]
    assert e["subject"] == "sub-s03"
    assert e["session"] == "ses-02"
    assert e["task"] == "task-cuedTS"
    assert e["run"] == "run-1"
    assert e["source"] == "behavioral-qc"
    assert e["action"] == "exclude"
    assert "15/20" in e["reason"]
    assert lock["_meta"]["n_exclusions"] == 1


def test_compile_with_lev1_outlier_generator_flows_through(tmp_path):
    """Integration coverage for the real lev1_outlier generator (see note
    on test_compile_with_qa_decisions_generator_flows_through)."""
    import csv
    from network_qa.exclusions.lev1_outlier import Lev1OutlierGenerator

    register_generator(Lev1OutlierGenerator())

    csv_path = tmp_path / "lev1_outliers.csv"
    fieldnames = ["subject", "session", "run", "task", "contrast",
                  "outlier_pct", "vif", "flagged_outliers", "flagged_vif"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"subject": "sub-s03", "session": "ses-02", "run": "1", "task": "cuedTS",
                    "contrast": "response_time", "outlier_pct": "2.0", "vif": "18.09",
                    "flagged_outliers": "0", "flagged_vif": "1"})

    args = Namespace(
        lev1_outliers_csv=csv_path,
        combined_vif=10.0, combined_outlier_pct=10.0,
        strict_vif=15.0, strict_outlier_pct=15.0,
    )
    lock = nqc.compile_exclusions(
        "discovery", {}, args, generator_names=["lev1_outlier"],
    )
    compiled = lock["exclusions"]
    assert len(compiled) == 1
    assert compiled[0]["source"] == "lev1_outlier"
    assert compiled[0]["subject"] == "sub-s03"
    assert compiled[0]["task"] == "task-cuedTS"
    assert compiled[0]["action"] == "exclude"
