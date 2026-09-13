"""Synthetic CLI/API reproductions of silent exclusion loss and cohort leakage."""
import csv
import json
import shutil
import subprocess
from argparse import Namespace

import pytest

from network_qa.cli import _build_parser, main
from network_qa.compile import compile_exclusions, is_excluded, load_lockfile
from network_qa.exclusions import base


def compile_args(tmp_path, generators, *extra):
    return ["compile", "--dataset", "discovery", "--out", str(tmp_path / "lock.json"),
            "--generators", *generators, *map(str, extra)]


def write_iqm(root, suffix="", **overrides):
    path = root / f"sub-s03_ses-01_task-flanker_run-1{suffix}_bold.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fd_mean": 0.1, "fd_perc": 30,
                               "fd_thres": 0.5, **overrides}))
    return path


def write_decisions(tmp_path, rows):
    path = tmp_path / "decisions.tsv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["subject", "session", "task", "run", "action", "reason"])
        writer.writerows(rows)
    return path


def test_unknown_generator_cannot_replace_a_lock(tmp_path):
    out = tmp_path / "lock.json"
    out.write_text("previous lock\n")
    with pytest.raises(ValueError, match="generator"):
        main(compile_args(tmp_path, ["motoin"]))
    assert out.read_text() == "previous lock\n"


def test_explicit_empty_generator_list_runs_nothing():
    lock = compile_exclusions("discovery", {}, Namespace(), generator_names=[])
    assert lock["exclusions"] == []
    assert lock["_meta"]["generators"] == []


@pytest.mark.parametrize("missing", ["argument", "directory"])
def test_selected_motion_requires_its_input(tmp_path, missing):
    extra = [] if missing == "argument" else ["--mriqc-dir", tmp_path / "absent"]
    with pytest.raises(FileNotFoundError, match="MRIQC|mriqc"):
        main(compile_args(tmp_path, ["motion"], *extra))
    assert not (tmp_path / "lock.json").exists()


@pytest.mark.parametrize("value", [None, "bad", float("nan"), -1, 101])
def test_invalid_task_motion_metric_cannot_be_locked(tmp_path, value):
    root = tmp_path / "mriqc"
    write_iqm(root, fd_perc=value)
    with pytest.raises(ValueError, match="fd_perc"):
        main(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    assert not (tmp_path / "lock.json").exists()


@pytest.mark.parametrize("content", ["{broken", "[]"])
def test_unreadable_motion_evidence_cannot_be_locked(tmp_path, content):
    root = tmp_path / "mriqc"
    write_iqm(root).write_text(content)
    with pytest.raises(ValueError):
        main(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    assert not (tmp_path / "lock.json").exists()


@pytest.mark.parametrize("value", [None, 0, float("nan")])
def test_motion_threshold_provenance_cannot_bypass_guard(tmp_path, value):
    root = tmp_path / "mriqc"
    write_iqm(root, provenance={"settings": {"fd_thres": value}})
    with pytest.raises((ValueError, SystemExit), match="fd_thres"):
        main(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    assert not (tmp_path / "lock.json").exists()


def test_duplicate_motion_identity_cannot_overwrite_failed_scan(tmp_path):
    root = tmp_path / "mriqc"
    write_iqm(root / "a", fd_perc=30)
    write_iqm(root / "z", fd_perc=0)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        main(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    assert not (tmp_path / "lock.json").exists()


@pytest.mark.parametrize("suffix", ["_acq-a_run-1_echo-1", "_run-1_echo-2", ""])
def test_subject_decision_expands_acquisition_and_echo_entities(tmp_path, suffix):
    func = tmp_path / "bids" / "sub-s03" / "ses-01" / "func"
    func.mkdir(parents=True)
    (func / f"sub-s03_ses-01_task-flanker{suffix}_bold.nii.gz").touch()
    tsv = write_decisions(tmp_path, [["s03", "-", "-", "-", "exclude", "reviewed"]])
    main(compile_args(tmp_path, ["qa_decisions"], "--bids-dir", tmp_path / "bids",
                      "--decisions-tsv", tsv))
    entries = load_lockfile(tmp_path / "lock.json")
    assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1", entries)
    assert not is_excluded("sub-s03", "ses-02", "task-flanker", "run-1", entries)
    assert not is_excluded("sub-s10", "ses-01", "task-flanker", "run-1", entries)


@pytest.mark.parametrize("session,task,run", [("", "flanker", "1"),
                                               ("-", "flanker", "1"),
                                               ("ses-01", "-", "1")])
def test_partial_scan_identity_does_not_become_subject_exclusion(tmp_path, session, task, run):
    tsv = write_decisions(tmp_path, [["s03", session, task, run, "exclude", "reviewed"]])
    with pytest.raises(ValueError, match="identity"):
        main(compile_args(tmp_path, ["qa_decisions"], "--decisions-tsv", tsv,
                          "--bids-dir", tmp_path))
    assert not (tmp_path / "lock.json").exists()


@pytest.mark.parametrize("subject,session,task,run", [
    ("s03", "01", "flanker", "1"), ("sub-s03", "ses-01", "task-flanker", "run-1")])
def test_conflicting_duplicate_decisions_are_not_order_dependent(tmp_path, subject, session, task, run):
    tsv = write_decisions(tmp_path, [
        ["s03", "01", "flanker", "1", "exclude", "noisy"],
        [subject, session, task, run, "pass", "reviewed"],
    ])
    with pytest.raises(ValueError, match="[Cc]onflicting"):
        main(compile_args(tmp_path, ["qa_decisions"], "--decisions-tsv", tsv))


@pytest.mark.parametrize("generator", ["motion", "behavioral", "lev1_outlier", "qa_decisions"])
@pytest.mark.parametrize("roster", ["empty", "missing", "member"])
def test_explicit_roster_scopes_every_generator(tmp_path, generator, roster):
    root = tmp_path / "mriqc"
    write_iqm(root)
    sidecar = tmp_path / "sourcedata/events_qc/sub-s03/ses-01/sub-s03_ses-01_task-flanker_run-01_desc-truncation.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"FractionTestTrialsDropped": 0.75,
                                  "NTestTrialsExpected": 20, "NTestTrialsRetained": 5}))
    csv_path = tmp_path / "outliers.csv"
    csv_path.write_text("subject,session,task,run,contrast,vif,outlier_pct\n"
                        "sub-s03,ses-01,flanker,1,incongruent,16,0\n")
    tsv = write_decisions(tmp_path, [["s03", "01", "flanker", "01", "exclude", "noisy"]])
    args = _build_parser().parse_args(compile_args(
        tmp_path, [generator], "--mriqc-dir", root, "--lev1-outliers-csv", csv_path,
        "--decisions-tsv", tsv))
    subjects = tmp_path / "subjects.txt"
    if roster != "missing":
        subjects.write_text("s03\n" if roster == "member" else "# empty cohort\n")
    config = {"bids_dir": tmp_path, "subjects_file": subjects}
    if roster == "member":
        entries = compile_exclusions("discovery", config, args, [generator])["exclusions"]
        assert len(entries) == 1
        assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1", entries)
    elif roster == "missing":
        with pytest.raises(FileNotFoundError):
            compile_exclusions("discovery", config, args, [generator])
    else:
        with pytest.raises(ValueError, match="no subjects"):
            compile_exclusions("discovery", config, args, [generator])


@pytest.mark.parametrize("generator,flag", [("lev1_outlier", "--lev1-outliers-csv"),
                                            ("qa_decisions", "--decisions-tsv")])
@pytest.mark.parametrize("content", ["", "unrelated\n"])
def test_headerless_evidence_is_not_a_valid_empty_table(tmp_path, generator, flag, content):
    path = tmp_path / "evidence.tsv"
    path.write_text(content)
    with pytest.raises(ValueError, match="columns"):
        main(compile_args(tmp_path, [generator], flag, path))


@pytest.mark.parametrize("generator,flag", [("motion", "--proportion-fd-threshold"),
    ("behavioral", "--nonmonotonic-exclude-fraction"), ("lev1_outlier", "--strict-vif")])
def test_nan_threshold_cannot_disable_exclusions(tmp_path, generator, flag):
    root = tmp_path / "mriqc"
    write_iqm(root)
    csv_path = tmp_path / "outliers.csv"
    csv_path.write_text("subject,session,task,run,contrast,vif,outlier_pct\n"
                        "sub-s03,ses-01,flanker,1,incongruent,16,0\n")
    with pytest.raises(ValueError, match="finite"):
        main(compile_args(tmp_path, [generator], flag, "nan", "--bids-dir", tmp_path,
                          "--mriqc-dir", root, "--lev1-outliers-csv", csv_path))


def test_installed_package_does_not_claim_enclosing_repository_sha(tmp_path, monkeypatch):
    # A real enclosing repository makes this reproduction independent of where
    # pytest creates its temporary directories.
    if shutil.which("git") is None:
        pytest.skip("Git is required for the enclosing-repository reproduction")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=QA Fixture", "-c", "user.email=qa@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "fixture"],
                   cwd=tmp_path, check=True, capture_output=True)
    wheel_root = tmp_path / ".venv/lib/python3.12"
    wheel_root.mkdir(parents=True)
    monkeypatch.setattr(base, "_REPO_ROOT", wheel_root)
    assert base._git_sha() is None
    monkeypatch.setattr(base, "_REPO_ROOT", tmp_path)
    assert base._git_sha() is not None


@pytest.mark.parametrize("task,metric,value,excluded", [
    ("rest", "fd_mean", 0.2, False), ("rest", "fd_mean", 0.20001, True),
    ("flanker", "fd_perc", 20, False), ("flanker", "fd_perc", 20.001, True)])
def test_motion_strict_boundary_through_cli(tmp_path, task, metric, value, excluded):
    root = tmp_path / "mriqc"
    path = write_iqm(root, **{metric: value})
    path.rename(path.with_name(path.name.replace("flanker", task)))
    main(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    assert is_excluded("sub-s03", "ses-01", f"task-{task}", "run-1",
                       load_lockfile(tmp_path / "lock.json")) is excluded


@pytest.mark.parametrize("metric", ["typo", "truncated-row", "truncated-optional-column",
                                    "blank-identity"])
def test_malformed_outlier_rows_cannot_be_locked(tmp_path, metric):
    path = tmp_path / "outliers.csv"
    required = "subject,session,task,run,contrast,vif,outlier_pct"
    rows = {
        "typo": (required, "sub-s03,ses-01,flanker,1,incongruent,broken,0\n"),
        "truncated-row": (required, "sub-s03,ses-01,flanker,1,incongruent\n"),
        # Truncated past the required columns: the real table also carries
        # flagged_outliers/flagged_vif.
        "truncated-optional-column": (f"{required},flagged_outliers,flagged_vif",
                                      "sub-s03,ses-01,flanker,1,go,18.0,1.0,0\n"),
        "blank-identity": (required, "sub-s03,ses-01,flanker,,incongruent,16,0\n"),
    }
    header, row = rows[metric]
    path.write_text(f"{header}\n{row}")
    with pytest.raises(ValueError):
        main(compile_args(tmp_path, ["lev1_outlier"], "--lev1-outliers-csv", path))
    assert not (tmp_path / "lock.json").exists()


@pytest.mark.parametrize("value,excluded", [("", False), ("NaN", False), ("inf", True),
                                           ("14.9999", False), ("15", True)])
def test_outlier_missing_and_singular_metrics_keep_existing_rules(tmp_path, value, excluded):
    path = tmp_path / "outliers.csv"
    path.write_text("subject,session,task,run,contrast,vif,outlier_pct\n"
                    f"sub-s03,ses-01,flanker,1,incongruent,{value},0\n")
    main(compile_args(tmp_path, ["lev1_outlier"], "--lev1-outliers-csv", path))
    assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1",
                       load_lockfile(tmp_path / "lock.json")) is excluded


def test_missing_behavioral_evidence_keeps_documented_fallback(tmp_path):
    directory = tmp_path / "sourcedata/events_qc/sub-s03/ses-01"
    directory.mkdir(parents=True)
    (directory / "sub-s03_ses-01_task-flanker_run-1_desc-truncation.json").write_text("{")
    (directory / "sub-s03_ses-01_task-flanker_run-2_desc-truncation.json").write_text("{}")
    main(compile_args(tmp_path, ["behavioral"], "--bids-dir", tmp_path))
    assert load_lockfile(tmp_path / "lock.json") == []


@pytest.mark.parametrize("data", [None, {}, {"exclusions": {}}, {"exclusions": [None]}])
def test_malformed_lock_is_not_an_empty_exclusion_set(tmp_path, data):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exclusions"):
        load_lockfile(path)


def test_motion_does_not_validate_unselected_subjects(tmp_path):
    root = tmp_path / "mriqc"
    write_iqm(root)
    unselected = root / "sub-s10_ses-01_task-flanker_run-1_bold.json"
    unselected.write_text(json.dumps({"fd_mean": 0.1, "fd_perc": None, "fd_thres": 0.5}))
    args = _build_parser().parse_args(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    # Bare IDs normalise to the same entity form the lock and roster use.
    lock = compile_exclusions("discovery", {"subjects": {"s03"}}, args, ["motion"])
    assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1", lock["exclusions"])
    assert not is_excluded("sub-s10", "ses-01", "task-flanker", "run-1", lock["exclusions"])


@pytest.mark.parametrize("subjects,roster", [(set(), None), ({"s03"}, "s10\n"),
                                             ({"s03", "s10"}, None)])
def test_motion_subject_selection_cannot_silently_empty_the_cohort(tmp_path, subjects, roster):
    root = tmp_path / "mriqc"
    write_iqm(root)
    args = _build_parser().parse_args(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    config = {"subjects": subjects}
    if roster is not None:
        subjects_file = tmp_path / "subjects.txt"
        subjects_file.write_text(roster)
        config["subjects_file"] = subjects_file
    if subjects == {"s03", "s10"}:
        entries = compile_exclusions("discovery", config, args, ["motion"])["exclusions"]
        assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1", entries)
    else:
        with pytest.raises(ValueError, match="selects no subjects"):
            compile_exclusions("discovery", config, args, ["motion"])


def test_padded_echo_labels_still_score_their_acquisition(tmp_path):
    root = tmp_path / "mriqc"
    root.mkdir()
    for echo in ("01", "02", "03"):
        (root / f"sub-s03_ses-01_task-flanker_run-1_echo-{echo}_bold.json").write_text(
            json.dumps({"fd_mean": 0.1, "fd_perc": 30, "fd_thres": 0.5}))
    main(compile_args(tmp_path, ["motion"], "--mriqc-dir", root))
    entries = load_lockfile(tmp_path / "lock.json")
    assert len(entries) == 1
    assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1", entries)


def test_duplicate_decisions_keep_every_distinct_reason(tmp_path):
    tsv = write_decisions(tmp_path, [
        ["s03", "01", "flanker", "1", "exclude", "motion"],
        ["sub-s03", "ses-01", "task-flanker", "run-01", "exclude", "excessive motion"],
        ["sub-s03", "ses-01", "task-flanker", "run-1", "exclude", "motion"],
    ])
    main(compile_args(tmp_path, ["qa_decisions"], "--decisions-tsv", tsv))
    entries = load_lockfile(tmp_path / "lock.json")
    assert len(entries) == 1
    assert entries[0]["reason"] == "qa_decisions: motion; excessive motion (scan-level)"


@pytest.mark.parametrize("vif,pct,excluded", [(9.999, 10, False), (10, 10, True),
    (10, 9.999, False), (0, 14.999, False), (0, 15, True)])
def test_outlier_combined_and_percentage_boundaries(tmp_path, vif, pct, excluded):
    path = tmp_path / "outliers.csv"
    path.write_text("subject,session,task,run,contrast,vif,outlier_pct\n"
                    f"sub-s03,ses-01,flanker,1,incongruent,{vif},{pct}\n")
    main(compile_args(tmp_path, ["lev1_outlier"], "--lev1-outliers-csv", path))
    assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1",
                       load_lockfile(tmp_path / "lock.json")) is excluded


def test_identical_manual_decisions_and_echoes_compile_to_one_acquisition(tmp_path):
    func = tmp_path / "sub-s03/ses-01/func"
    func.mkdir(parents=True)
    for echo in (1, 2, 3):
        (func / f"sub-s03_ses-01_task-flanker_run-01_echo-{echo}_bold.nii.gz").touch()
    row = ["s03", "-", "-", "-", "exclude", "reviewed"]
    tsv = write_decisions(tmp_path, [row, row])
    main(compile_args(tmp_path, ["qa_decisions"], "--decisions-tsv", tsv,
                      "--bids-dir", tmp_path))
    entries = load_lockfile(tmp_path / "lock.json")
    assert len(entries) == 1
    assert is_excluded("sub-s03", "ses-01", "task-flanker", "run-1", entries)


def test_subject_decision_cannot_expand_into_a_different_subject(tmp_path):
    func = tmp_path / "sub-s03/ses-01/func"
    func.mkdir(parents=True)
    (func / "sub-s10_ses-01_task-flanker_run-1_bold.nii.gz").touch()
    tsv = write_decisions(tmp_path, [["s03", "-", "-", "-", "exclude", "reviewed"]])
    with pytest.raises(ValueError, match="identity"):
        main(compile_args(tmp_path, ["qa_decisions"], "--decisions-tsv", tsv,
                          "--bids-dir", tmp_path))
    assert not (tmp_path / "lock.json").exists()
