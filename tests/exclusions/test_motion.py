"""Tests for network_qa.exclusions.motion — MRIQC IQMs as the motion source.

`fd_perc` counts frames above whatever `--fd_thres` MRIQC ran with, so the generator must
verify that threshold rather than assume it. Everything else keeps the shared generator
contract: BIDS-prefixed entities, one entry per excluded acquisition.
"""
import json
from argparse import Namespace

import pytest

from network_qa.exclusions.motion import MotionGenerator, inspect_motion
from network_qa.functional import FunctionalEvidence
from network_qa.manifest import functional_key


def iqm(path, *, fd_mean=0.05, fd_perc=0.0, dvars_std=1.0, fd_thres=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fd_mean": fd_mean, "fd_perc": fd_perc, "dvars_std": dvars_std,
        "provenance": {"settings": {"fd_thres": fd_thres}},
    }))


def args(**kw):
    base = dict(mriqc_dir=None, fd_threshold=0.2, proportion_fd_threshold=0.2,
                expect_fd_thres=0.5)
    return Namespace(**{**base, **kw})


def func(tmp_path, sub="sub-s03", ses="ses-05"):
    return tmp_path / sub / ses / "func"


def functional(
    task, *, representative_echo=2, observed_echoes=(1, 2, 3),
    subject="sub-s03", session="ses-05", run="1", flags=(),
):
    """Build reviewed functional evidence without needing a NIfTI fixture."""
    return FunctionalEvidence(
        key=functional_key(subject, session, task, run),
        expected_echoes=(1, 2, 3),
        observed_echoes=observed_echoes,
        missing_echoes=tuple(echo for echo in (1, 2, 3) if echo not in observed_echoes),
        representative_echo=representative_echo,
        tr_count=100,
        original_tr_count=107,
        expected_tr_count_mean=107.0,
        tr_count_fraction=1.0,
        flags=flags,
    )


def inspect_fixture(task, fd_mean, fd_perc, tmp_path):
    evidence = functional(task)
    path = func(tmp_path) / (
        f"sub-s03_ses-05_task-{task}_run-1_echo-2_bold.json"
    )
    iqm(path, fd_mean=fd_mean, fd_perc=fd_perc)
    motion, = inspect_motion([evidence], tmp_path)
    return motion


def report(tmp_path, task="nBack", run="1"):
    """Create the root-level, acquisition-level MRIQC BOLD report."""
    path = tmp_path / f"sub-s03_ses-05_task-{task}_run-{run}_bold.html"
    path.write_text("MRIQC report")
    return path


@pytest.mark.parametrize("task,fd_mean,fd_perc,flagged", [
    ("rest", 0.2, 0.0, True),
    ("rest", 0.1, 20.0, False),
    ("nBack", 0.2, 0.0, True),
    ("nBack", 0.1, 20.0, True),
    ("nBack", 0.199, 19.9, False),
])
def test_motion_boundaries(task, fd_mean, fd_perc, flagged, tmp_path):
    evidence = inspect_fixture(task, fd_mean, fd_perc, tmp_path)
    assert ("excessive_motion" in evidence.flags) is flagged


def test_inspection_uses_echo_two_and_records_its_report(tmp_path):
    evidence = functional("nBack")
    echo_one = func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-1_bold.json"
    echo_two = func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json"
    iqm(echo_one, fd_mean=0.1, fd_perc=50.0)
    iqm(echo_two, fd_mean=0.1, fd_perc=0.0)
    actual_report = report(tmp_path)

    motion, = inspect_motion([evidence], tmp_path)

    assert motion.fd_perc == 0.0
    assert "excessive_motion" not in motion.flags
    assert motion.report_path == actual_report
    assert "missing_report" not in motion.flags


def test_missing_or_ambiguous_root_level_mriqc_report_is_review_evidence(tmp_path):
    missing = functional("nBack", run="1")
    ambiguous = functional("nBack", run="2")
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json")
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-2_echo-2_bold.json")
    report(tmp_path, run="2")
    report(tmp_path, run="02")

    missing_motion, ambiguous_motion = inspect_motion([missing, ambiguous], tmp_path)

    assert missing_motion.report_path is None
    assert "missing_report" in missing_motion.flags
    assert ambiguous_motion.report_path is None
    assert "ambiguous_report" in ambiguous_motion.flags


def test_multi_echo_without_echo_two_is_review_evidence_not_echo_one_fallback(tmp_path):
    evidence = functional("nBack", representative_echo=None, observed_echoes=(1, 3))
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-1_bold.json", fd_perc=50.0)

    motion, = inspect_motion([evidence], tmp_path)

    assert motion.fd_perc is None
    assert "missing_echo_2" in motion.flags
    assert "excessive_motion" not in motion.flags


@pytest.mark.parametrize(
    "representative_echo,flags",
    [
        (None, ()),
        (1, ()),
        (2, ("unequal_echo_counts",)),
        (2, ("invalid_nifti",)),
        (2, ("ambiguous_echo",)),
    ],
)
@pytest.mark.parametrize("observed_echoes", [(2,), (1, 2), (2, 3), (1, 2, 3)])
def test_untrusted_observed_echo_two_is_not_scored(representative_echo, flags, observed_echoes, tmp_path):
    evidence = functional(
        "nBack", representative_echo=representative_echo, flags=flags, observed_echoes=observed_echoes,
    )
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json", fd_perc=50.0)

    motion, = inspect_motion([evidence], tmp_path)

    assert motion.fd_perc is None
    assert "untrusted_echo_2" in motion.flags
    assert "missing_echo_2" not in motion.flags
    assert "excessive_motion" not in motion.flags


@pytest.mark.parametrize("observed", [(1,), (2,), (3,), (1, 2), (2, 3), (1, 3)])
def test_incomplete_study_echo_groups_only_use_observed_trusted_echo_two(tmp_path, observed):
    representative = 2 if 2 in observed else (observed[0] if len(observed) == 1 else None)
    evidence = functional("nBack", representative_echo=representative, observed_echoes=observed,
                          flags=("missing_echo",))
    # Even available MRIQC output cannot establish that a missing BIDS echo exists.
    for echo in (1, 2, 3):
        iqm(func(tmp_path) / f"sub-s03_ses-05_task-nBack_run-1_echo-{echo}_bold.json", fd_perc=20.0)

    motion, = inspect_motion([evidence], tmp_path)

    if 2 in observed:
        assert motion.fd_perc == 20.0
        assert "excessive_motion" in motion.flags
    else:
        assert (motion.fd_mean, motion.fd_perc, motion.dvars_std, motion.fd_thres) == (None,) * 4
        assert "missing_echo_2" in motion.flags
        assert "excessive_motion" not in motion.flags
    assert evidence.tr_count == 100


@pytest.mark.parametrize("kind", ["empty", "directory", "broken_symlink", "unreadable", "valid_symlink"])
def test_motion_report_validity_requires_readable_nonempty_file(tmp_path, kind):
    evidence = functional("nBack")
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json")
    path = tmp_path / "sub-s03_ses-05_task-nBack_run-1_bold.html"
    if kind == "directory":
        path.mkdir()
    elif kind in {"broken_symlink", "valid_symlink"}:
        target = tmp_path / "annex-content"
        if kind == "valid_symlink":
            target.write_text("report content")
        path.symlink_to(target)
    else:
        path.write_text("" if kind == "empty" else "report content")
        if kind == "unreadable":
            path.chmod(0)
    try:
        motion, = inspect_motion([evidence], tmp_path)
    finally:
        if kind == "unreadable":
            path.chmod(0o644)
    assert motion.fd_mean == .05
    if kind == "valid_symlink":
        assert motion.report_path == path and not motion.flags
    else:
        assert motion.report_path is None and "invalid_report" in motion.flags


@pytest.mark.parametrize("metric", ["fd_mean", "fd_perc", "dvars_std", "fd_thres"])
@pytest.mark.parametrize("value", [-1, True, "0.1", [], {}, None, float("nan"), float("inf"), 10 ** 400],
                         ids=["negative", "boolean", "string", "list", "object", "missing", "nan", "inf", "huge"])
def test_invalid_motion_metrics_withhold_all_values(tmp_path, metric, value):
    evidence = functional("nBack")
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json", **{metric: value})
    motion, = inspect_motion([evidence], tmp_path)
    assert "malformed_iqm" in motion.flags
    assert (motion.fd_mean, motion.fd_perc, motion.dvars_std, motion.fd_thres) == (None,) * 4
    assert "excessive_motion" not in motion.flags


@pytest.mark.parametrize("payload", [b"\xff", b"{", b"[]", b'{"fd_mean":' + b"9" * 5000 + b"}"],
                         ids=["invalid_utf8", "invalid_json", "array", "integer_digit_limit"])
def test_unreadable_or_unparseable_motion_iqm_is_malformed_evidence(tmp_path, payload):
    evidence = functional("nBack")
    path = func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    motion, = inspect_motion([evidence], tmp_path)
    assert "malformed_iqm" in motion.flags
    assert (motion.fd_mean, motion.fd_perc, motion.dvars_std, motion.fd_thres) == (None,) * 4


@pytest.mark.parametrize("fd_perc,malformed", [(0, False), (100, False), (100.1, True)])
def test_motion_metric_domains_include_zero_and_percent_endpoints(tmp_path, fd_perc, malformed):
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json",
        fd_mean=0, fd_perc=fd_perc, dvars_std=0)
    motion, = inspect_motion([functional("nBack")], tmp_path)
    assert ("malformed_iqm" in motion.flags) is malformed
    if malformed:
        assert (motion.fd_mean, motion.fd_perc, motion.dvars_std, motion.fd_thres) == (None,) * 4
    else:
        assert (motion.fd_mean, motion.fd_perc, motion.dvars_std, motion.fd_thres) == (0, fd_perc, 0, .5)


@pytest.mark.parametrize("fd_thres", [0.2, 0.5000000001])
def test_missing_iqm_and_mismatched_threshold_require_review(fd_thres, tmp_path):
    missing, mismatch = functional("nBack", run="1"), functional("nBack", run="2")
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-2_echo-2_bold.json", fd_thres=fd_thres)

    missing_motion, mismatched_motion = inspect_motion([missing, mismatch], tmp_path)

    assert "missing_iqm" in missing_motion.flags
    assert "fd_thres_mismatch" in mismatched_motion.flags
    assert "excessive_motion" not in mismatched_motion.flags


def test_malformed_iqm_identity_mismatch_and_dvars_remain_explicit(tmp_path):
    malformed = functional("nBack", run="1")
    mismatch = functional("nBack", run="2")
    dvars_only = functional("nBack", run="3")
    broken = func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_echo-2_bold.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{")
    wrong_parent = tmp_path / "sub-s99" / "ses-05" / "func" / (
        "sub-s03_ses-05_task-nBack_run-2_echo-2_bold.json"
    )
    iqm(wrong_parent)
    iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-3_echo-2_bold.json", dvars_std=99.0)

    malformed_motion, mismatch_motion, dvars_motion = inspect_motion(
        [malformed, mismatch, dvars_only], tmp_path,
    )

    assert "malformed_iqm" in malformed_motion.flags
    assert "identity_mismatch" in mismatch_motion.flags
    assert dvars_motion.dvars_std == 99.0
    assert "excessive_motion" not in dvars_motion.flags


class TestCriteria:
    def test_rest_excluded_on_fd_mean(self, tmp_path):
        iqm(func(tmp_path) / "sub-s03_ses-05_task-rest_run-1_bold.json", fd_mean=0.30)
        out = MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path)))
        assert len(out) == 1
        e = out[0]
        assert (e["subject"], e["session"], e["task"], e["run"]) == \
            ("sub-s03", "ses-05", "task-rest", "run-1")
        assert e["source"] == "motion" and e["action"] == "exclude"
        assert "fd_mean" in e["reason"]

    def test_task_excluded_on_fd_perc(self, tmp_path):
        # 30% of frames over 0.5 mm, against a 20% cutoff.
        iqm(func(tmp_path) / "sub-s03_ses-05_task-stopSignal_run-1_bold.json", fd_perc=30.0)
        out = MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path)))
        assert len(out) == 1 and out[0]["task"] == "task-stopSignal"

    def test_clean_run_passes(self, tmp_path):
        iqm(func(tmp_path) / "sub-s03_ses-05_task-nBack_run-1_bold.json",
            fd_mean=0.05, fd_perc=3.0)
        assert MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path))) == []

    def test_rest_criterion_does_not_apply_to_task(self, tmp_path):
        """A task run with high mean FD but few spikes is kept -- different criterion."""
        iqm(func(tmp_path) / "sub-s03_ses-05_task-flanker_run-1_bold.json",
            fd_mean=0.30, fd_perc=1.0)
        assert MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path))) == []

    def test_dvars_never_excludes_but_is_recorded(self, tmp_path):
        """No DVARS criterion: measured on both cohorts, it added nothing FD missed."""
        iqm(func(tmp_path) / "sub-s03_ses-05_task-rest_run-1_bold.json",
            fd_mean=0.30, dvars_std=99.0)
        out = MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path)))
        assert len(out) == 1                          # excluded on FD alone
        assert "dvars" not in out[0]["reason"]
        assert out[0]["metrics"]["dvars_std"] == 99.0  # kept as evidence


class TestThresholdGuard:
    def test_mismatched_fd_thres_refuses(self, tmp_path):
        """IQMs from a 0.2 run must not be scored against the 0.5 criterion."""
        iqm(func(tmp_path) / "sub-s03_ses-05_task-flanker_run-1_bold.json",
            fd_perc=30.0, fd_thres=0.2)
        with pytest.raises(SystemExit, match="fd_thres"):
            MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path)))

    def test_matching_threshold_is_accepted(self, tmp_path):
        iqm(func(tmp_path) / "sub-s03_ses-05_task-flanker_run-1_bold.json",
            fd_perc=30.0, fd_thres=0.2)
        out = MotionGenerator().generate(
            "discovery", {}, args(mriqc_dir=str(tmp_path), expect_fd_thres=0.2))
        assert len(out) == 1


class TestMultiEcho:
    def test_one_entry_per_acquisition(self, tmp_path):
        """Head motion is shared across echoes, so echo-1 stands for the acquisition."""
        for e in (1, 2, 3):
            iqm(func(tmp_path) / f"sub-s03_ses-05_task-flanker_run-1_echo-{e}_bold.json",
                fd_perc=30.0)
        out = MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path)))
        assert len(out) == 1 and out[0]["run"] == "run-1"


class TestScoping:
    def test_vcs_administration_iqm_is_not_a_motion_acquisition(self, tmp_path):
        iqm(
            tmp_path / ".git" / "annex" / "objects" /
            "sub-s03_ses-05_task-rest_run-1_bold.json",
            fd_mean=0.9,
        )

        assert MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path))) == []

    def test_selected_generator_requires_mriqc_dir(self):
        with pytest.raises(FileNotFoundError, match="mriqc-dir"):
            MotionGenerator().generate("discovery", {}, args())

    def test_absent_attribute_has_clear_error(self):
        with pytest.raises(FileNotFoundError, match="mriqc-dir"):
            MotionGenerator().generate("discovery", {}, Namespace())

    def test_missing_dir_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope"):
            MotionGenerator().generate(
                "discovery", {}, args(mriqc_dir=str(tmp_path / "nope")))

    def test_dataset_subjects_filter(self, tmp_path):
        for sub in ("sub-s03", "sub-s99"):
            iqm(func(tmp_path, sub) / f"{sub}_ses-05_task-rest_run-1_bold.json", fd_mean=0.9)
        out = MotionGenerator().generate(
            "discovery", {"subjects": {"sub-s03"}}, args(mriqc_dir=str(tmp_path)))
        assert [e["subject"] for e in out] == ["sub-s03"]
