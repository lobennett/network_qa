"""Tests for network_qa.exclusions.motion — MRIQC IQMs as the motion source.

`fd_perc` counts frames above whatever `--fd_thres` MRIQC ran with, so the generator must
verify that threshold rather than assume it. Everything else keeps the shared generator
contract: BIDS-prefixed entities, one entry per excluded acquisition.
"""
import json
from argparse import Namespace

import pytest

from network_qa.exclusions.motion import MotionGenerator


def iqm(path, *, fd_mean=0.05, fd_perc=0.0, dvars_std=1.0, fd_thres=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fd_mean": fd_mean, "fd_perc": fd_perc, "dvars_std": dvars_std,
        "provenance": {"settings": {"fd_thres": fd_thres}},
    }))


def args(**kw):
    base = dict(mriqc_dir=None, fd_threshold=0.2, proportion_fd_threshold=0.2,
                expect_fd_thres=0.5, dvars_std_threshold=None)
    return Namespace(**{**base, **kw})


def func(tmp_path, sub="sub-s03", ses="ses-05"):
    return tmp_path / sub / ses / "func"


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

    def test_dvars_is_off_by_default(self, tmp_path):
        iqm(func(tmp_path) / "sub-s03_ses-05_task-flanker_run-1_bold.json", dvars_std=99.0)
        assert MotionGenerator().generate("discovery", {}, args(mriqc_dir=str(tmp_path))) == []
        out = MotionGenerator().generate(
            "discovery", {}, args(mriqc_dir=str(tmp_path), dvars_std_threshold=1.5))
        assert len(out) == 1 and "dvars_std" in out[0]["reason"]


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
    def test_no_mriqc_dir_is_a_noop(self):
        """--mriqc-dir is non-required: a subset compile leaves it None."""
        assert MotionGenerator().generate("discovery", {}, args()) == []

    def test_absent_attribute_is_a_noop(self):
        assert MotionGenerator().generate("discovery", {}, Namespace()) == []

    def test_missing_dir_is_a_noop(self, tmp_path):
        out = MotionGenerator().generate(
            "discovery", {}, args(mriqc_dir=str(tmp_path / "nope")))
        assert out == []

    def test_dataset_subjects_filter(self, tmp_path):
        for sub in ("sub-s03", "sub-s99"):
            iqm(func(tmp_path, sub) / f"{sub}_ses-05_task-rest_run-1_bold.json", fd_mean=0.9)
        out = MotionGenerator().generate(
            "discovery", {"subjects": {"sub-s03"}}, args(mriqc_dir=str(tmp_path)))
        assert [e["subject"] for e in out] == ["sub-s03"]
