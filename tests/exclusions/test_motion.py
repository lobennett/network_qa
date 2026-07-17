"""Tests for network_qa.exclusions.motion — behavior change from the monolith.

The monolith's `motion` generator (neuro_workflow/.../exclusions/motion.py)
recomputed FD/DVARS from fmriprep confound TSVs directly. In the modular
pipeline, `motion_qa` already computes those into a `motion_metrics.tsv`; this
generator reads that TSV and applies the same study thresholds (preserving
the monolith's threshold logic at exclusions/motion.py lines ~95-127 and its
output dict shape: subject/session/task/run/source/action/reason[/metrics]).
"""
from argparse import Namespace
from pathlib import Path
import pandas as pd
from network_qa.exclusions.motion import MotionGenerator


def _write_tsv(p, rows):
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)


def test_motion_flags_rest_fd_and_task_prop(tmp_path):
    tsv = tmp_path / "motion_metrics.tsv"
    _write_tsv(tsv, [
        {"subject": "s03", "session": "05", "task": "rest", "run": "1",
         "fmriprep_fd_mean": 0.30, "fmriprep_proportion_fd_over_0.5": 0.0,
         "fmriprep_proportion_std_dvars_over_1.5": 0.0},
        {"subject": "s03", "session": "05", "task": "stroop", "run": "1",
         "fmriprep_fd_mean": 0.05, "fmriprep_proportion_fd_over_0.5": 0.30,
         "fmriprep_proportion_std_dvars_over_1.5": 0.0},
        {"subject": "s03", "session": "05", "task": "nback", "run": "1",
         "fmriprep_fd_mean": 0.05, "fmriprep_proportion_fd_over_0.5": 0.0,
         "fmriprep_proportion_std_dvars_over_1.5": 0.0},
    ])
    gen = MotionGenerator()
    args = Namespace(motion_metrics_tsv=str(tsv), fd_threshold=0.2,
                     proportion_fd_threshold=0.2, proportion_dvars_threshold=0.2)
    out = gen.generate("discovery", {}, args)
    keys = {(e["subject"], e["task"]) for e in out}
    # All four generators emit BIDS-prefixed entities (sub-/ses-/task-/run-),
    # matching the monolith + what is_excluded/lev1 query with. motion_qa's TSV
    # stores bare subject/session, so the generator must add the sub-/ses-
    # prefixes at the source (task/run get task-/run- prefixes too).
    assert keys == {("sub-s03", "task-rest"), ("sub-s03", "task-stroop")}   # nback passes
    assert all(e["subject"] == "sub-s03" and e["session"] == "ses-05" for e in out)
    assert all(e["run"] == "run-1" for e in out)
    assert all(e["action"] == "exclude" and e["source"] == "motion" for e in out)


def test_motion_no_ops_when_tsv_arg_is_none():
    """`--motion-metrics-tsv` is non-required (all generators share one compile
    subparser), so a subset compile without motion inputs leaves it None. The
    generator must no-op (return []) rather than crash on Path(None)."""
    gen = MotionGenerator()
    args = Namespace(motion_metrics_tsv=None, fd_threshold=0.2,
                     proportion_fd_threshold=0.2, proportion_dvars_threshold=0.2)
    assert gen.generate("discovery", {}, args) == []


def test_motion_no_ops_when_tsv_arg_missing():
    """Even if the attribute is entirely absent from args, no crash, no entry."""
    gen = MotionGenerator()
    assert gen.generate("discovery", {}, Namespace()) == []
