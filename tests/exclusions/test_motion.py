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
    # NOTE: plan draft asserted bare task names ("rest"/"stroop"); every other
    # generator (lev1_outlier, qa_decisions) and render.py's _bold_relpath
    # expect the BIDS-prefixed "task-<name>" form, which is what the
    # implementation actually emits — asserting the prefixed form here to
    # match that established, codebase-wide convention.
    assert keys == {("s03", "task-rest"), ("s03", "task-stroop")}   # nback passes
    assert all(e["action"] == "exclude" and e["source"] == "motion" for e in out)
