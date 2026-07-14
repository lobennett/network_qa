"""Tests for network_fmri.qa_runs — flag short functional runs by per-task mean."""

from network_qa.qa_runs import flag_short_runs


def _verdicts(runs, **kw):
    return {(r["subject"], r["session"], r["task"]): r["verdict"]
            for r in flag_short_runs(runs, **kw)}


def test_drops_below_half_of_per_task_cohort_mean():
    runs = [
        {"subject": "s1", "session": "01", "task": "goNogo", "run": "1", "n_volumes": 150},
        {"subject": "s2", "session": "01", "task": "goNogo", "run": "1", "n_volumes": 150},
        {"subject": "s3", "session": "01", "task": "goNogo", "run": "1", "n_volumes": 10},   # abort
        {"subject": "s3", "session": "01", "task": "goNogo", "run": "2", "n_volumes": 148},  # redo
    ]
    # goNogo mean=(150+150+10+148)/4=114.5, thr=57.25; 10<57.25 -> drop, others keep
    per_run = {(r["subject"], r["session"], r["task"], r["run"]): r["verdict"]
               for r in flag_short_runs(runs)}
    assert per_run[("s3", "01", "goNogo", "1")] == "drop"
    assert per_run[("s3", "01", "goNogo", "2")] == "keep"
    assert per_run[("s1", "01", "goNogo", "1")] == "keep"


def test_near_equal_pair_both_kept():
    # two near-complete runs of the same task -> both above half-mean -> both keep
    runs = [
        {"subject": "s1", "session": "04", "task": "rest", "run": "1", "n_volumes": 100},
        {"subject": "s1", "session": "04", "task": "rest", "run": "2", "n_volumes": 101},
        {"subject": "s2", "session": "04", "task": "rest", "run": "1", "n_volumes": 100},
    ]
    per_run = {(r["run"]): r["verdict"] for r in flag_short_runs(runs)}
    assert per_run["1"] == "keep" and per_run["2"] == "keep"


def test_singleton_task_always_kept():
    runs = [{"subject": "s1", "session": "01", "task": "flanker", "run": "1", "n_volumes": 5}]
    assert flag_short_runs(runs)[0]["verdict"] == "keep"  # mean==itself, 5 >= 0.5*5


def test_reports_task_mean_and_threshold():
    runs = [
        {"subject": "s1", "session": "01", "task": "nBack", "run": "1", "n_volumes": 200},
        {"subject": "s2", "session": "01", "task": "nBack", "run": "1", "n_volumes": 100},
    ]
    out = flag_short_runs(runs, frac=0.5)
    assert out[0]["task_mean"] == 150.0
    assert out[0]["threshold"] == 75.0
