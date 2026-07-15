"""Short-run exclusion generator — flag functional runs too short to model.

A prematurely-ended / aborted BOLD acquisition produces a run with far fewer
volumes than the task normally has. This generator scans a BIDS tree's func
NIfTIs (header-only — no data load, no BIDS modification), computes a per-task
expected length as the statistical MODE of dim4 across that task's runs, and
excludes any run whose dim4 is strictly less than `fraction` (default 0.5) of
that mode. It is the network_qa port of the monolith's `scripts/check_tr.sh`
heuristic + `qa_runs.py`'s multi-echo dedup, expressed as an ExclusionGenerator.

The mode (not mean) is used so a handful of aborted runs cannot drag the
expected length down: shorts are rare and never shift the mode, so every task's
runs are judged against the length a complete run of that task actually has.
"""
from __future__ import annotations

import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path

from network_qa.exclusions.base import load_dataset_subjects, register_generator

# Default cutoff: exclude runs shorter than 50% of the per-task mode. Strict `<`
# (a run at exactly the fraction is kept) matches the monolith's check_tr.sh /
# qa_runs.py convention.
SHORT_RUN_FRACTION = 0.5

# Parse BIDS `key-value` entities from a filename. Multi-echo runs carry an
# `echo-` entity that we intentionally ignore so all echoes of a run collapse
# to one logical (subject, session, task, run).
_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z]+)-(?P<value>[a-zA-Z0-9]+)")


def _parse_entities(name: str) -> dict[str, str]:
    return {m.group("key"): m.group("value") for m in _ENTITY_RE.finditer(name)}


def _mode(values: list[int]) -> int:
    """Return the most common value; ties broken by first-seen (Counter order)."""
    return Counter(values).most_common(1)[0][0]


def _scan_short_runs(
    bids_dir: Path, fraction: float, subjects: set[str] | None = None,
) -> list[dict]:
    """Scan `sub-*/ses-*/func/*_bold.nii.gz` and emit one exclusion entry per run
    whose volume count is strictly below `fraction * <per-task mode>`.

    Multi-echo runs collapse to one logical run (echoes share a volume count;
    the first-seen echo is read). The per-task mode is computed from ALL that
    task's runs in this dataset, including the short ones — shorts are rare and
    do not shift the mode. A missing directory or an unreadable/corrupt NIfTI
    is skipped: no crash, no entry.
    """
    import nibabel as nib

    bids_dir = Path(bids_dir)
    # dim4 per logical run, keyed by BIDS-prefixed (subject, session, task, run).
    runs: dict[tuple[str, str, str, str], int] = {}
    for path in sorted(bids_dir.glob("sub-*/ses-*/func/*_bold.nii.gz")):
        ents = _parse_entities(path.name)
        if not all(k in ents for k in ("sub", "ses", "task")):
            continue
        subject = f"sub-{ents['sub']}"
        if subjects is not None and subject not in subjects:
            continue
        key = (
            subject,
            f"ses-{ents['ses']}",
            f"task-{ents['task']}",
            f"run-{ents.get('run', '1')}",
        )
        if key in runs:  # another echo of a run we've already counted
            continue
        try:
            shape = nib.load(str(path)).shape
        except Exception:
            # Unreadable / corrupt / non-NIfTI file — skip, don't crash.
            continue
        if len(shape) < 4:
            continue
        runs[key] = int(shape[3])

    # Per-task expected length = mode of dim4 across all that task's runs.
    by_task: dict[str, list[int]] = {}
    for (_subject, _session, task, _run), dim4 in runs.items():
        by_task.setdefault(task, []).append(dim4)
    task_mode = {task: _mode(vols) for task, vols in by_task.items()}

    threshold_pct = fraction * 100
    entries: list[dict] = []
    for (subject, session, task, run), dim4 in sorted(runs.items()):
        mode = task_mode[task]
        if not (dim4 < fraction * mode):  # strict `<`: run at the cutoff is kept
            continue
        pct = round(100 * dim4 / mode)
        entries.append({
            "subject": subject,
            "session": session,
            "task": task,
            "run": run,
            "source": "short-run",
            "action": "exclude",
            "reason": (
                f"{dim4}/{mode} TRs ({pct}% of task mode, "
                f"<{threshold_pct:.0f}% cutoff)"
            ),
            "metrics": {"dim4": dim4, "expected": mode, "fraction": dim4 / mode},
        })
    return entries


class ShortRunGenerator:
    name = "short_run"
    description = (
        "Exclude functional runs too short to model (dim4 < fraction * per-task "
        "mode; aborted / prematurely-ended scans)"
    )

    def add_cli_args(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--short-run-fraction",
            type=float,
            default=SHORT_RUN_FRACTION,
            help=(
                "Exclude a run whose volume count is strictly less than this "
                f"fraction of its task's mode dim4 (default {SHORT_RUN_FRACTION})."
            ),
        )

    def generate(self, dataset_name: str, dataset_config: dict, args: Namespace) -> list[dict]:
        bids_dir = Path(dataset_config["bids_dir"])
        fraction = getattr(args, "short_run_fraction", SHORT_RUN_FRACTION)
        subjects = load_dataset_subjects(dataset_config)
        entries = _scan_short_runs(bids_dir, fraction, subjects)
        print(f"Short-run: {len(entries)} exclusions (fraction={fraction})")
        return entries


register_generator(ShortRunGenerator())
