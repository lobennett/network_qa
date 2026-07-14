"""QA: flag short functional runs (likely aborted / false starts) by volume count.

Curation keeps ALL runs (run-1/run-2/…) to reflect what was acquired; deciding
which to *use* is this downstream QA step. Rule (PI decision): for each task,
compute the mean number of volumes across the whole cohort; flag any run with
fewer than ``frac`` (default 0.5) of that mean to drop. Clear aborts fall out;
two near-complete runs of a task both survive. Flagged runs feed the processing
selection (``bids-filter-file`` + ``scans.tsv``); a human can override.

Usage:
    nf-qa-runs /path/to/bids            # TSV report to stdout
    nf-qa-runs /path/to/bids --frac 0.5 --out qa_runs.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import mean

_ENTITY_RE = re.compile(
    r"sub-(?P<subject>[A-Za-z0-9]+)_ses-(?P<session>[A-Za-z0-9]+)"
    r"_task-(?P<task>[A-Za-z0-9]+).*?_run-(?P<run>[A-Za-z0-9]+)"
)


def flag_short_runs(runs: list[dict], frac: float = 0.5) -> list[dict]:
    """Annotate each run with its task's cohort mean, the threshold, and a verdict.

    ``runs`` is a list of dicts with at least ``task`` and ``n_volumes``. Returns
    the same rows (copies) plus ``task_mean``, ``threshold``, ``verdict``
    (``keep``/``drop``); a run drops when ``n_volumes < frac * mean(task)``.
    """
    by_task: dict[str, list[int]] = {}
    for r in runs:
        by_task.setdefault(r["task"], []).append(r["n_volumes"])
    task_mean = {t: mean(v) for t, v in by_task.items()}
    out = []
    for r in runs:
        m = task_mean[r["task"]]
        thr = frac * m
        out.append({**r, "task_mean": round(m, 1), "threshold": round(thr, 1),
                    "verdict": "drop" if r["n_volumes"] < thr else "keep"})
    return out


def scan_bold_volumes(bids_root: str | Path) -> list[dict]:
    """Scan a BIDS tree for func ``*_bold.nii.gz`` and return one row per run.

    Volume count is read from the NIfTI header (no data load). Multi-echo runs
    collapse to one row (echoes share a volume count); the first echo is used.
    """
    import nibabel as nib

    root = Path(bids_root)
    seen: dict[tuple, dict] = {}
    for p in sorted(root.glob("sub-*/ses-*/func/*_bold.nii.gz")):
        m = _ENTITY_RE.search(p.name)
        if not m:
            continue
        key = (m["subject"], m["session"], m["task"], m["run"])
        if key in seen:  # already counted this run (another echo)
            continue
        shape = nib.load(str(p)).shape
        seen[key] = {
            "subject": m["subject"], "session": m["session"],
            "task": m["task"], "run": m["run"],
            "n_volumes": int(shape[3]) if len(shape) > 3 else 1,
        }
    return list(seen.values())


def format_report(rows: list[dict]) -> str:
    cols = ["subject", "session", "task", "run", "n_volumes", "task_mean", "threshold", "verdict"]
    lines = ["\t".join(cols)]
    for r in sorted(rows, key=lambda r: (r["subject"], r["session"], r["task"], r["run"])):
        lines.append("\t".join(str(r[c]) for c in cols))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nf-qa-runs", description=__doc__.splitlines()[0])
    ap.add_argument("bids_root", help="BIDS dataset root")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="drop runs with < frac * per-task cohort-mean volumes (default 0.5)")
    ap.add_argument("--out", help="write TSV here (default: stdout)")
    args = ap.parse_args(argv)

    rows = flag_short_runs(scan_bold_volumes(args.bids_root), frac=args.frac)
    report = format_report(rows)
    (Path(args.out).write_text(report + "\n") if args.out else print(report))
    n_drop = sum(r["verdict"] == "drop" for r in rows)
    sys.stderr.write(f"\n{len(rows)} runs, {n_drop} flagged to drop (frac={args.frac}).\n")


if __name__ == "__main__":
    main()
