"""Render the compiled exclusion set into the three data-selection channels.

- bids-filter-file: coarse, per-pipeline (config-driven task list + canonical anat),
  or per-subject-session when a pipeline fans out that way (see
  :func:`render_bids_filter_per_session`).
- scans.tsv: per-scan human-readable "why".
- .bidsignore: genuinely-invalid files only (source == "invalid").
Per-scan quality exclusions the filter can't express are enforced downstream at
lev1 from the lockfile.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# The keys babs writes into its own generated filter file
# (babs/templates/bidsapp_run.sh.jinja2). Ours supersedes babs's -- babs emits its
# `--bids-filter-file` before the args from `bids_app_args`, so the later one wins
# by argparse -- so we have to carry its scoping forward, with one deliberate
# difference: babs session-scopes the ANAT keys too, which is wrong for a cohort
# that doesn't acquire anat every session (see `anat_scope`).
_FUNC_KEYS = (
    ("sbref", {"datatype": "func", "suffix": "sbref"}),
    ("fmap", {"datatype": "fmap"}),
)
_ANAT_KEYS = (
    ("flair", {"datatype": "anat", "suffix": "FLAIR"}),
    ("t2w", {"datatype": "anat", "suffix": "T2w"}),
    ("roi", {"datatype": "anat", "suffix": "roi"}),
)


def render_bids_filter(pipeline_cfg: dict) -> dict:
    """Coarse pybids filter: canonical anat acquisition + the task set the pipeline runs."""
    return {
        "t1w": {"acquisition": pipeline_cfg["anat_acquisition"], "suffix": "T1w"},
        "bold": {"task": list(pipeline_cfg["tasks"])},
    }


def _sessions_on_disk(bids_dir) -> list[tuple[str, str]]:
    """Every (sub-*, ses-*) pair present in the tree, sorted."""
    bids_dir = Path(bids_dir)
    return sorted(
        (sub.name, ses.name)
        for sub in bids_dir.glob("sub-*")
        if sub.is_dir()
        for ses in sub.glob("ses-*")
        if ses.is_dir()
    )


def _runs_of_task(bids_dir, sub: str, ses: str, task: str) -> set[str]:
    """Run labels present on disk for one (subject, session, task).

    Globs a single echo so a 3-echo scan counts once. The trailing ``_`` after the
    run label keeps ``run-1`` from matching ``run-10``.
    """
    func = Path(bids_dir) / sub / ses / "func"
    if not func.is_dir():
        return set()
    runs = set()
    for path in func.glob(f"{sub}_{ses}_{task}_run-*_bold.nii.gz"):
        runs.add(path.name.split("_run-")[1].split("_")[0])
    return runs


def render_bids_filter_per_session(
    pipeline_cfg: dict,
    entries: list[dict],
    bids_dir,
    *,
    exclude_sources: tuple[str, ...] = ("short-run",),
    anat_scope: str = "any",
) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    """Per-(subject, session) pybids filters that drop excluded scans.

    A filter dict has no per-task run selectivity, so a task can only be dropped
    from a session when **every** run of it in that session is excluded. Anything
    else would take a good sibling run down with it, so it stays selected and is
    returned in ``residual`` for the caller to report — those scans remain excluded
    downstream at lev1 via the lockfile.

    ``exclude_sources`` picks which exclusion sources withhold a scan from
    *preprocessing*. Default is ``short-run`` only: a truncated 4-D file can crash
    the BIDS app (and under BABS that fails the whole subject-session job), whereas
    a ``behavioral-qc`` exclusion means the events log is defective and the BOLD is
    fine — a lev1 call, not a preprocessing one.

    ``anat_scope`` defaults to ``"any"``: the anat keys carry no ``session``, so a
    session-level job finds the subject's anat wherever it was acquired. Our cohorts
    acquire anat sparsely (7 of 61 discovery sessions have a SagMPRAGE T1w), so
    babs's own session-scoped anat would starve 54/61 jobs. Pass ``"session"`` for a
    cohort that images anat every session.

    Returns ``({(subject, session): filter_dict}, residual_entries)``, keyed with
    BIDS-prefixed labels so callers can build babs's ``${subid}_${sesid}`` filenames.
    """
    all_tasks = list(pipeline_cfg["tasks"])
    anat_acq = pipeline_cfg["anat_acquisition"]

    requested: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        if entry.get("source") not in exclude_sources:
            continue
        requested.setdefault((entry["subject"], entry["session"]), []).append(entry)

    filters: dict[tuple[str, str], dict] = {}
    residual: list[dict] = []
    for sub, ses in _sessions_on_disk(bids_dir):
        bare_ses = _strip(ses, "ses-")
        drop: set[str] = set()
        for entry in requested.get((sub, ses), []):
            task = entry["task"]                      # e.g. task-goNogo
            excluded_runs = {
                _strip(e["run"], "run-")
                for e in requested[(sub, ses)]
                if e["task"] == task
            }
            present = _runs_of_task(bids_dir, sub, ses, task)
            if present and present <= excluded_runs:
                drop.add(_strip(task, "task-"))
            else:
                residual.append(entry)

        f = {
            "bold": {
                "datatype": "func",
                "session": bare_ses,
                "suffix": "bold",
                "task": [t for t in all_tasks if t not in drop],
            },
            "t1w": {
                "datatype": "anat",
                "suffix": "T1w",
                "acquisition": anat_acq,
            },
        }
        for key, base in _FUNC_KEYS:
            f[key] = {**base, "session": bare_ses}
        for key, base in _ANAT_KEYS:
            f[key] = dict(base)
        if anat_scope == "session":
            for key in ("t1w", *(k for k, _ in _ANAT_KEYS)):
                f[key]["session"] = bare_ses
        filters[(sub, ses)] = f

    return filters, residual


def write_bids_filter_per_session(
    pipeline_cfg: dict,
    entries: list[dict],
    bids_dir,
    *,
    pipeline: str,
    out_dir,
    exclude_sources: tuple[str, ...] = ("short-run",),
) -> list[Path]:
    """Write one filter per subject-session as
    ``bids-filter_<pipeline>_<subid>_<sesid>.json``.

    The name is reconstructible inside a babs job from its ``${subid}``/``${sesid}``
    shell variables (which keep their BIDS prefixes), so the participant job can
    point ``--bids-filter-file`` at its own file. Written as plain text in the
    DataLad tree (``text2git``), so jobs read them without ``datalad get``.
    """
    filters, residual = render_bids_filter_per_session(
        pipeline_cfg, entries, bids_dir, exclude_sources=exclude_sources
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (sub, ses), f in sorted(filters.items()):
        path = out_dir / f"bids-filter_{pipeline}_{sub}_{ses}.json"
        path.write_text(json.dumps(f, indent=2, sort_keys=True) + "\n")
        written.append(path)
    for entry in residual:
        print(
            "  note: kept {task} {run} in {subject}/{session} — a sibling run "
            "survives, so the filter can't drop it (still excluded at lev1)".format(**entry)
        )
    return written


def _strip(value: str, prefix: str) -> str:
    """Drop a leading BIDS `<prefix>-` if present, so path reconstruction works
    whether entries are prefixed (the normal case now) or bare."""
    return value[len(prefix):] if value.startswith(prefix) else value


def _bare_entities(e: dict) -> tuple[str, str, str, str]:
    """Return (subject, session, task, run) as bare labels (no BIDS prefixes)."""
    return (
        _strip(e["subject"], "sub-"),
        _strip(e["session"], "ses-"),
        _strip(e["task"], "task-"),
        _strip(e["run"], "run-"),
    )


def _bold_relpath(e: dict) -> str:
    sub, ses, task, run = _bare_entities(e)
    return f"func/sub-{sub}_ses-{ses}_task-{task}_run-{run}_bold.nii.gz"


def _scan_relpaths(e: dict, ses_dir: Path) -> list[str]:
    """Actual on-disk bold file(s) for an exclusion entry, relative to the session
    dir. Globs the func dir so multi-echo (``_echo-1/2/3_``) and any extra entities
    resolve to the REAL filenames — the BIDS validator's scans.tsv check
    (SCANS_FILENAME_NOT_MATCH_DATASET) requires the listed path to exist. The
    trailing ``_`` after the run label keeps ``run-1`` from matching ``run-10``.
    Falls back to the constructed bare name if nothing matches, so a missing file
    still records its ``why``."""
    sub, ses, task, run = _bare_entities(e)
    func = ses_dir / "func"
    pattern = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_*bold.nii.gz"
    matches = sorted(func.glob(pattern)) if func.is_dir() else []
    return [f"func/{m.name}" for m in matches] or [_bold_relpath(e)]


def render_scans_tsv(entries: list[dict], bids_dir) -> list[Path]:
    """Write/refresh a per-session scans.tsv (filename + why) for excluded scans.

    Emits one row per real on-disk bold file, so a multi-echo scan contributes a
    row per echo (each carrying the same ``why``) and the filenames match the
    dataset."""
    bids_dir = Path(bids_dir)
    by_session: dict[tuple, list[dict]] = {}
    for e in entries:
        sub, ses, _, _ = _bare_entities(e)
        by_session.setdefault((sub, ses), []).append(e)
    written = []
    for (sub, ses), rows in by_session.items():
        ses_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}"
        ses_dir.mkdir(parents=True, exist_ok=True)
        out = ses_dir / f"sub-{sub}_ses-{ses}_scans.tsv"
        with out.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["filename", "why"])
            for e in rows:
                why = e.get("reason", "")
                for relpath in _scan_relpaths(e, ses_dir):
                    w.writerow([relpath, why])
        written.append(out)
    return written


def render_bidsignore(entries: list[dict], out_path) -> list[str]:
    """.bidsignore holds ONLY genuinely-invalid files (source == 'invalid')."""
    lines = []
    for e in entries:
        if e.get("source") != "invalid":
            continue
        sub, ses, _, _ = _bare_entities(e)
        lines.append(_bold_relpath(e).replace("func/", f"sub-{sub}/ses-{ses}/func/"))
    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return lines
