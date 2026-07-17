"""Render the compiled exclusion set into the three data-selection channels.

- bids-filter-file: coarse, per-pipeline (config-driven task list + canonical anat).
- scans.tsv: per-scan human-readable "why".
- .bidsignore: genuinely-invalid files only (source == "invalid").
Surgical per-scan quality exclusions are enforced downstream at lev1 from the lockfile.
"""
from __future__ import annotations

import csv
from pathlib import Path


def render_bids_filter(pipeline_cfg: dict) -> dict:
    """Coarse pybids filter: canonical anat acquisition + the task set the pipeline runs."""
    return {
        "t1w": {"acquisition": pipeline_cfg["anat_acquisition"], "suffix": "T1w"},
        "bold": {"task": list(pipeline_cfg["tasks"])},
    }


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
