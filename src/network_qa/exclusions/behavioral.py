"""Behavioral exclusion generator — runs behavioral QC and produces exclusion entries.

Also applies the monolith's non-monotonic-onset-truncation exclusion rule (see
`_scan_nonmonotonic_exclusions`): `network_events` truncates a run at the first
backward-clock ExpFactory glitch and writes the resulting trial-retention
metric as a truncation-QC sidecar at
`sourcedata/events_qc/<sub>/<ses>/<sub>_<ses>_task-<task>_run-<run>_desc-truncation.json`
(`NTestTrialsExpected` / `NTestTrialsRetained` / `FractionTestTrialsDropped`);
`network_events` makes no exclusion decision from that number. The sidecar lives
under `sourcedata/` with a non-reserved `_desc-truncation` name (not as an
`_events.json` in `func/`, which BIDS reserves for events-column descriptions
and bids-validator would reject). This generator reads those sidecars and
excludes any run whose dropped fraction exceeds `nonmonotonic_exclude_fraction`
— the decision half of that split.
"""
from __future__ import annotations

import json
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

from network_qa.exclusions.base import load_dataset_subjects, register_generator

# Matches neuro_workflow.events.qc.NONMONOTONIC_EXCLUDE_FRACTION exactly.
NONMONOTONIC_EXCLUDE_FRACTION = 0.5

_TRUNCATION_JSON_RE = re.compile(
    r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_desc-truncation\.json$"
)


@dataclass(frozen=True)
class Thresholds:
    """Behavioral-generator thresholds. Defaults match the monolith."""
    nonmonotonic_exclude_fraction: float = NONMONOTONIC_EXCLUDE_FRACTION


def _scan_nonmonotonic_exclusions(
    bids_dir: Path, threshold: float, subjects: set[str] | None = None,
) -> list[dict]:
    """Scan `sourcedata/events_qc/sub-*/ses-*/*_desc-truncation.json` sidecars
    network_events writes and emit one exclusion entry per run whose
    `FractionTestTrialsDropped` exceeds `threshold`.

    Matches neuro_workflow.events.qc.run_qc's non-monotonic-truncation rule
    exactly: strict `>` comparison (a run dropping EXACTLY the threshold
    fraction is kept, not excluded), source="behavioral-qc", action="exclude".
    A missing/unreadable sidecar (run whose events were never generated, or an
    events package predating this metric) is treated as 0 dropped -- no
    crash, no entry. One sidecar = one run, so no further aggregation is
    needed: this already emits one entry per (subject, session, task, run).
    """
    entries: list[dict] = []
    for sidecar in sorted(bids_dir.glob("sourcedata/events_qc/sub-*/ses-*/*_desc-truncation.json")):
        m = _TRUNCATION_JSON_RE.match(sidecar.name)
        if not m:
            continue
        subject = m.group("subject")
        if subjects is not None and subject not in subjects:
            continue
        try:
            sidecar_data = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        frac = sidecar_data.get("FractionTestTrialsDropped")
        if frac is None or not (frac > threshold):
            continue

        expected = sidecar_data.get("NTestTrialsExpected", 0)
        retained = sidecar_data.get("NTestTrialsRetained", 0)
        dropped = expected - retained
        entries.append({
            "subject": subject,
            "session": m.group("session"),
            "task": f"task-{m.group('task')}",
            "run": f"run-{m.group('run')}",
            "action": "exclude",
            "source": "behavioral-qc",
            "reason": (
                "non-monotonic onset truncation drops "
                f"{dropped}/{expected} test trials (>{int(threshold * 100)}%)"
            ),
            "metrics": {
                "NTestTrialsExpected": expected,
                "NTestTrialsRetained": retained,
                "FractionTestTrialsDropped": frac,
            },
        })
    return entries


class BehavioralGenerator:
    name = "behavioral"
    description = (
        "Generate exclusions from behavioral QC (accuracy, RT, omission thresholds) "
        "+ non-monotonic-truncation trial-retention"
    )

    def add_cli_args(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--behavioral-dir",
            required=False,
            default=None,
            help="Path to sourcedata behavioral directory (default: {bids_dir}/sourcedata)",
        )
        parser.add_argument(
            "--nonmonotonic-exclude-fraction",
            type=float,
            default=NONMONOTONIC_EXCLUDE_FRACTION,
            help=(
                "Exclude a run if network_events' truncation-QC sidecar "
                "(sourcedata/events_qc/.../_desc-truncation.json) reports "
                "FractionTestTrialsDropped strictly greater than this "
                f"(default {NONMONOTONIC_EXCLUDE_FRACTION})."
            ),
        )

    def generate(self, dataset_name: str, dataset_config: dict, args: Namespace) -> list[dict]:
        bids_dir = Path(dataset_config["bids_dir"])
        behavioral_dir = Path(args.behavioral_dir) if getattr(args, "behavioral_dir", None) else bids_dir / "sourcedata"

        try:
            from network_events.qc import run_qc
        except ImportError:
            print("Error: pandas required for behavioral generator. Install with: uv pip install -e '.[events]'")
            exclusion_entries, trim_entries = [], []
        else:
            exclusion_entries, trim_entries = run_qc(
                behavioral_dir=behavioral_dir,
                bids_dir=bids_dir,
            )
            # Source field is set by the exclusions system when saving, but include for clarity
            for entry in exclusion_entries:
                entry["source"] = "behavioral-qc"

        threshold = getattr(args, "nonmonotonic_exclude_fraction", NONMONOTONIC_EXCLUDE_FRACTION)
        subjects = load_dataset_subjects(dataset_config)
        nonmonotonic_entries = _scan_nonmonotonic_exclusions(bids_dir, threshold, subjects)
        exclusion_entries.extend(nonmonotonic_entries)

        print(
            f"Behavioral QC: {len(exclusion_entries)} exclusions, {len(trim_entries)} trim entries "
            f"({len(nonmonotonic_entries)} non-monotonic-truncation)"
        )
        return exclusion_entries


register_generator(BehavioralGenerator())
