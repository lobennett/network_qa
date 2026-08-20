"""Behavioral exclusions from `network_events`' trial-retention metric.

`network_events` truncates a run twice -- at a backward-clock ExpFactory glitch, and at the
end of the acquired scan -- and records what each cost in a sidecar at
`sourcedata/events_qc/<sub>/<ses>/<sub>_<ses>_task-<T>_run-<N>_desc-truncation.json`. It makes
no exclusion decision from those numbers; this generator is the decision half, excluding any
run whose dropped fraction exceeds `nonmonotonic_exclude_fraction`.

NOT IMPLEMENTED: the accuracy / RT / omission criteria this study previously applied. Those lived
in `network_events.qc`, which was removed; the per-task thresholds survive only as
`network_events.qc_globals`. Restoring them means reimplementing the computation.
"""
from __future__ import annotations

import json
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

from network_qa.exclusions.base import load_dataset_subjects, register_generator

# A run losing more than half its test trials to truncation is excluded.
NONMONOTONIC_EXCLUDE_FRACTION = 0.5

_TRUNCATION_JSON_RE = re.compile(
    r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_desc-truncation\.json$"
)


@dataclass(frozen=True)
class Thresholds:
    """Behavioral-generator thresholds."""
    nonmonotonic_exclude_fraction: float = NONMONOTONIC_EXCLUDE_FRACTION


def _scan_nonmonotonic_exclusions(
    bids_dir: Path, threshold: float, subjects: set[str] | None = None,
) -> list[dict]:
    """Scan `sourcedata/events_qc/sub-*/ses-*/*_desc-truncation.json` sidecars
    network_events writes and emit one exclusion entry per run whose
    `FractionTestTrialsDropped` exceeds `threshold`.

    Strict `>`, so a run dropping exactly the threshold fraction is kept. A missing or
    unreadable sidecar counts as 0 dropped rather than raising -- that covers runs whose
    events were never generated. One sidecar is one run, so no aggregation is needed.
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
    description = "Exclude runs whose events truncation dropped too many test trials"

    def add_cli_args(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--nonmonotonic-exclude-fraction",
            type=float,
            default=NONMONOTONIC_EXCLUDE_FRACTION,
            help=("exclude a run whose truncation sidecar reports "
                  "FractionTestTrialsDropped strictly greater than this "
                  f"(default {NONMONOTONIC_EXCLUDE_FRACTION})"),
        )

    def generate(self, dataset_name: str, dataset_config: dict, args: Namespace) -> list[dict]:
        bids_dir = Path(dataset_config["bids_dir"])
        threshold = getattr(args, "nonmonotonic_exclude_fraction", NONMONOTONIC_EXCLUDE_FRACTION)
        entries = _scan_nonmonotonic_exclusions(
            bids_dir, threshold, load_dataset_subjects(dataset_config))
        print(f"Behavioral: {len(entries)} exclusions (trial retention < {1 - threshold:.0%})")
        return entries


register_generator(BehavioralGenerator())
