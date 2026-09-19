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

from network_qa.manifest import AcquisitionKey
from network_qa.exclusions.base import load_dataset_subjects, register_generator, run_entity, validate_number

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
    validate_number(threshold, "nonmonotonic_exclude_fraction", maximum=1)
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
            "run": run_entity(m.group('run')),
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


# The manifest reader is independent of the historical lockfile exclusion policy.
@dataclass(frozen=True)
class BehavioralEvidence:
    key: AcquisitionKey
    behavioral_status: str
    event_status: str
    truncation_status: str
    truncation_metrics: dict
    paths: tuple[str, ...]
    flags: tuple[str, ...]
    exception: dict | None = None
    conversion_error: dict | None = None


def _evidence_key(values):
    run = values['run']
    if not run.isascii() or not run.isdigit():
        raise ValueError('non-numeric run')
    return AcquisitionKey('acquisition', values['subject'], values['session'], 'func',
                          'bold', task=values['task'], acquisition=values.get('acq', ''),
                          direction=values.get('dir', ''), run=str(int(run)))


def _evidence_table(path, columns):
    import csv
    result = {}
    try:
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t', strict=True)
            if not set(columns) <= set(reader.fieldnames or ()):
                return {}, False
            for raw in reader:
                if None in raw or any(not (raw.get(name) or '').strip() for name in columns):
                    return {}, False
                values = {name: (value or '').strip() for name, value in raw.items()}
                key = _evidence_key(values)
                if key in result:
                    return {}, False
                result[key] = values
    except (OSError, UnicodeError, csv.Error, ValueError, TypeError):
        return {}, False
    return result, True


def _evidence_paths(root, pattern, suffix):
    """Index exact BIDS identities; never borrow another acquisition's evidence."""
    result = {}
    for path in sorted(root.glob(pattern)):
        try:
            parts = [part.split('-', 1) for part in path.name.removesuffix(suffix).split('_')]
            values = dict(parts)
            if len(values) != len(parts) or set(values) - {'sub', 'ses', 'task', 'run', 'acq', 'dir'}:
                continue
            values.update(subject='sub-' + values.pop('sub'), session='ses-' + values.pop('ses'))
            key = _evidence_key(values)
            if path.parent.name in {'beh', 'func'}:
                actual = path.parents[2].name, path.parents[1].name
            else:
                actual = path.parents[1].name, path.parent.name
            if actual != (key.subject, key.session):
                continue
            result.setdefault(key, []).append(path)
        except (KeyError, ValueError):
            continue
    return result


def _read_table_file(paths, *, events=False):
    import csv
    import math
    if len(paths) != 1:
        return False
    try:
        with paths[0].open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t' if events else ',', strict=True)
            if not reader.fieldnames or (events and not {'onset', 'duration'} <= set(reader.fieldnames)):
                return False
            count = 0
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    return False
                if events and any(not math.isfinite(float(row[name])) for name in ('onset', 'duration')):
                    return False
                count += 1
            return count > 0
    except (OSError, UnicodeError, csv.Error, ValueError, TypeError):
        return False


def _read_truncation(paths):
    import math
    if len(paths) != 1:
        return None
    try:
        data = json.loads(paths[0].read_text(encoding='utf-8'))
        names = ('NTestTrialsExpected', 'NTestTrialsRetained', 'FractionTestTrialsDropped',
                 'ScanDurationSeconds', 'NScanTestTrialsDropped', 'FractionScanTestTrialsDropped')
        if not isinstance(data, dict):
            return None
        for name in names:
            value = data.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return None
            if name.startswith('N') and int(value) != value:
                return None
            if name.startswith('Fraction') and value > 1:
                return None
        expected, retained = data['NTestTrialsExpected'], data['NTestTrialsRetained']
        if retained > expected or data['NScanTestTrialsDropped'] > retained:
            return None
        for count, total, fraction in ((expected - retained, expected, data['FractionTestTrialsDropped']),
                                       (data['NScanTestTrialsDropped'], retained, data['FractionScanTestTrialsDropped'])):
            if not math.isclose(fraction, count / total if total else 0, abs_tol=1e-8):
                return None
        return {name: data[name] for name in names}
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def behavioral_evidence(bids_dir: Path, functionals) -> tuple[BehavioralEvidence, ...]:
    """Read canonical evidence without applying an automatic behavioral drop rule.

    Rest and reviewed absence exceptions need no events or truncation files. Other
    missing/invalid inputs remain unknown; positive trial losses require review.
    """
    behavior_root = bids_dir / 'sourcedata/behavioral'
    qc_root = bids_dir / 'sourcedata/events_qc'
    exception_path = behavior_root / 'behavioral_exceptions.tsv'
    errors_path = qc_root / 'conversion_errors.tsv'
    exceptions, exceptions_valid = _evidence_table(
        exception_path, ('subject', 'session', 'task', 'run', 'reason', 'detail'))
    errors, errors_valid = _evidence_table(
        errors_path, ('subject', 'session', 'task', 'run', 'source_path', 'exception_class', 'message'))
    behavior = _evidence_paths(behavior_root, 'sub-*/ses-*/beh/*_beh.csv', '_beh.csv')
    events = _evidence_paths(bids_dir, 'sub-*/ses-*/func/*_events.tsv', '_events.tsv')
    truncation = _evidence_paths(qc_root, 'sub-*/ses-*/*_desc-truncation.json', '_desc-truncation.json')
    evidence = []
    for functional in functionals:
        key = functional.key
        if key.task == 'rest':
            evidence.append(BehavioralEvidence(key, 'not_applicable', 'not_applicable',
                                               'not_applicable', {}, (), ()))
            continue
        raw, event, sidecar = behavior.get(key, []), events.get(key, []), truncation.get(key, [])
        exception, error = exceptions.get(key), errors.get(key)
        paths = tuple(str(path.relative_to(bids_dir)) for path in
                      [exception_path, errors_path, *raw, *event, *sidecar])
        flags = []
        behavioral_status = 'unknown'
        if not exceptions_valid or (exception and raw):
            flags.append('behavioral_evidence_unknown')
        elif exception:
            behavioral_status = 'reviewed_exception'
        elif _read_table_file(raw):
            behavioral_status = 'available'
        else:
            flags.append('behavioral_evidence_unknown')
        if not errors_valid:
            flags.append('event_conversion_unknown')
        if error:
            event_status = 'failed'
            flags.append('event_conversion_failed')
        elif behavioral_status == 'reviewed_exception' and not event:
            event_status = 'not_applicable'
        elif _read_table_file(event, events=True):
            event_status = 'available' if errors_valid else 'unknown'
            if behavioral_status == 'reviewed_exception':
                flags.append('behavioral_exception_conflict')
        else:
            event_status = 'unknown'
            flags.append('events_unknown')
        metrics = _read_truncation(sidecar)
        if behavioral_status == 'reviewed_exception' and not sidecar:
            truncation_status = 'not_applicable'
        elif metrics is None:
            truncation_status = 'unknown'
            flags.append('truncation_unknown')
        else:
            truncation_status = 'available'
            if metrics['FractionTestTrialsDropped'] > 0:
                flags.append('nonmonotonic_trial_loss')
            if metrics['FractionScanTestTrialsDropped'] > 0:
                flags.append('scan_length_trial_loss')
        evidence.append(BehavioralEvidence(key, behavioral_status, event_status,
            truncation_status, metrics or {}, paths, tuple(sorted(set(flags))), exception, error))
    return tuple(evidence)
