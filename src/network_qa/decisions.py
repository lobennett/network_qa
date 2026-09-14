"""Load user QC decisions from a sidecar TSV."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from network_qa.exclusions.base import run_entity


Action = Literal["pass", "exclude", "review"]
_VALID_ACTIONS = {"pass", "exclude", "review"}


@dataclass(frozen=True)
class ScanKey:
    subject: str
    session: str
    task: str
    run: str


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


def load_decisions(path: Path) -> dict[ScanKey | str, Decision]:
    """Read a QC decisions TSV.

    Schema (tab-separated; reason is optional):
        subject  session  task  run  action  reason

    Subject-level decisions use "-" for session/task/run; the key in the
    returned dict is the subject string. Scan-level decisions use a
    `ScanKey` as the dict key. Keys preserve the first row's spelling.

    Returns an empty dict if the file does not exist.
    Raises ValueError on malformed rows, or when rows sharing one canonical scan
    identity disagree on the action. Duplicates that agree on the action are one
    decision whose reason joins their distinct nonempty reasons in first-seen
    file order, separated by '; '.
    """
    if not path.is_file():
        return {}

    out: dict[ScanKey | str, Decision] = {}
    seen: dict[tuple, tuple[ScanKey | str, list[str]]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"subject", "session", "task", "run", "action"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            if None in row or any(row.get(field) is None for field in required):
                raise ValueError(f"Malformed decision row in {path}:{reader.line_num}")
            row = {field: (value or "").strip() for field, value in row.items()}
            action = row["action"]
            if action not in _VALID_ACTIONS:
                raise ValueError(
                    f"invalid action {action!r} in {path}; "
                    f"valid: {sorted(_VALID_ACTIONS)}"
                )
            reason = row.get("reason", "")
            identity = [row[field] for field in ("subject", "session", "task", "run")]
            subject_level = identity[1:] == ["-", "-", "-"]
            canonical = []
            for value, prefix in zip(identity, ("sub", "ses", "task", "run")):
                if subject_level and prefix != "sub":
                    canonical.append("-")
                    continue
                label = value.removeprefix(f"{prefix}-")
                if not re.fullmatch(r"[A-Za-z0-9]+", label):
                    raise ValueError(f"Invalid decision identity in {path}:{reader.line_num}: {identity}")
                canonical.append(run_entity(label) if prefix == "run" else f"{prefix}-{label}")
            canonical_key = tuple(canonical)
            if canonical_key in seen:
                key, reasons = seen[canonical_key]
                if out[key].action != action:
                    raise ValueError(f"Conflicting decisions for {canonical_key} in {path}:{reader.line_num}")
                if reason and reason not in reasons:
                    reasons.append(reason)
                    out[key] = Decision(action=action, reason="; ".join(reasons))
                continue
            if subject_level:
                key = row["subject"]
            else:
                key = ScanKey(
                    subject=row["subject"],
                    session=row["session"],
                    task=row["task"],
                    run=row["run"],
                )
            seen[canonical_key] = (key, [reason] if reason else [])
            out[key] = Decision(action=action, reason=reason)
    return out
