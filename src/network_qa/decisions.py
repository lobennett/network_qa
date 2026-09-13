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

    Schema (tab-separated):
        subject  session  task  run  action  reason

    Subject-level decisions use "-" for session/task/run; the key in the
    returned dict is the subject string. Scan-level decisions use a
    `ScanKey` as the dict key.

    Returns an empty dict if the file does not exist.
    Raises ValueError on malformed rows or conflicting duplicate decisions.
    Only explicit '-' in all three scan fields denotes a subject decision.
    """
    if not path.is_file():
        return {}

    out: dict[ScanKey | str, Decision] = {}
    seen: dict[tuple, Decision] = {}
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
            decision = Decision(action=action, reason=row.get("reason", ""))
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
                if seen[canonical_key] != decision:
                    raise ValueError(f"Conflicting decisions for {canonical_key} in {path}:{reader.line_num}")
                continue
            seen[canonical_key] = decision
            if subject_level:
                out[row["subject"]] = decision
            else:
                key = ScanKey(
                    subject=row["subject"],
                    session=row["session"],
                    task=row["task"],
                    run=row["run"],
                )
                out[key] = decision
    return out
