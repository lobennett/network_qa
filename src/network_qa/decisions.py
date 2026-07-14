"""Load user QC decisions from a sidecar TSV."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
    Raises ValueError on invalid action values.
    """
    if not path.is_file():
        return {}

    out: dict[ScanKey | str, Decision] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("subject"):
                continue
            action = row["action"].strip()
            if action not in _VALID_ACTIONS:
                raise ValueError(
                    f"invalid action {action!r} in {path}; "
                    f"valid: {sorted(_VALID_ACTIONS)}"
                )
            decision = Decision(action=action, reason=row.get("reason", "").strip())
            session = row.get("session", "-").strip()
            if session == "-" or not session:
                out[row["subject"]] = decision
            else:
                key = ScanKey(
                    subject=row["subject"],
                    session=session,
                    task=row["task"].strip(),
                    run=row["run"].strip(),
                )
                out[key] = decision
    return out
