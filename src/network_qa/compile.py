"""Compile registered exclusion generators into one provenance-stamped lockfile.

network_qa's compile is a leaner, in-memory reimplementation of the monolith
pipeline's `core/exclusions.py` compile step: generators run directly (no sources/*.json
disk cache, no manual force-include/force-exclude overrides file -- that
persistence + override layer isn't part of this plan's scope). The output is
a single self-contained lockfile `{"_meta": ..., "exclusions": [...]}` rather
than the monolith's two-file split (a meta-only `<dataset>_lock.json` pointing
at a separate bare-list `compiled_exclusions.json`). See
tests/exclusions/test_compile.py for the full rationale + verification notes.
"""
from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from network_qa.exclusions.base import _git_sha, get_generator, list_generators

_KEY = ("subject", "session", "task", "run", "source")


def compile_exclusions(dataset_name, dataset_config, args, generator_names=None) -> dict:
    """Run each named (or all registered) generator, merge + dedupe, wrap with _meta."""
    names = generator_names or list(list_generators())
    seen, merged = set(), []
    for name in names:
        gen = get_generator(name)
        if gen is None:
            continue
        for entry in gen.generate(dataset_name, dataset_config, args):
            k = tuple(entry.get(f) for f in _KEY)
            if k in seen:
                continue
            seen.add(k)
            merged.append(entry)
    return {
        "_meta": {
            "dataset": dataset_name,
            "compiled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_sha": _git_sha(),
            "generators": names,
            "n_exclusions": len(merged),
        },
        "exclusions": merged,
    }


def write_lockfile(lockfile: dict, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lockfile, indent=2, sort_keys=True) + "\n")
    return out_path


def load_lockfile(path) -> list[dict]:
    """Read exclusions from a lockfile; accepts wrapped {_meta,exclusions} or bare list."""
    data = json.loads(Path(path).read_text())
    return data["exclusions"] if isinstance(data, dict) else data


def _scan_key(entry: dict) -> tuple:
    return (entry["subject"], entry["session"], entry["task"], entry["run"])


def is_excluded(subject: str, session: str, task: str, run: str,
                exclusions: list[dict]) -> bool:
    """Return True if the given scan is excluded in the compiled exclusions list.

    Consumer-side query helper for network_glm/lev1 (pass an already-loaded
    exclusions list, e.g. from `load_lockfile`). Matches the monolith's
    core/exclusions.py::is_excluded exactly: an EXACT tuple match on
    (subject, session, task, run) against each entry, considering only entries
    whose action is in {"exclude", "trim"}. No entity normalization — the
    lockfile stores BIDS-prefixed entities and the caller queries with the same
    prefixed form (normalization is the caller's responsibility).

    Uses `e.get("action")` (a defensive superset of the monolith's `e["action"]`)
    so an entry missing the field is skipped rather than raising KeyError.
    """
    key = (subject, session, task, run)
    return any(
        _scan_key(e) == key for e in exclusions if e.get("action") in ("exclude", "trim")
    )
