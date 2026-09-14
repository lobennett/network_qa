"""Compile registered exclusion generators into one provenance-stamped lockfile.

Generators run in memory and their output merges into a single self-contained
`{"_meta": ..., "exclusions": [...]}` file. There is no on-disk per-source cache and no
force-include/force-exclude override layer: a compile is cheap enough to just re-run, and
manual overrides go through the `qa_decisions` generator instead.
"""
from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from network_qa.exclusions.base import (
    code_sha, get_generator, list_generators, load_dataset_subjects,
)

_KEY = ("subject", "session", "task", "run", "source")


def compile_exclusions(dataset_name, dataset_config, args, generator_names=None) -> dict:
    """Run each named (or all registered) generator, merge + dedupe, wrap with _meta."""
    names = list(list_generators()) if generator_names is None else list(generator_names)
    unknown = set(names) - set(list_generators())
    if unknown:
        raise ValueError(f"Unknown exclusion generators: {sorted(unknown)}")
    # Invalid cohort selectors must not produce a lock, even with no generators.
    load_dataset_subjects(dataset_config)
    seen, merged = set(), []
    for name in names:
        gen = get_generator(name)
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
            "code_sha": code_sha(),
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
    entries = data.get("exclusions") if isinstance(data, dict) else data
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError(f"{path}: exclusions must be a list of objects")
    return entries


def _scan_key(entry: dict) -> tuple:
    return (entry["subject"], entry["session"], entry["task"], entry["run"])


def is_excluded(subject: str, session: str, task: str, run: str,
                exclusions: list[dict]) -> bool:
    """Return True if the given scan is excluded in the compiled exclusions list.

    Consumer-side query helper for network_glm/lev1 (pass an already-loaded exclusions
    list, e.g. from `load_lockfile`). Exact tuple match on (subject, session, task, run),
    considering only entries whose action is in {"exclude", "trim"}. No entity
    normalization at query time: compiled entries have BIDS-prefixed entities and
    unpadded numeric runs (run-1, not run-01), matching GLM's keys. The caller must
    query with the same form. An entry missing `action` is skipped, not an error.
    """
    key = (subject, session, task, run)
    return any(
        _scan_key(e) == key for e in exclusions if e.get("action") in ("exclude", "trim")
    )
