from __future__ import annotations

import subprocess
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

_REGISTRY: dict[str, ExclusionGenerator] = {}


@runtime_checkable
class ExclusionGenerator(Protocol):
    name: str
    description: str

    def add_cli_args(self, parser: ArgumentParser) -> None: ...
    def generate(self, dataset_name: str, dataset_config: dict, args: Namespace) -> list[dict]: ...


def register_generator(generator: ExclusionGenerator) -> None:
    _REGISTRY[generator.name] = generator


def get_generator(name: str) -> ExclusionGenerator | None:
    return _REGISTRY.get(name)


def list_generators() -> dict[str, ExclusionGenerator]:
    return dict(_REGISTRY)


def load_dataset_subjects(dataset_config: dict) -> set[str] | None:
    """Return the dataset's subject IDs (with `sub-` prefix) from `subjects_file`,
    or None if the config has no resolvable subjects file. Bare IDs in the file
    (e.g. `s10`) are normalised to `sub-s10` to match BIDS-prefixed entity IDs.
    """
    raw = dataset_config.get("subjects_file")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        # subjects_file is stored relative to the cwd at registration time.
        # Try cwd first; the user runs CLI from the repo root.
        path = Path.cwd() / raw
    if not path.is_file():
        return None
    subjects: set[str] = set()
    for line in path.read_text().splitlines():
        sid = line.strip()
        if not sid or sid.startswith("#"):
            continue
        subjects.add(sid if sid.startswith("sub-") else f"sub-{sid}")
    return subjects or None


# Repo root resolved from this file's location: src/network_qa/exclusions/base.py
# -> parents[0]=exclusions, parents[1]=network_qa, parents[2]=src, parents[3]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_sha() -> str | None:
    """Return the current git HEAD short SHA, with '+dirty' suffix if the working
    tree has uncommitted changes. Returns None if git is unavailable or this is
    not a git repo.

    Subprocess runs with cwd=_REPO_ROOT so SLURM jobs invoked from scratch dirs
    still resolve the correct repo (otherwise git fails when CWD is outside any
    repo, and code_sha drops to null in production lockfiles).
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=_REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not sha:
        return None
    try:
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
            cwd=_REPO_ROOT,
        ).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sha
    return f"{sha}+dirty" if dirty else sha


def _jsonify(value):
    """Convert non-JSON-native values to JSON-safe forms.

    Walks dicts and lists recursively. Path becomes str. Callables are
    stripped from dicts (argparse Namespaces carry an internal `func`
    callback that's not user-meaningful and not JSON-serializable). Other
    unknown types are stringified with their class name so json.dumps
    doesn't crash on the audit-trail write path.
    """
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        # Skip at dict level; here we land on a top-level callable, which
        # shouldn't happen but stringify defensively.
        return f"<callable:{getattr(value, '__name__', type(value).__name__)}>"
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if callable(v):
                # argparse subparsers attach a `func` callback to every
                # parsed Namespace; drop it from the audit trail.
                continue
            out[k] = _jsonify(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    # Unknown type — stringify so json.dumps doesn't crash.
    return f"<{type(value).__name__}:{value!r}>"


def make_meta(
    generator_name: str,
    args: "Namespace | dict | None",
    n_entries: int,
) -> dict:
    """Build the _meta block for a generator's saved sources file.

    args can be argparse.Namespace, a plain dict, or None (for callers that
    don't have an args object — e.g., cmd_exclusions_import / cmd_events_qc).
    """
    if args is None:
        args_dict = None
    elif hasattr(args, "__dict__") and not isinstance(args, dict):
        args_dict = vars(args)
    else:
        args_dict = dict(args)

    return {
        "generator": generator_name,
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": _git_sha(),
        "args": _jsonify(args_dict) if args_dict is not None else None,
        "n_entries": n_entries,
    }
