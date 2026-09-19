"""Traverse public evidence without entering version-control administration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


_VCS_ADMINISTRATION = frozenset({
    '.git', '.hg', '.svn', '.bzr', '.jj', '.pijul', '_darcs',
    'CVS', 'RCS', 'SCCS', '.fossil-settings',
})


def is_vcs_administration_path(relative: Path) -> bool:
    """Use the same exclusion for filesystem inventory and committed Git paths."""
    return bool(_VCS_ADMINISTRATION.intersection(relative.parts))


def reject_directory_symlink(path: Path) -> None:
    if path.is_symlink() and path.is_dir():
        raise ValueError(f'evidence directory symlink is not supported: {path}')


def _raise_walk_error(error: OSError) -> None:
    raise error


def iter_evidence_files(root: Path) -> Iterator[Path]:
    """Prune administration before descent; retain public file symlinks as paths.

    Opening a public annex symlink still reads its target's content. The internal
    object storage is never inventoried as an additional acquisition/evidence path.
    """
    reject_directory_symlink(root)
    if root.name in _VCS_ADMINISTRATION or not root.is_dir():
        return
    for directory, subdirectories, files in os.walk(root, onerror=_raise_walk_error, followlinks=False):
        subdirectories[:] = [name for name in subdirectories if name not in _VCS_ADMINISTRATION]
        for name in subdirectories:
            reject_directory_symlink(Path(directory) / name)
        for name in files:
            if name in _VCS_ADMINISTRATION:
                continue
            path = Path(directory) / name
            if path.is_file() or path.is_symlink():
                yield path
