"""Functional echo-completeness and scan-length evidence."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

from network_qa.manifest import AcquisitionKey


EXPECTED_ECHOES = (1, 2, 3)
N_DUMMY = 7


@dataclass(frozen=True)
class FunctionalEvidence:
    """Echo and volume-count evidence for one logical BOLD acquisition."""

    key: AcquisitionKey
    expected_echoes: tuple[int, ...]
    observed_echoes: tuple[int, ...]
    missing_echoes: tuple[int, ...]
    representative_echo: int | None
    tr_count: int | None
    original_tr_count: int | None
    expected_tr_count_mean: float | None
    tr_count_fraction: float | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class _ObservedImage:
    echo: int
    tr_count: int


def inspect_functionals(bids_dir: Path) -> tuple[FunctionalEvidence, ...]:
    """Group BOLD echoes, read dim4, and calculate task-level expected lengths."""
    grouped: dict[AcquisitionKey, list[_ObservedImage]] = defaultdict(list)
    for path in sorted(bids_dir.glob("sub-*/ses-*/func/*_bold.nii.gz")):
        key, echo = _functional_identity(path)
        grouped[key].append(_ObservedImage(echo=echo, tr_count=_tr_count(path)))

    preliminary = tuple(_build_evidence(key, images) for key, images in sorted(grouped.items()))
    expected_means = _task_count_means(preliminary)
    return tuple(_with_scan_length_flags(evidence, expected_means) for evidence in preliminary)


def _functional_identity(path: Path) -> tuple[AcquisitionKey, int]:
    filename = path.name.removesuffix(".nii.gz")
    parts = filename.split("_")
    if not parts or parts[-1] != "bold":
        raise ValueError(f"not a BOLD NIfTI: {path}")

    entities: dict[str, str] = {}
    for part in parts[:-1]:
        name, separator, value = part.partition("-")
        if not separator or not value:
            raise ValueError(f"invalid BIDS entity in {path}")
        if name in entities:
            raise ValueError(f"duplicate BIDS entity {name!r} in {path}")
        entities[name] = value

    try:
        key = AcquisitionKey(
            record_type="acquisition",
            subject=f"sub-{entities['sub']}",
            session=f"ses-{entities['ses']}",
            datatype="func",
            suffix="bold",
            task=entities["task"],
            acquisition=entities.get("acq", ""),
            direction=entities.get("dir", ""),
            run=_canonical_index(entities["run"], "run", path),
        )
    except KeyError as error:
        raise ValueError(f"missing BIDS entity {error.args[0]!r} in {path}") from error
    echo = _canonical_index(entities.get("echo", "1"), "echo", path)
    return key, int(echo)


def _canonical_index(value: str, entity: str, path: Path) -> str:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"invalid {entity} entity in {path}: {value!r}")
    return value.lstrip("0") or "0"


def _tr_count(path: Path) -> int:
    shape = nib.load(str(path)).shape
    return shape[3] if len(shape) > 3 else 1


def _build_evidence(key: AcquisitionKey, images: list[_ObservedImage]) -> FunctionalEvidence:
    counts_by_echo = {image.echo: image.tr_count for image in images}
    if len(counts_by_echo) != len(images):
        raise ValueError(f"duplicate echo images for {key}")
    observed_echoes = tuple(sorted(counts_by_echo))
    missing_echoes = tuple(echo for echo in EXPECTED_ECHOES if echo not in counts_by_echo)
    counts_agree = len(set(counts_by_echo.values())) == 1
    representative_echo, tr_count = _representative_count(counts_by_echo, counts_agree)
    flags = []
    if missing_echoes:
        flags.append("missing_echo")
    if not counts_agree:
        flags.append("unequal_echo_counts")
    original_tr_count = tr_count + N_DUMMY if tr_count is not None else None
    return FunctionalEvidence(
        key=key,
        expected_echoes=EXPECTED_ECHOES,
        observed_echoes=observed_echoes,
        missing_echoes=missing_echoes,
        representative_echo=representative_echo,
        tr_count=tr_count,
        original_tr_count=original_tr_count,
        expected_tr_count_mean=None,
        tr_count_fraction=None,
        flags=tuple(flags),
    )


def _representative_count(
    counts_by_echo: dict[int, int], counts_agree: bool
) -> tuple[int | None, int | None]:
    if not counts_agree:
        return None, None
    if 2 in counts_by_echo:
        return 2, counts_by_echo[2]
    if len(counts_by_echo) == 1:
        echo, count = next(iter(counts_by_echo.items()))
        return echo, count
    return None, next(iter(counts_by_echo.values()))


def _task_count_means(rows: tuple[FunctionalEvidence, ...]) -> dict[str, float]:
    counts_by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.original_tr_count is not None:
            counts_by_task[row.key.task].append(row.original_tr_count)
    return {
        task: sum(counts) / len(counts)
        for task, counts in counts_by_task.items()
    }


def _with_scan_length_flags(
    evidence: FunctionalEvidence, expected_means: dict[str, float]
) -> FunctionalEvidence:
    if evidence.original_tr_count is None:
        return evidence
    expected_mean = expected_means[evidence.key.task]
    flags = evidence.flags
    if evidence.original_tr_count < 0.5 * expected_mean:
        flags = (*flags, "short_scan")
    return FunctionalEvidence(
        key=evidence.key,
        expected_echoes=evidence.expected_echoes,
        observed_echoes=evidence.observed_echoes,
        missing_echoes=evidence.missing_echoes,
        representative_echo=evidence.representative_echo,
        tr_count=evidence.tr_count,
        original_tr_count=evidence.original_tr_count,
        expected_tr_count_mean=expected_mean,
        tr_count_fraction=evidence.original_tr_count / expected_mean,
        flags=flags,
    )
