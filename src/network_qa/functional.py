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
    tr_count: int | None
    flags: tuple[str, ...] = ()


def inspect_functionals(bids_dir: Path) -> tuple[FunctionalEvidence, ...]:
    """Group BOLD echoes, read dim4, and calculate task-level expected lengths."""
    grouped: dict[AcquisitionKey, list[_ObservedImage]] = defaultdict(list)
    for path in _bold_paths(bids_dir):
        key, echo, identity_flags = _functional_identity(path)
        grouped[key].append(_read_image(path, echo, identity_flags))

    preliminary = tuple(_build_evidence(key, images) for key, images in sorted(grouped.items()))
    expected_means = _task_count_means(preliminary)
    return tuple(_with_scan_length_flags(evidence, expected_means) for evidence in preliminary)


def _bold_paths(bids_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(
        (*bids_dir.glob("sub-*/ses-*/func/*_bold.nii"),
         *bids_dir.glob("sub-*/ses-*/func/*_bold.nii.gz")),
    ))


def _functional_identity(path: Path) -> tuple[AcquisitionKey, int, tuple[str, ...]]:
    filename = _nifti_stem(path)
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
    flags = ()
    if (key.subject != path.parents[2].name or key.session != path.parents[1].name):
        flags = ("identity_mismatch",)
    return key, int(echo), flags


def _nifti_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name.removesuffix(".nii.gz")
    if path.name.endswith(".nii"):
        return path.name.removesuffix(".nii")
    raise ValueError(f"not a NIfTI: {path}")


def _canonical_index(value: str, entity: str, path: Path) -> str:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"invalid {entity} entity in {path}: {value!r}")
    return value.lstrip("0") or "0"


def _read_image(path: Path, echo: int, flags: tuple[str, ...]) -> _ObservedImage:
    try:
        shape = nib.load(str(path)).shape
    except Exception:
        return _ObservedImage(echo=echo, tr_count=None, flags=(*flags, "invalid_nifti"))
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        return _ObservedImage(echo=echo, tr_count=None, flags=(*flags, "invalid_nifti"))
    return _ObservedImage(echo=echo, tr_count=shape[3], flags=flags)


def _build_evidence(key: AcquisitionKey, images: list[_ObservedImage]) -> FunctionalEvidence:
    observed_echoes = tuple(sorted({image.echo for image in images}))
    missing_echoes = tuple(echo for echo in EXPECTED_ECHOES if echo not in observed_echoes)
    duplicate_echo = len(observed_echoes) != len(images)
    invalid_identity = any("identity_mismatch" in image.flags for image in images)
    counts = tuple(image.tr_count for image in images)
    countable = not duplicate_echo and not invalid_identity and all(count is not None for count in counts)
    counts_agree = countable and len(set(counts)) == 1
    unexpected_echo = bool(set(observed_echoes) - set(EXPECTED_ECHOES))
    representative_echo, tr_count = _representative_count(
        observed_echoes, counts[0] if counts_agree else None, unexpected_echo,
    )
    flags = {flag for image in images for flag in image.flags}
    if missing_echoes:
        flags.add("missing_echo")
    if unexpected_echo:
        flags.add("unexpected_echo")
    if duplicate_echo:
        flags.add("ambiguous_echo")
    if countable and not counts_agree:
        flags.add("unequal_echo_counts")
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
        flags=tuple(sorted(flags)),
    )


def _representative_count(
    observed_echoes: tuple[int, ...], agreed_count: int | None, unexpected_echo: bool,
) -> tuple[int | None, int | None]:
    if agreed_count is None or unexpected_echo:
        return None, None
    if 2 in observed_echoes:
        return 2, agreed_count
    if len(observed_echoes) == 1:
        return observed_echoes[0], agreed_count
    return None, agreed_count


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
