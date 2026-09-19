"""Anatomical inventory and MRIQC evidence for scan-review manifests."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from network_qa.manifest import AcquisitionKey
from network_qa._evidence_paths import iter_evidence_files, valid_report


METRIC_NAMES = ("cjv", "cnr", "snr", "efc", "fber", "qi_2", "wm2max")
SECONDARY_DIRECTIONS = {
    "snr": "higher",
    "efc": "lower",
    "fber": "higher",
    "qi_2": "lower",
}
_ANATOMICAL_SUFFIXES = ("T1w", "T2w")


@dataclass(frozen=True)
class AnatomicalEvidence:
    """MRIQC evidence and a nonbinding duplicate-selection recommendation."""

    key: AcquisitionKey
    metrics: dict[str, float | None]
    report_path: Path | None
    flags: tuple[str, ...]
    recommendation: str
    recommendation_status: str
    recommendation_rationale: str


@dataclass(frozen=True)
class _ImageCandidate:
    path: Path
    key: AcquisitionKey
    identity_flags: tuple[str, ...]


@dataclass(frozen=True)
class _IqmCandidate:
    path: Path
    key: AcquisitionKey
    identity_flags: tuple[str, ...]


@dataclass(frozen=True)
class _ImageGroup:
    key: AcquisitionKey
    images: tuple[_ImageCandidate, ...]


@dataclass(frozen=True)
class _ReportCandidate:
    path: Path
    valid: bool


def inspect_anatomicals(
    bids_dir: Path, mriqc_dir: Path, subjects: Iterable[str],
) -> tuple[AnatomicalEvidence, ...]:
    """Flag missing or duplicate anatomy and recommend among duplicate pairs.

    This produces evidence only.  It deliberately has no decision or approval state:
    downstream manifest compilation owns the review workflow.
    """
    selected_subjects = {_canonical_subject(subject) for subject in subjects}
    images = tuple(candidate for candidate in _image_candidates(bids_dir)
                   if candidate.key.subject in selected_subjects)
    iqms = _iqm_candidates(mriqc_dir)
    reports = _mriqc_reports(mriqc_dir)
    by_subject_suffix: dict[tuple[str, str], list[_ImageGroup]] = defaultdict(list)
    for group in _image_groups(images):
        by_subject_suffix[group.key.subject, group.key.suffix].append(group)

    rows: list[AnatomicalEvidence] = []
    for subject in sorted(selected_subjects):
        for suffix in _ANATOMICAL_SUFFIXES:
            groups = sorted(by_subject_suffix[subject, suffix], key=lambda item: item.key)
            if not groups:
                rows.append(_missing_expected(subject, suffix))
                continue
            evidence = [_observed_evidence(group, iqms, reports) for group in groups]
            physical_count = sum(len(group.images) for group in groups)
            count_review = physical_count != 1 or any(_untrusted_image(row) for row in evidence)
            if count_review:
                evidence = [_with_count_flag(row) for row in evidence]
                evidence = _with_recommendation(evidence)
            rows.extend(evidence)
    return tuple(sorted(rows, key=lambda row: row.key))


def _image_candidates(bids_dir: Path) -> tuple[_ImageCandidate, ...]:
    if not bids_dir.is_dir():
        return ()
    paths = sorted((*bids_dir.glob("sub-*/ses-*/anat/*_T1w.nii"),
                    *bids_dir.glob("sub-*/ses-*/anat/*_T1w.nii.gz"),
                    *bids_dir.glob("sub-*/ses-*/anat/*_T2w.nii"),
                    *bids_dir.glob("sub-*/ses-*/anat/*_T2w.nii.gz")))
    candidates = []
    invalid_paths = []
    for path in paths:
        candidate = _parse_candidate(path, _nifti_stem(path))
        if candidate is None:
            invalid_paths.append(path)
        else:
            candidates.append(_physical_image_candidate(candidate))
    # Reserve every trusted identity before allocating deterministic observation
    # keys. A malformed filename must never disappear or alias a valid image.
    occupied = {candidate.key for candidate in candidates}
    for path in invalid_paths:
        acquisition = "invalid" + hashlib.sha256(path.relative_to(bids_dir).as_posix().encode()).hexdigest()
        while True:
            key = AcquisitionKey(
                "acquisition", path.parents[2].name, path.parents[1].name, "anat",
                _nifti_stem(path).split("_")[-1], acquisition=acquisition,
            )
            if key not in occupied:
                break
            acquisition += "x"
        occupied.add(key)
        candidates.append(_ImageCandidate(path, key, ("invalid_identity", "untrusted_identity")))
    return tuple(candidates)


def _physical_image_candidate(candidate: _ImageCandidate) -> _ImageCandidate:
    """Bind a BIDS inventory entry to its selected physical subject/session parent."""
    path = candidate.path
    physical_key = AcquisitionKey(
        "acquisition",
        path.parent.parent.parent.name,
        path.parent.parent.name,
        "anat",
        candidate.key.suffix,
        acquisition=candidate.key.acquisition,
        run=candidate.key.run,
    )
    flags = set(candidate.identity_flags)
    if (candidate.key.subject, candidate.key.session) != (
        physical_key.subject, physical_key.session,
    ):
        flags.update(("identity_mismatch", "untrusted_identity"))
    return _ImageCandidate(path, physical_key, tuple(sorted(flags)))


def _image_groups(images: tuple[_ImageCandidate, ...]) -> tuple[_ImageGroup, ...]:
    grouped: dict[AcquisitionKey, list[_ImageCandidate]] = defaultdict(list)
    for image in images:
        grouped[image.key].append(image)
    return tuple(
        _ImageGroup(key, tuple(sorted(group, key=lambda item: item.path)))
        for key, group in sorted(grouped.items())
    )


def _iqm_candidates(mriqc_dir: Path) -> tuple[_IqmCandidate, ...]:
    if not mriqc_dir.is_dir():
        return ()
    candidates = []
    for path in sorted(path for path in iter_evidence_files(mriqc_dir)
                       if path.name.endswith(("_T1w.json", "_T2w.json"))):
        parsed = _parse_candidate(path, path.name.removesuffix(".json"))
        if parsed is not None:
            candidates.append(_IqmCandidate(parsed.path, parsed.key, parsed.identity_flags))
    return tuple(candidates)


def _mriqc_reports(mriqc_dir: Path) -> dict[AcquisitionKey, tuple[_ReportCandidate, ...]]:
    """Index root-level, per-acquisition reports by their complete BIDS identity."""
    reports: dict[AcquisitionKey, list[_ReportCandidate]] = defaultdict(list)
    if not mriqc_dir.is_dir():
        return {}
    for path in sorted((*mriqc_dir.glob("*_T1w.html"), *mriqc_dir.glob("*_T2w.html"))):
        parsed = _parse_candidate(path, path.name.removesuffix(".html"))
        if parsed is not None:
            reports[parsed.key].append(_ReportCandidate(path, valid_report(path)))
    return {key: tuple(paths) for key, paths in reports.items()}


def _parse_candidate(path: Path, stem: str) -> _ImageCandidate | None:
    parts = stem.split("_")
    if not parts or parts[-1] not in _ANATOMICAL_SUFFIXES:
        return None
    entities: dict[str, str] = {}
    for part in parts[:-1]:
        name, separator, value = part.partition("-")
        if not separator or not value or name in entities or name not in {"sub", "ses", "acq", "run"}:
            return None
        entities[name] = value
    try:
        key = AcquisitionKey(
            "acquisition",
            f"sub-{entities['sub']}",
            f"ses-{entities['ses']}",
            "anat",
            parts[-1],
            acquisition=entities.get("acq", ""),
            run=_canonical_run(entities.get("run", "")),
        )
    except (KeyError, ValueError):
        return None
    return _ImageCandidate(path, key, _identity_flags(path, key))


def _nifti_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name.removesuffix(".nii.gz")
    return path.name.removesuffix(".nii")


def _canonical_subject(subject: str) -> str:
    label = subject.removeprefix("sub-")
    return f"sub-{label}"


def _canonical_run(value: str) -> str:
    if not value:
        return ""
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"invalid run entity: {value!r}")
    return value.lstrip("0") or "0"


def _identity_flags(path: Path, key: AcquisitionKey) -> tuple[str, ...]:
    if path.parent.name != "anat":
        return ()
    if path.parent.parent.name != key.session or path.parent.parent.parent.name != key.subject:
        return ("identity_mismatch",)
    return ()


def _missing_expected(subject: str, suffix: str) -> AnatomicalEvidence:
    return AnatomicalEvidence(
        key=AcquisitionKey("missing_expected", subject, "", "anat", suffix),
        metrics={},
        report_path=None,
        flags=("anatomical_count",),
        recommendation="",
        recommendation_status="indeterminate",
        recommendation_rationale=f"No {suffix} acquisition was observed for {subject}.",
    )


def _observed_evidence(
    group: _ImageGroup,
    iqms: tuple[_IqmCandidate, ...],
    reports: dict[AcquisitionKey, tuple[_ReportCandidate, ...]],
) -> AnatomicalEvidence:
    flags = {flag for image in group.images for flag in image.identity_flags}
    if len(group.images) > 1:
        flags.update(("ambiguous_image", "untrusted_image"))
    if "untrusted_identity" in flags:
        return AnatomicalEvidence(
            group.key, _empty_metrics(), None, tuple(sorted(flags)), "", "", "",
        )
    if "untrusted_image" in flags:
        return AnatomicalEvidence(
            group.key, _empty_metrics(), None, tuple(sorted(flags)), "", "", "",
        )
    report_path, report_flags = _report_evidence(group.key, reports)
    flags.update(report_flags)
    matches = [iqm for iqm in iqms if iqm.key == group.key]
    if not matches:
        flags.add("missing_iqm")
        return AnatomicalEvidence(group.key, _empty_metrics(), report_path, tuple(sorted(flags)), "", "", "")
    if len(matches) > 1:
        flags.add("ambiguous_iqm")
        return AnatomicalEvidence(group.key, _empty_metrics(), report_path, tuple(sorted(flags)), "", "", "")
    iqm = matches[0]
    flags.update(iqm.identity_flags)
    metrics, metric_flags = _read_metrics(iqm.path)
    flags.update(metric_flags)
    return AnatomicalEvidence(group.key, metrics, report_path, tuple(sorted(flags)), "", "", "")


def _report_evidence(
    key: AcquisitionKey, reports: dict[AcquisitionKey, tuple[_ReportCandidate, ...]],
) -> tuple[Path | None, tuple[str, ...]]:
    matches = reports.get(key, ())
    if not matches:
        return None, ("missing_report",)
    if any(not report.valid for report in matches):
        return None, ("invalid_report",)
    if len(matches) > 1:
        return None, ("ambiguous_report",)
    return matches[0].path, ()


def _empty_metrics() -> dict[str, float | None]:
    return {name: None for name in METRIC_NAMES}


def _read_metrics(path: Path) -> tuple[dict[str, float | None], tuple[str, ...]]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return _empty_metrics(), ("malformed_iqm",)
    if not isinstance(raw, dict):
        return _empty_metrics(), ("malformed_iqm",)
    metrics = {
        "cjv": _finite_number(raw.get("cjv")),
        "cnr": _finite_number(raw.get("cnr")),
        "snr": _snr(raw),
        "efc": _finite_number(raw.get("efc")),
        "fber": _finite_number(raw.get("fber")),
        "qi_2": _finite_number(raw.get("qi_2")),
        "wm2max": _finite_number(raw.get("wm2max")),
    }
    return metrics, (("malformed_iqm",) if any(value is None for value in metrics.values()) else ())


def _snr(raw: dict) -> float | None:
    direct = _finite_number(raw.get("snr_total"))
    if direct is not None:
        return direct
    value = raw.get("snr")
    if isinstance(value, dict):
        return _finite_number(value.get("total"))
    return _finite_number(value)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _with_count_flag(row: AnatomicalEvidence) -> AnatomicalEvidence:
    return AnatomicalEvidence(
        row.key, row.metrics, row.report_path,
        tuple(sorted((*row.flags, "anatomical_count"))),
        row.recommendation, row.recommendation_status, row.recommendation_rationale,
    )


def _with_recommendation(rows: list[AnatomicalEvidence]) -> list[AnatomicalEvidence]:
    recommendation, status, rationale = _recommend(rows)
    return [
        AnatomicalEvidence(
            row.key, row.metrics, row.report_path, row.flags,
            recommendation, status, rationale,
        )
        for row in rows
    ]


def _recommend(rows: list[AnatomicalEvidence]) -> tuple[str, str, str]:
    if len(rows) != 2:
        return "", "indeterminate", "MRIQC ranking is defined for a duplicate pair."
    first, second = rows
    if _untrusted_image(first) or _untrusted_image(second):
        return "", "indeterminate", "At least one duplicate image identity is untrusted."
    first_valid, second_valid = _complete_mriqc(first), _complete_mriqc(second)
    if first_valid != second_valid:
        winner = 0 if first_valid else 1
        return _keep_label(winner), "clear", "Complete valid MRIQC output selects the recommendation."
    if not first_valid:
        return "", "indeterminate", "Neither duplicate has complete valid MRIQC output."

    cjv = _compare(first.metrics["cjv"], second.metrics["cjv"], "lower")
    cnr = _compare(first.metrics["cnr"], second.metrics["cnr"], "higher")
    if cjv is not None and cjv == cnr:
        return _keep_label(cjv), "clear", "CJV and CNR select the same image."
    if cjv is None or cnr is None or cjv == cnr:
        return "", "indeterminate", "Primary MRIQC metrics do not select an image."

    votes = _secondary_votes(first.metrics, second.metrics)
    first_votes = votes.count(0)
    second_votes = votes.count(1)
    if first_votes == second_votes:
        return "", "indeterminate", "Secondary MRIQC metrics are tied or unavailable."
    winner = 0 if first_votes > second_votes else 1
    return _keep_label(winner), "clear", (
        f"CJV and CNR disagree; secondary MRIQC majority selects {['first', 'second'][winner]} "
        f"({max(first_votes, second_votes)} to {min(first_votes, second_votes)})."
    )


def _complete_mriqc(row: AnatomicalEvidence) -> bool:
    evidence_flags = {
        "identity_mismatch", "missing_iqm", "ambiguous_iqm", "malformed_iqm",
        "missing_report", "ambiguous_report", "invalid_report", "untrusted_image",
        "untrusted_identity",
    }
    return row.report_path is not None and not evidence_flags.intersection(row.flags) and all(
        row.metrics.get(name) is not None for name in METRIC_NAMES
    )


def _untrusted_image(row: AnatomicalEvidence) -> bool:
    return bool({"untrusted_image", "untrusted_identity"}.intersection(row.flags))


def _compare(first: float | None, second: float | None, direction: str) -> int | None:
    if first is None or second is None or first == second:
        return None
    if direction == "higher":
        return 0 if first > second else 1
    return 0 if first < second else 1


def _secondary_votes(
    first: dict[str, float | None], second: dict[str, float | None],
) -> list[int]:
    votes = []
    for metric, direction in SECONDARY_DIRECTIONS.items():
        if metric == "fber" and (-1 in (first[metric], second[metric])):
            continue
        vote = _compare(first[metric], second[metric], direction)
        if vote is not None:
            votes.append(vote)
    wm2max = _wm2max_comparison(first["wm2max"], second["wm2max"])
    if wm2max is not None:
        votes.append(wm2max)
    return votes


def _wm2max_comparison(first: float | None, second: float | None) -> int | None:
    if first is None or second is None:
        return None
    first_distance = _interval_distance(first, 0.6, 0.8)
    second_distance = _interval_distance(second, 0.6, 0.8)
    return _compare(first_distance, second_distance, "lower")


def _interval_distance(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _keep_label(index: int) -> str:
    return ("keep-first", "keep-second")[index]
