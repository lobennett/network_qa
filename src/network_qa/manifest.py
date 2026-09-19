"""Typed, deterministic contract for scan-review decisions."""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


RecordType = Literal["acquisition", "missing_expected"]
Decision = Literal["keep", "drop", "review"]


@dataclass(frozen=True, order=True)
class AcquisitionKey:
    """The BIDS entities identifying one logical acquisition."""

    record_type: RecordType
    subject: str
    session: str
    datatype: str
    suffix: str
    task: str = ""
    acquisition: str = ""
    direction: str = ""
    run: str = ""

    def __post_init__(self) -> None:
        if self.record_type not in {"acquisition", "missing_expected"}:
            raise ValueError(f"invalid record type: {self.record_type!r}")
        if not all((self.subject, self.datatype, self.suffix)):
            raise ValueError("acquisition identity requires subject, datatype, and suffix")


@dataclass(frozen=True)
class DecisionRow:
    """All evidence, recommendation, and review state for one acquisition."""

    key: AcquisitionKey
    expected_echoes: tuple[int, ...] = ()
    observed_echoes: tuple[int, ...] = ()
    missing_echoes: tuple[int, ...] = ()
    representative_echo: int | None = None
    tr_count: int | None = None
    original_tr_count: int | None = None
    expected_tr_count_mean: float | None = None
    tr_count_fraction: float | None = None
    fd_mean: float | None = None
    fd_perc: float | None = None
    fd_thres: float | None = None
    dvars_std: float | None = None
    mriqc_cjv: float | None = None
    mriqc_cnr: float | None = None
    mriqc_snr: float | None = None
    mriqc_efc: float | None = None
    mriqc_fber: float | None = None
    mriqc_qi2: float | None = None
    mriqc_wm2max: float | None = None
    mriqc_report_path: str = ""
    behavioral_status: str = ""
    event_status: str = ""
    flags: tuple[str, ...] = ()
    recommendation: str = ""
    recommendation_status: str = ""
    recommendation_rationale: str = ""
    decision: Decision = "keep"
    approval_required: bool = False
    approved: bool = False
    reason_code: str = ""
    reason_detail: str = ""
    reviewer: str = ""
    reviewed_at: str = ""

    def __post_init__(self) -> None:
        if self.decision not in {"keep", "drop", "review"}:
            raise ValueError(f"invalid decision: {self.decision!r}")

    @classmethod
    def clean(cls, key: AcquisitionKey) -> DecisionRow:
        """Build the auto-approved representation for an unflagged acquisition."""
        return cls(key=key)


_COLUMNS = (
    "record_type",
    "subject",
    "session",
    "datatype",
    "suffix",
    "task",
    "acquisition",
    "direction",
    "run",
    "expected_echoes",
    "observed_echoes",
    "missing_echoes",
    "representative_echo",
    "tr_count",
    "original_tr_count",
    "expected_tr_count_mean",
    "tr_count_fraction",
    "fd_mean",
    "fd_perc",
    "fd_thres",
    "dvars_std",
    "mriqc_cjv",
    "mriqc_cnr",
    "mriqc_snr",
    "mriqc_efc",
    "mriqc_fber",
    "mriqc_qi2",
    "mriqc_wm2max",
    "mriqc_report_path",
    "behavioral_status",
    "event_status",
    "flags",
    "recommendation",
    "recommendation_status",
    "recommendation_rationale",
    "decision",
    "approval_required",
    "approved",
    "reason_code",
    "reason_detail",
    "reviewer",
    "reviewed_at",
)


def anatomical_key(subject: str, session: str, suffix: str) -> AcquisitionKey:
    return AcquisitionKey("acquisition", subject, session, "anat", suffix)


def functional_key(subject: str, session: str, task: str, run: str) -> AcquisitionKey:
    return AcquisitionKey(
        "acquisition", subject, session, "func", "bold", task=task, run=run,
    )


def write_manifest(path: Path, rows: Iterable[DecisionRow]) -> str:
    """Write a deterministic TSV and return the SHA-256 of its exact bytes."""
    row_tuple = tuple(rows)
    _ensure_unique(row_tuple)
    ordered = tuple(sorted(row_tuple, key=lambda row: row.key))
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(_COLUMNS)
    writer.writerows(_serialize_row(row) for row in ordered)
    data = output.getvalue().encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_manifest(path: Path) -> tuple[DecisionRow, ...]:
    """Read a manifest that conforms exactly to the stable TSV schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _COLUMNS:
            raise ValueError(f"unexpected manifest columns in {path}")
        rows = tuple(_parse_row(raw, path, reader.line_num) for raw in reader)
    _ensure_unique(rows)
    return rows


def _serialize_row(row: DecisionRow) -> tuple[str, ...]:
    values = {
        "record_type": row.key.record_type,
        "subject": row.key.subject,
        "session": row.key.session,
        "datatype": row.key.datatype,
        "suffix": row.key.suffix,
        "task": row.key.task,
        "acquisition": row.key.acquisition,
        "direction": row.key.direction,
        "run": row.key.run,
        "expected_echoes": _serialize_tuple(row.expected_echoes),
        "observed_echoes": _serialize_tuple(row.observed_echoes),
        "missing_echoes": _serialize_tuple(row.missing_echoes),
        "representative_echo": _serialize_optional(row.representative_echo),
        "tr_count": _serialize_optional(row.tr_count),
        "original_tr_count": _serialize_optional(row.original_tr_count),
        "expected_tr_count_mean": _serialize_optional(row.expected_tr_count_mean),
        "tr_count_fraction": _serialize_optional(row.tr_count_fraction),
        "fd_mean": _serialize_optional(row.fd_mean),
        "fd_perc": _serialize_optional(row.fd_perc),
        "fd_thres": _serialize_optional(row.fd_thres),
        "dvars_std": _serialize_optional(row.dvars_std),
        "mriqc_cjv": _serialize_optional(row.mriqc_cjv),
        "mriqc_cnr": _serialize_optional(row.mriqc_cnr),
        "mriqc_snr": _serialize_optional(row.mriqc_snr),
        "mriqc_efc": _serialize_optional(row.mriqc_efc),
        "mriqc_fber": _serialize_optional(row.mriqc_fber),
        "mriqc_qi2": _serialize_optional(row.mriqc_qi2),
        "mriqc_wm2max": _serialize_optional(row.mriqc_wm2max),
        "mriqc_report_path": row.mriqc_report_path,
        "behavioral_status": row.behavioral_status,
        "event_status": row.event_status,
        "flags": _serialize_tuple(row.flags),
        "recommendation": row.recommendation,
        "recommendation_status": row.recommendation_status,
        "recommendation_rationale": row.recommendation_rationale,
        "decision": row.decision,
        "approval_required": _serialize_bool(row.approval_required),
        "approved": _serialize_bool(row.approved),
        "reason_code": row.reason_code,
        "reason_detail": row.reason_detail,
        "reviewer": row.reviewer,
        "reviewed_at": row.reviewed_at,
    }
    return tuple(values[column] for column in _COLUMNS)


def _parse_row(raw: dict[str, str | None], path: Path, line_num: int) -> DecisionRow:
    if None in raw or any(value is None for value in raw.values()):
        raise ValueError(f"malformed manifest row in {path}:{line_num}")
    values = {field: value or "" for field, value in raw.items()}
    try:
        return DecisionRow(
            key=AcquisitionKey(**{field: values[field] for field in _COLUMNS[:9]}),
            expected_echoes=_parse_int_tuple(values["expected_echoes"]),
            observed_echoes=_parse_int_tuple(values["observed_echoes"]),
            missing_echoes=_parse_int_tuple(values["missing_echoes"]),
            representative_echo=_parse_optional_int(values["representative_echo"]),
            tr_count=_parse_optional_int(values["tr_count"]),
            original_tr_count=_parse_optional_int(values["original_tr_count"]),
            expected_tr_count_mean=_parse_optional_float(values["expected_tr_count_mean"]),
            tr_count_fraction=_parse_optional_float(values["tr_count_fraction"]),
            fd_mean=_parse_optional_float(values["fd_mean"]),
            fd_perc=_parse_optional_float(values["fd_perc"]),
            fd_thres=_parse_optional_float(values["fd_thres"]),
            dvars_std=_parse_optional_float(values["dvars_std"]),
            mriqc_cjv=_parse_optional_float(values["mriqc_cjv"]),
            mriqc_cnr=_parse_optional_float(values["mriqc_cnr"]),
            mriqc_snr=_parse_optional_float(values["mriqc_snr"]),
            mriqc_efc=_parse_optional_float(values["mriqc_efc"]),
            mriqc_fber=_parse_optional_float(values["mriqc_fber"]),
            mriqc_qi2=_parse_optional_float(values["mriqc_qi2"]),
            mriqc_wm2max=_parse_optional_float(values["mriqc_wm2max"]),
            mriqc_report_path=values["mriqc_report_path"],
            behavioral_status=values["behavioral_status"],
            event_status=values["event_status"],
            flags=_parse_string_tuple(values["flags"]),
            recommendation=values["recommendation"],
            recommendation_status=values["recommendation_status"],
            recommendation_rationale=values["recommendation_rationale"],
            decision=values["decision"],
            approval_required=_parse_bool(values["approval_required"]),
            approved=_parse_bool(values["approved"]),
            reason_code=values["reason_code"],
            reason_detail=values["reason_detail"],
            reviewer=values["reviewer"],
            reviewed_at=values["reviewed_at"],
        )
    except ValueError as exc:
        raise ValueError(f"invalid manifest row in {path}:{line_num}: {exc}") from exc


def _ensure_unique(rows: Iterable[DecisionRow]) -> None:
    seen: set[AcquisitionKey] = set()
    for row in rows:
        if row.key in seen:
            raise ValueError(f"duplicate acquisition: {row.key}")
        seen.add(row.key)


def _serialize_tuple(values: tuple[object, ...]) -> str:
    return ",".join(str(value) for value in values)


def _serialize_optional(value: int | float | None) -> str:
    return "" if value is None else str(value)


def _serialize_bool(value: bool) -> str:
    return "yes" if value else "no"


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return () if not value else tuple(int(item) for item in value.split(","))


def _parse_string_tuple(value: str) -> tuple[str, ...]:
    return () if not value else tuple(value.split(","))


def _parse_optional_int(value: str) -> int | None:
    return None if not value else int(value)


def _parse_optional_float(value: str) -> float | None:
    return None if not value else float(value)


def _parse_bool(value: str) -> bool:
    if value == "yes":
        return True
    if value == "no":
        return False
    raise ValueError(f"boolean must be 'yes' or 'no', got {value!r}")
