"""MRIQC motion evidence and legacy motion exclusions.

MRIQC already runs on every session, so its IQMs are the study's single motion source --
no recomputation from fMRIPrep confounds, and no dependency on fMRIPrep having run. That
means the exclusion set is known before preprocessing rather than after.

Two criteria map straight onto IQMs:

* rest scans -- ``fd_mean`` above ``--fd-threshold``.
* task scans -- ``fd_perc`` (percentage of frames over MRIQC's ``--fd_thres``) above
  ``--proportion-fd-threshold``. **MRIQC must have run with ``--fd_thres 0.5``** for that
  to be the study's criterion; the campaign config sets it, and ``--expect-fd-thres``
  refuses a mismatch rather than silently applying the wrong cutoff.

The study's third criterion, *proportion of frames with std_dvars > 1.5*, is NOT applied:
MRIQC reports mean ``dvars_std``, not a proportion, and a mean-based substitute was measured
against both cohorts before being dropped. It excluded nothing FD had not already caught --
0 additional runs in discovery (291 acquisitions, max mean 1.392) and 0 in validation (2308,
max 1.699, both over-threshold runs already excluded on FD). Reinstating a real spike-count
criterion needs per-frame FD/DVARS, which MRIQC does not publish in its IQMs.

``inspect_motion`` is the review-manifest interface.  It consumes already-reviewed
functional inventory rows, uses their trusted representative image (echo 2 for
multi-echo acquisitions), and makes missing or invalid evidence explicit.  The legacy
``MotionGenerator`` remains below for the pre-existing exclusion-lock CLI contract.
"""
from __future__ import annotations

import json
import math
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from network_qa.exclusions.base import (
    load_dataset_subjects, register_generator, run_entity, validate_number,
)
from network_qa.functional import FunctionalEvidence
from network_qa.manifest import AcquisitionKey


FD_MEAN_THRESHOLD = 0.2
FD_PERCENT_THRESHOLD = 20.0
EXPECTED_FD_THRES = 0.5


@dataclass(frozen=True)
class MotionEvidence:
    """MRIQC motion values and review flags for one functional acquisition."""

    key: AcquisitionKey
    fd_mean: float | None
    fd_perc: float | None
    dvars_std: float | None
    fd_thres: float | None
    report_path: Path | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class _IqmCandidate:
    path: Path
    key: AcquisitionKey
    echo: int
    identity_flags: tuple[str, ...]


def inspect_motion(
    functionals: Iterable[FunctionalEvidence], mriqc_dir: Path
) -> tuple[MotionEvidence, ...]:
    """Read trusted MRIQC IQMs and flag reviewable motion evidence problems.

    A multi-echo group without trusted echo 2 deliberately receives no substitute
    metric.  A genuine single-echo acquisition can use its sole representative image.
    """
    candidates = _motion_candidates(mriqc_dir)
    reports = _motion_reports(mriqc_dir)
    evidence = []
    for functional in sorted(functionals, key=lambda row: row.key):
        report_path, report_flags = _report_evidence(functional.key, reports)
        echo, echo_flags = _trusted_motion_echo(functional)
        if echo is None:
            evidence.append(_empty_motion(functional.key, (*report_flags, *echo_flags), report_path))
            continue
        matches = [
            candidate for candidate in candidates
            if candidate.key == functional.key and candidate.echo == echo
        ]
        if not matches:
            evidence.append(_empty_motion(functional.key, (*report_flags, "missing_iqm"), report_path))
            continue
        if len(matches) > 1:
            evidence.append(_empty_motion(functional.key, (*report_flags, "ambiguous_iqm"), report_path))
            continue
        evidence.append(_read_motion_iqm(matches[0], report_path, report_flags))
    return tuple(evidence)


def _motion_candidates(mriqc_dir: Path) -> tuple[_IqmCandidate, ...]:
    if not mriqc_dir.is_dir():
        return ()
    candidates = []
    for path in sorted(mriqc_dir.rglob("*_bold.json")):
        candidate = _parse_iqm_candidate(path)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _motion_reports(mriqc_dir: Path) -> dict[AcquisitionKey, tuple[Path, ...]]:
    """Index MRIQC's root-level, acquisition-level BOLD reports.

    MRIQC writes reports at the derivative root and omits the echo entity.  Restricting
    discovery to that layout prevents an echo IQM's imagined sibling from becoming a
    report reference.
    """
    reports: dict[AcquisitionKey, list[Path]] = {}
    if not mriqc_dir.is_dir():
        return {}
    for path in sorted(mriqc_dir.glob("*_bold.html")):
        if "_echo-" in path.name:
            continue
        candidate = _parse_iqm_candidate(path.with_suffix(".json"))
        if candidate is not None:
            reports.setdefault(candidate.key, []).append(path)
    return {key: tuple(paths) for key, paths in reports.items()}


def _report_evidence(
    key: AcquisitionKey, reports: dict[AcquisitionKey, tuple[Path, ...]],
) -> tuple[Path | None, tuple[str, ...]]:
    matches = reports.get(key, ())
    if not matches:
        return None, ("missing_report",)
    if len(matches) > 1:
        return None, ("ambiguous_report",)
    return matches[0], ()


_UNTRUSTED_FUNCTIONAL_FLAGS = frozenset({
    "ambiguous_echo", "identity_mismatch", "invalid_nifti", "unequal_echo_counts",
    "unexpected_echo",
})


def _trusted_motion_echo(functional: FunctionalEvidence) -> tuple[int | None, tuple[str, ...]]:
    """Validate that the reviewed representative can safely supply motion evidence."""
    observed = tuple(functional.observed_echoes)
    observed_set = set(observed)
    echo_two_observed = 2 in observed_set
    structurally_ambiguous = len(observed) != len(observed_set)
    disqualified = (
        structurally_ambiguous
        or functional.tr_count is None
        or bool(_UNTRUSTED_FUNCTIONAL_FLAGS.intersection(functional.flags))
    )

    if len(observed_set) > 1:
        if not echo_two_observed:
            return None, ("missing_echo_2",)
        if functional.representative_echo != 2 or disqualified:
            return None, ("untrusted_echo_2",)
        return 2, ()

    if len(observed_set) == 1:
        sole_echo, = observed_set
        if functional.representative_echo == sole_echo and not disqualified:
            return sole_echo, ()
        return None, (("untrusted_echo_2",) if echo_two_observed else ("missing_echo_2",))

    return None, ("missing_echo_2",)


def _parse_iqm_candidate(path: Path) -> _IqmCandidate | None:
    stem = path.name.removesuffix(".json")
    if not stem.endswith("_bold"):
        return None
    entities: dict[str, str] = {}
    for part in stem.removesuffix("_bold").split("_"):
        name, separator, value = part.partition("-")
        if not separator or not value or name in entities:
            return None
        entities[name] = value
    try:
        key = AcquisitionKey(
            "acquisition",
            f"sub-{entities['sub']}",
            f"ses-{entities['ses']}",
            "func",
            "bold",
            task=entities["task"],
            acquisition=entities.get("acq", ""),
            direction=entities.get("dir", ""),
            run=_canonical_run(entities["run"]),
        )
        echo = _canonical_index(entities.get("echo", "1"))
    except (KeyError, ValueError):
        return None
    return _IqmCandidate(path, key, echo, _iqm_identity_flags(path, key))


def _canonical_index(value: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"invalid numeric BIDS entity: {value!r}")
    return int(value)


def _canonical_run(value: str) -> str:
    return str(_canonical_index(value))


def _iqm_identity_flags(path: Path, key: AcquisitionKey) -> tuple[str, ...]:
    """Compare filename entities to a conventional derivative ``sub/ses/func`` path."""
    if path.parent.name != "func":
        return ()
    session_parent = path.parent.parent.name
    subject_parent = path.parent.parent.parent.name
    if subject_parent != key.subject or session_parent != key.session:
        return ("identity_mismatch",)
    return ()


def _empty_motion(
    key: AcquisitionKey, flags: tuple[str, ...], report_path: Path | None,
) -> MotionEvidence:
    return MotionEvidence(key, None, None, None, None, report_path, tuple(sorted(set(flags))))


def _read_motion_iqm(
    candidate: _IqmCandidate, report_path: Path | None, report_flags: tuple[str, ...],
) -> MotionEvidence:
    flags = set((*candidate.identity_flags, *report_flags))
    try:
        iqm = json.loads(candidate.path.read_text())
    except (OSError, json.JSONDecodeError):
        return MotionEvidence(candidate.key, None, None, None, None, report_path,
                              tuple(sorted((*flags, "malformed_iqm"))))
    if not isinstance(iqm, dict):
        return MotionEvidence(candidate.key, None, None, None, None, report_path,
                              tuple(sorted((*flags, "malformed_iqm"))))

    fd_mean = _finite_number(iqm.get("fd_mean"))
    fd_perc = _finite_number(iqm.get("fd_perc"))
    dvars_std = _finite_number(iqm.get("dvars_std"))
    fd_thres = _iqm_fd_thres(iqm)
    if None in (fd_mean, fd_perc, dvars_std, fd_thres) or not 0 <= fd_perc <= 100:
        flags.add("malformed_iqm")
    if fd_thres is not None and fd_thres != EXPECTED_FD_THRES:
        flags.add("fd_thres_mismatch")
    if "malformed_iqm" not in flags and "fd_thres_mismatch" not in flags:
        high_motion = fd_mean >= FD_MEAN_THRESHOLD
        if candidate.key.task != "rest":
            high_motion = high_motion or fd_perc >= FD_PERCENT_THRESHOLD
        if high_motion:
            flags.add("excessive_motion")
    return MotionEvidence(
        candidate.key, fd_mean, fd_perc, dvars_std, fd_thres, report_path,
        tuple(sorted(flags)),
    )


def _iqm_fd_thres(iqm: dict) -> float | None:
    provenance = iqm.get("provenance")
    if isinstance(provenance, dict):
        settings = provenance.get("settings")
        if isinstance(settings, dict) and "fd_thres" in settings:
            return _finite_number(settings["fd_thres"])
    return _finite_number(iqm.get("fd_thres"))


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None

ENTITIES = re.compile(
    r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)"
    r"(?:_acq-[^_]+)?(?:_run-(?P<run>[^_]+))?(?:_echo-(?P<echo>\d+))?_bold\.json$"
)


def _iqm_files(mriqc_dir: Path, subjects: set[str] | None = None) -> list[Path]:
    """One IQM file per acquisition: echo-1 where multi-echo, else the only file."""
    keep: dict[tuple, Path] = {}
    for p in sorted(mriqc_dir.rglob("*_bold.json")):
        m = ENTITIES.match(p.name)
        if not m:
            continue
        if subjects is not None and m["subject"] not in subjects:
            continue
        echo = m.group("echo")
        if echo is not None and int(echo) != 1:
            continue
        key = (m["subject"], m["session"], m["task"], run_entity(m["run"] or "1"))
        if key in keep:
            raise ValueError(f"Duplicate MRIQC acquisition {key}: {keep[key]} and {p}")
        keep[key] = p
    return list(keep.values())


class MotionGenerator:
    name = "motion"
    description = "Motion exclusions from MRIQC IQMs (fd_mean on rest, fd_perc on task)"

    def add_cli_args(self, parser: ArgumentParser) -> None:
        # Not argparse-required: every generator's args share one compile subparser, so a
        # global required=True would break a subset compile that never selects motion.
        parser.add_argument("--mriqc-dir", required=False, default=None,
                            help="MRIQC derivatives holding the IQM JSONs "
                                 "(required when generators includes 'motion')")
        parser.add_argument("--fd-threshold", type=float, default=0.2,
                            help="rest fd_mean threshold in mm (default 0.2)")
        parser.add_argument("--proportion-fd-threshold", type=float, default=0.2,
                            help="task threshold on the fraction of frames over MRIQC's "
                                 "fd_thres (default 0.2)")
        parser.add_argument("--expect-fd-thres", type=float, default=0.5,
                            help="refuse IQMs whose fd_thres differs from this, since "
                                 "fd_perc would then mean something else (default 0.5)")

    def generate(self, dataset_name: str, dataset_config: dict, args: Namespace) -> list[dict]:
        root = getattr(args, "mriqc_dir", None)
        if not root:
            raise FileNotFoundError("motion generator requires --mriqc-dir")
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"No MRIQC derivatives at {root}")

        subjects = load_dataset_subjects(dataset_config)
        fd_t = validate_number(args.fd_threshold, "fd_threshold")
        pfd_t = validate_number(args.proportion_fd_threshold, "proportion_fd_threshold", maximum=1)
        expect = getattr(args, "expect_fd_thres", None)
        if expect is not None:
            validate_number(expect, "expect_fd_thres")

        entries, seen, mismatched = [], 0, set()
        for path in _iqm_files(root, subjects):
            m = ENTITIES.match(path.name)
            try:
                iqm = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read MRIQC IQM {path}: {exc}") from exc
            if not isinstance(iqm, dict):
                raise ValueError(f"MRIQC IQM must be a JSON object: {path}")
            seen += 1

            # fd_perc is a percentage of frames above the threshold MRIQC ran with, so a
            # different fd_thres makes it a different criterion entirely.
            try:
                got = iqm.get("provenance", {}).get("settings", {}).get(
                    "fd_thres", iqm.get("fd_thres"))
            except AttributeError as exc:
                raise ValueError(f"Malformed MRIQC provenance in {path}") from exc
            if expect is not None:
                validate_number(got, f"{path}: fd_thres")
            if expect is not None and abs(got - expect) > 1e-9:
                mismatched.add(got)
                continue

            reasons = []
            fd_mean = iqm.get("fd_mean")
            fd_perc = iqm.get("fd_perc")
            if m["task"] == "rest":
                validate_number(fd_mean, f"{path}: fd_mean")
                if fd_mean > fd_t:
                    reasons.append(f"rest fd_mean ({fd_mean:.3f}) > {fd_t}")
            else:
                validate_number(fd_perc, f"{path}: fd_perc", maximum=100)
                if fd_perc / 100.0 > pfd_t:
                    reasons.append(f"fd_perc ({fd_perc:.1f}%) > {pfd_t:.0%} of frames "
                                   f"over {expect} mm")

            if reasons:
                entries.append({
                    "subject": m["subject"], "session": m["session"],
                    "task": f"task-{m['task']}", "run": run_entity(m['run'] or '1'),
                    "source": "motion", "action": "exclude",
                    "reason": "; ".join(reasons),
                    # dvars_std is recorded as evidence though nothing thresholds on it.
                    "metrics": {"fd_mean": fd_mean, "fd_perc": fd_perc,
                                "dvars_std": iqm.get("dvars_std"), "fd_thres": got},
                })

        if mismatched:
            raise SystemExit(
                f"MRIQC IQMs under {root} were produced with fd_thres {sorted(mismatched)}, "
                f"expected {expect}. fd_perc counts frames above whatever threshold MRIQC "
                f"ran with, so applying the task criterion to these would silently use the "
                f"wrong cutoff. Re-run MRIQC with --fd_thres {expect}, or pass "
                f"--expect-fd-thres to match deliberately."
            )
        print(f"Motion: {len(entries)} exclusions from {seen} acquisitions")
        return entries


register_generator(MotionGenerator())
