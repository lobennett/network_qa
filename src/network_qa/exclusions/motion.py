"""MRIQC motion evidence for scan review and legacy exclusion lockfiles.

Review uses trusted echo 2, mean FD >=0.2 mm, and (for tasks) >=20% of frames
above 0.5 mm. A different IQM cutoff can be recalculated from matching MRIQC
timeseries after checking units, frame counts, and the original mean/percentage.
Percentages include the initial undefined-FD volume in their denominator, matching
MRIQC, and describe the frames remaining after its nonsteady-state removal.
Original IQMs remain unchanged. Standardized DVARS is recorded, not thresholded.

The legacy MotionGenerator keeps its original IQM-only command-line contract.
"""
from __future__ import annotations

import csv
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
from network_qa._evidence_paths import iter_evidence_files, valid_report
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
    fd_method: str = 'mriqc_iqm'
    fd_source: str = ''
    fd_original_thres: float | None = None
    fd_n_volumes: int | None = None
    fd_dummy_trs: int | None = None


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

    The study expects echoes 1, 2, and 3. Without trusted observed echo 2, a group
    deliberately receives no substitute metric, even if only one image was found.
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
        evidence.append(_read_motion_iqm(matches[0], report_path, report_flags, functional.tr_count))
    return tuple(evidence)


def _motion_candidates(mriqc_dir: Path) -> tuple[_IqmCandidate, ...]:
    if not mriqc_dir.is_dir():
        return ()
    candidates = []
    for path in sorted(path for path in iter_evidence_files(mriqc_dir) if path.name.endswith("_bold.json")):
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
    if any(not valid_report(path) for path in matches):
        return None, ("invalid_report",)
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

    if not echo_two_observed:
        return None, ("missing_echo_2",)
    if functional.representative_echo != 2 or disqualified:
        return None, ("untrusted_echo_2",)
    return 2, ()


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
    tr_count: int | None = None,
) -> MotionEvidence:
    flags = set((*candidate.identity_flags, *report_flags))
    try:
        iqm = json.loads(candidate.path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return MotionEvidence(candidate.key, None, None, None, None, report_path,
                              tuple(sorted((*flags, "malformed_iqm"))))
    if not isinstance(iqm, dict):
        return MotionEvidence(candidate.key, None, None, None, None, report_path,
                              tuple(sorted((*flags, "malformed_iqm"))))

    fd_mean = _finite_number(iqm.get("fd_mean"))
    fd_perc = _finite_number(iqm.get("fd_perc"))
    dvars_std = _finite_number(iqm.get("dvars_std"))
    fd_thres = _iqm_fd_thres(iqm)
    values = (fd_mean, fd_perc, dvars_std, fd_thres)
    if any(value is None or value < 0 for value in values) or fd_perc > 100:
        flags.add("malformed_iqm")
        return _empty_motion(candidate.key, tuple(flags), report_path)
    calculation = {'fd_original_thres': fd_thres}
    if fd_thres != EXPECTED_FD_THRES:
        try:
            fd_perc, details = _recalculate_fd(candidate.path, iqm, tr_count, fd_thres)
            fd_thres = EXPECTED_FD_THRES
            calculation.update(details)
        except FileNotFoundError:
            flags.add('fd_thres_mismatch')
        except (OSError, ValueError, TypeError, KeyError, csv.Error):
            flags.update(('fd_thres_mismatch', 'invalid_fd_timeseries'))
    if "malformed_iqm" not in flags and "fd_thres_mismatch" not in flags:
        high_motion = fd_mean >= FD_MEAN_THRESHOLD
        if candidate.key.task != "rest":
            high_motion = high_motion or fd_perc >= FD_PERCENT_THRESHOLD
        if high_motion:
            flags.add("excessive_motion")
    return MotionEvidence(
        candidate.key, fd_mean, fd_perc, dvars_std, fd_thres, report_path,
        tuple(sorted(flags)), **calculation,
    )


def _recalculate_fd(path: Path, iqm: dict, tr_count: int | None, original_threshold: float):
    """Reproduce MRIQC's original metrics before applying our framewise cutoff."""
    series = path.with_name(path.name.replace('_bold.json', '_timeseries.tsv'))
    metadata = json.loads(series.with_suffix('.json').read_text())
    if metadata['framewise_displacement']['Units'] != 'mm':
        raise ValueError('FD units must be mm')
    with series.open(newline='') as stream:
        values = [row['framewise_displacement'] for row in csv.DictReader(stream, delimiter='\t')]
    if len(values) < 2 or values[0] != 'n/a':
        raise ValueError('expected one initial undefined FD sample')
    fd = [float(value) for value in values[1:]]
    if any(not math.isfinite(value) or value < 0 for value in fd):
        raise ValueError('invalid FD sample')
    n, dummy = iqm['size_t'], iqm['dummy_trs']
    if type(n) is not int or type(dummy) is not int or dummy < 0 or n != len(values) or n + dummy != tr_count:
        raise ValueError('MRIQC timeseries does not match analyzed and BIDS frame counts')
    # MRIQC's fd_perc includes the initial volume in its denominator; fd_mean does not.
    original_percent = 100 * sum(value > original_threshold for value in fd) / n
    if not (math.isclose(sum(fd) / len(fd), iqm['fd_mean'], rel_tol=1e-6, abs_tol=1e-8)
            and math.isclose(original_percent, iqm['fd_perc'], rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError('timeseries does not reproduce original MRIQC metrics')
    return 100 * sum(value > EXPECTED_FD_THRES for value in fd) / n, {
        'fd_method': 'verified_mriqc_timeseries', 'fd_source': str(series),
        'fd_n_volumes': n, 'fd_dummy_trs': dummy,
    }


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
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None

ENTITIES = re.compile(
    r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)"
    r"(?:_acq-[^_]+)?(?:_run-(?P<run>[^_]+))?(?:_echo-(?P<echo>\d+))?_bold\.json$"
)


def _iqm_files(mriqc_dir: Path, subjects: set[str] | None = None) -> list[Path]:
    """One IQM file per acquisition: echo-1 where multi-echo, else the only file."""
    keep: dict[tuple, Path] = {}
    for p in sorted(path for path in iter_evidence_files(mriqc_dir)
                    if path.name.endswith("_bold.json")):
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
