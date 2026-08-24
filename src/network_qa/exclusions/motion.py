"""Motion exclusions from MRIQC's IQMs.

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

Multi-echo: MRIQC writes one IQM file per echo. Head motion is shared, so echo-1 stands
for the acquisition and the others are ignored.
"""
from __future__ import annotations

import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

from network_qa.exclusions.base import register_generator

ENTITIES = re.compile(
    r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)"
    r"(?:_acq-[^_]+)?(?:_run-(?P<run>[^_]+))?(?:_echo-(?P<echo>\d+))?_bold\.json$"
)


def _iqm_files(mriqc_dir: Path) -> list[Path]:
    """One IQM file per acquisition: echo-1 where multi-echo, else the only file."""
    keep: dict[tuple, Path] = {}
    for p in sorted(mriqc_dir.rglob("*_bold.json")):
        m = ENTITIES.match(p.name)
        if not m:
            continue
        echo = m.group("echo")
        if echo is not None and echo != "1":
            continue
        keep[(m["subject"], m["session"], m["task"], m["run"] or "1")] = p
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
            return []                       # subset compile that did not select motion
        root = Path(root)
        if not root.is_dir():
            print(f"No MRIQC derivatives at {root}")
            return []

        subjects = dataset_config.get("subjects")
        fd_t = args.fd_threshold
        pfd_t = args.proportion_fd_threshold
        expect = getattr(args, "expect_fd_thres", None)

        entries, seen, mismatched = [], 0, set()
        for path in _iqm_files(root):
            m = ENTITIES.match(path.name)
            if subjects and m["subject"] not in subjects:
                continue
            try:
                iqm = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            seen += 1

            # fd_perc is a percentage of frames above the threshold MRIQC ran with, so a
            # different fd_thres makes it a different criterion entirely.
            got = (iqm.get("provenance", {}).get("settings", {}).get("fd_thres")
                   or iqm.get("fd_thres"))
            if expect is not None and got is not None and abs(float(got) - expect) > 1e-9:
                mismatched.add(float(got))
                continue

            reasons = []
            fd_mean = iqm.get("fd_mean")
            fd_perc = iqm.get("fd_perc")
            if m["task"] == "rest":
                if fd_mean is not None and fd_mean > fd_t:
                    reasons.append(f"rest fd_mean ({fd_mean:.3f}) > {fd_t}")
            elif fd_perc is not None and fd_perc / 100.0 > pfd_t:
                reasons.append(f"fd_perc ({fd_perc:.1f}%) > {pfd_t:.0%} of frames "
                               f"over {expect} mm")

            if reasons:
                entries.append({
                    "subject": m["subject"], "session": m["session"],
                    "task": f"task-{m['task']}", "run": f"run-{m['run'] or '1'}",
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
