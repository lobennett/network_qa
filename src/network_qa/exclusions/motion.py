"""Motion exclusion generator: reads motion_qa's motion_metrics.tsv + applies study thresholds."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

from network_qa.exclusions.base import register_generator


class MotionGenerator:
    name = "motion"
    description = "Motion exclusions from motion_qa's motion_metrics.tsv (study FD/DVARS thresholds)"

    def add_cli_args(self, parser: ArgumentParser) -> None:
        # Not argparse-required: every generator's args land on the same shared
        # compile subparser, so a global required=True would break unrelated
        # subset compiles (e.g. `--generators short_run behavioral`) that never
        # supply a motion TSV. The runtime guard in generate() no-ops cleanly
        # when this source isn't selected/supplied (matches lev1_outlier /
        # qa_decisions, which are likewise non-required).
        parser.add_argument("--motion-metrics-tsv", required=False, default=None,
                            help="motion_qa motion_metrics.tsv for the cohort "
                                 "(required when generators includes 'motion')")
        parser.add_argument("--fd-threshold", type=float, default=0.2,
                            help="rest FD-mean threshold (default 0.2)")
        parser.add_argument("--proportion-fd-threshold", type=float, default=0.2,
                            help="task proportion FD>0.5 threshold (default 0.2)")
        parser.add_argument("--proportion-dvars-threshold", type=float, default=0.2,
                            help="proportion std_dvars>1.5 threshold (default 0.2)")

    def generate(self, dataset_name: str, dataset_config: dict, args: Namespace) -> list[dict]:
        if pd is None:
            print("Error: pandas required for motion generator")
            return []
        tsv_arg = getattr(args, "motion_metrics_tsv", None)
        if not tsv_arg:
            # No TSV supplied (subset compile that doesn't select motion) — no-op.
            return []
        tsv = Path(tsv_arg)
        if not tsv.is_file():
            print(f"No motion metrics TSV at {tsv}")
            return []
        df = pd.read_csv(tsv, sep="\t", dtype=str)
        fd_t, pfd_t, pdv_t = (args.fd_threshold, args.proportion_fd_threshold,
                              args.proportion_dvars_threshold)
        entries = []
        for _, row in df.iterrows():
            task = str(row["task"])
            fd_mean = float(row.get("fmriprep_fd_mean", 0) or 0)
            prop_fd = float(row.get("fmriprep_proportion_fd_over_0.5", 0) or 0)
            prop_dv = float(row.get("fmriprep_proportion_std_dvars_over_1.5", 0) or 0)
            reasons = []
            if task == "rest":
                if fd_mean > fd_t:
                    reasons.append(f"Resting FD mean ({fd_mean:.3f}) > {fd_t}")
            else:
                if prop_fd > pfd_t:
                    reasons.append(f"Proportion FD>0.5 ({prop_fd:.3f}) > {pfd_t}")
            if prop_dv > pdv_t:
                reasons.append(f"Proportion std_dvars>1.5 ({prop_dv:.3f}) > {pdv_t}")
            if reasons:
                # motion_qa's TSV stores BARE subject/session; add sub-/ses-
                # prefixes here so all four generators emit the same
                # BIDS-prefixed convention as the monolith (matches what
                # is_excluded/lev1 query with, and what render.py expects).
                entries.append({
                    "subject": f"sub-{row['subject']}", "session": f"ses-{row['session']}",
                    "task": f"task-{task}", "run": f"run-{row['run']}",
                    "source": "motion", "action": "exclude", "reason": "; ".join(reasons),
                })
        print(f"Motion generator: {len(entries)} exclusions from {len(df)} scans")
        return entries


register_generator(MotionGenerator())
