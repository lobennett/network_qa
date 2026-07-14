"""Behavioral exclusion generator — runs behavioral QC and produces exclusion entries."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from network_qa.exclusions.base import register_generator


class BehavioralGenerator:
    name = "behavioral"
    description = "Generate exclusions from behavioral QC (accuracy, RT, omission thresholds)"

    def add_cli_args(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--behavioral-dir",
            required=False,
            default=None,
            help="Path to sourcedata behavioral directory (default: {bids_dir}/sourcedata)",
        )

    def generate(self, dataset_name: str, dataset_config: dict, args: Namespace) -> list[dict]:
        try:
            from network_events.qc import run_qc
        except ImportError:
            print("Error: pandas required for behavioral generator. Install with: uv pip install -e '.[events]'")
            return []

        bids_dir = Path(dataset_config["bids_dir"])
        behavioral_dir = Path(args.behavioral_dir) if getattr(args, "behavioral_dir", None) else bids_dir / "sourcedata"

        exclusion_entries, trim_entries = run_qc(
            behavioral_dir=behavioral_dir,
            bids_dir=bids_dir,
        )

        # Source field is set by the exclusions system when saving, but include for clarity
        for entry in exclusion_entries:
            entry["source"] = "behavioral-qc"

        print(f"Behavioral QC: {len(exclusion_entries)} exclusions, {len(trim_entries)} trim entries")
        return exclusion_entries


register_generator(BehavioralGenerator())
