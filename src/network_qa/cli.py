"""network-qa CLI: compile registered exclusion generators into a lockfile,
and render the lockfile (+ pipeline config) into the 3 data-selection
channels (bids-filter-file / scans.tsv / .bidsignore).

Also keeps the pre-existing `qa_runs` (short-run flagging) entrypoint
reachable as a `qa-runs` subcommand, in addition to its own `nf-qa-runs`
console script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import generators so they self-register with the exclusions.base registry.
from network_qa.exclusions import behavioral  # noqa: F401
from network_qa.exclusions import motion  # noqa: F401
from network_qa.exclusions import lev1_outlier  # noqa: F401
from network_qa.exclusions import qa_decisions  # noqa: F401
from network_qa.exclusions import short_run  # noqa: F401

from network_qa.exclusions.base import list_generators
from network_qa.compile import compile_exclusions, write_lockfile, load_lockfile
from network_qa.render import render_bids_filter, render_scans_tsv, render_bidsignore
from network_qa.qa_runs import flag_short_runs, scan_bold_volumes, format_report


def _cmd_compile(args: argparse.Namespace) -> None:
    lock = compile_exclusions(args.dataset, {}, args, generator_names=None)
    write_lockfile(lock, args.out)
    n = len(lock.get("exclusions", []))
    print(f"Compiled {n} exclusions for '{args.dataset}' -> {args.out}")


def _cmd_render_bids_filter(args: argparse.Namespace) -> None:
    cfg = {"anat_acquisition": args.anat_acquisition, "tasks": args.task}
    result = render_bids_filter(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote bids-filter-file -> {out_path}")


def _cmd_render_scans_tsv(args: argparse.Namespace) -> None:
    entries = load_lockfile(args.lockfile)
    written = render_scans_tsv(entries, args.bids_dir)
    print(f"Wrote {len(written)} scans.tsv file(s) under {args.bids_dir}")


def _cmd_render_bidsignore(args: argparse.Namespace) -> None:
    entries = load_lockfile(args.lockfile)
    lines = render_bidsignore(entries, args.out)
    print(f"Wrote {len(lines)} line(s) -> {args.out}")


def _cmd_qa_runs(args: argparse.Namespace) -> None:
    rows = flag_short_runs(scan_bold_volumes(args.bids_root), frac=args.frac)
    report = format_report(rows)
    if args.out:
        Path(args.out).write_text(report + "\n")
    else:
        print(report)
    n_drop = sum(r["verdict"] == "drop" for r in rows)
    sys.stderr.write(f"\n{len(rows)} runs, {n_drop} flagged to drop (frac={args.frac}).\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="network-qa", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    # compile
    comp_p = sub.add_parser("compile", help="Run registered generators -> provenance lockfile")
    comp_p.add_argument("--dataset", required=True, help="Dataset name (e.g. discovery)")
    comp_p.add_argument("--out", required=True, help="Path to write the lockfile JSON")
    for gen in list_generators().values():
        gen.add_cli_args(comp_p)
    comp_p.set_defaults(func=_cmd_compile)

    # render
    render_p = sub.add_parser("render", help="Render the 3 data-selection channels")
    render_sub = render_p.add_subparsers(dest="render_command", required=True)

    bf_p = render_sub.add_parser("bids-filter", help="Coarse per-pipeline pybids filter")
    bf_p.add_argument("--anat-acquisition", required=True, help="Canonical anat acquisition label")
    bf_p.add_argument("--task", action="append", required=True, help="Task name (repeatable)")
    bf_p.add_argument("--out", required=True, help="Path to write the bids-filter-file JSON")
    bf_p.set_defaults(func=_cmd_render_bids_filter)

    st_p = render_sub.add_parser("scans-tsv", help="Per-session scans.tsv (filename + why)")
    st_p.add_argument("--lockfile", required=True, help="Compiled lockfile JSON")
    st_p.add_argument("--bids-dir", required=True, help="BIDS dataset root")
    st_p.set_defaults(func=_cmd_render_scans_tsv)

    bi_p = render_sub.add_parser("bidsignore", help=".bidsignore (invalid-only) from the lockfile")
    bi_p.add_argument("--lockfile", required=True, help="Compiled lockfile JSON")
    bi_p.add_argument("--out", required=True, help="Path to write .bidsignore")
    bi_p.set_defaults(func=_cmd_render_bidsignore)

    # qa-runs (pre-existing flag_short_runs entrypoint, also its own nf-qa-runs script)
    qa_p = sub.add_parser("qa-runs", help="Flag short functional runs by per-task cohort-mean volumes")
    qa_p.add_argument("bids_root", help="BIDS dataset root")
    qa_p.add_argument("--frac", type=float, default=0.5,
                      help="drop runs with < frac * per-task cohort-mean volumes (default 0.5)")
    qa_p.add_argument("--out", help="write TSV here (default: stdout)")
    qa_p.set_defaults(func=_cmd_qa_runs)

    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
