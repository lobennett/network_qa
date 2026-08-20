"""network-qa CLI: compile registered exclusion generators into a lockfile.

Two compiles gate the models, each downstream of the step that produces its evidence:
motion + behavioural after fMRIPrep decide what enters the first level, lev1 outliers after
the cohort QC decide what enters the second. The full BIDS tree is preprocessed unfiltered,
so nothing here reshapes a pipeline's inputs.
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

from network_qa.exclusions.base import list_generators
from network_qa.compile import compile_exclusions, write_lockfile, load_lockfile


def _cmd_compile(args: argparse.Namespace) -> None:
    # behavioral/qa_decisions scan the dataset via dataset_config["bids_dir"].
    # --generators selects a subset (None => all registered).
    dataset_config: dict = {}
    if getattr(args, "bids_dir", None):
        dataset_config["bids_dir"] = args.bids_dir
    lock = compile_exclusions(
        args.dataset, dataset_config, args, generator_names=args.generators
    )
    write_lockfile(lock, args.out)
    n = len(lock.get("exclusions", []))
    print(f"Compiled {n} exclusions for '{args.dataset}' -> {args.out}")






def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="network-qa", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    # compile
    comp_p = sub.add_parser("compile", help="Run registered generators -> provenance lockfile")
    comp_p.add_argument("--dataset", required=True, help="Dataset name (e.g. discovery)")
    comp_p.add_argument("--out", required=True, help="Path to write the lockfile JSON")
    comp_p.add_argument(
        "--generators", nargs="+", default=None,
        help="subset of registered generators to run; default all",
    )
    comp_p.add_argument(
        "--bids-dir", default=None,
        help="BIDS dataset root (needed by generators that scan the tree: "
             "behavioral / qa_decisions)",
    )
    for gen in list_generators().values():
        gen.add_cli_args(comp_p)
    comp_p.set_defaults(func=_cmd_compile)


    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
