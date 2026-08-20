# network_qa

Compiles the r01network study's exclusions into one provenance-stamped lockfile. Pure and
orchestration-free: it reads evidence other stages produced and decides what to exclude, but never
runs a pipeline or touches Flywheel.

Normally invoked through `network_fmri qa-motion` / `qa-lev1`, which pin this package at a commit.

## Where it sits

Nothing is filtered before preprocessing — the full BIDS tree goes through fMRIPrep and MRIQC, and
exclusion happens at the point of use. So there are two compiles, each downstream of the step that
produces its evidence:

| Compile | Runs after | Generators | Gates |
|---|---|---|---|
| `qa-motion` | fMRIPrep | `motion`, `behavioral` | what enters lev1 |
| `qa-lev1` | `glm-outliers` | `motion`, `behavioral`, `lev1_outlier` | what enters lev2 |

```bash
network-qa compile --dataset discovery --generators motion behavioral \
    --bids-dir <bids> --out lock.json --motion-metrics-tsv <motion_metrics.tsv>
```

`glm-lev1 --exclusions-file lock.json` reads the result. The lockfile carries the package commit
and each entry's source and reason, so a model's exclusion set is traceable to the evidence.

## Generators

| Generator | Reads | Excludes a run when |
|---|---|---|
| `motion` | fMRIPrep's `motion_metrics.tsv` | FD/DVARS exceed the study thresholds |
| `behavioral` | `network_events`' `_desc-truncation.json` sidecars | truncation dropped more than half its test trials |
| `lev1_outlier` | `network_glm`'s `lev1_outliers.csv` | VIF or outlier-percentage rules flag it |
| `qa_decisions` | a hand-reviewed decisions TSV | a human said so |

`behavioral` is the decision half of a deliberate split: `network_events` truncates a run — at a
backward-clock glitch, or at the end of an aborted scan — and records what that cost, but makes no
exclusion decision. This is where the threshold is applied.

**Not implemented:** the accuracy / RT / omission criteria the monolith also applied. They lived in
`network_events.qc`, which was removed; the per-task thresholds survive as
`network_events.qc_globals` but the computation would need rewriting.

## Layout

```
src/network_qa/
  cli.py                    one subcommand: compile
  compile.py                run the generators, merge, dedupe, stamp provenance
  decisions.py              parse a hand-reviewed decisions TSV
  exclusions/base.py        generator registry + provenance helpers
  exclusions/motion.py      FD/DVARS
  exclusions/behavioral.py  trial retention after truncation
  exclusions/lev1_outlier.py  VIF / outlier percentage
  exclusions/qa_decisions.py  manual overrides
```

## Setup

```bash
uv sync
uv run pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep the venv and cache off `$HOME`:
`export UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/network_qa UV_CACHE_DIR=$SCRATCH/.uv`.
