# network_qa

Quality-assurance metrics and decisions for the **r01network** neuroimaging
pipeline. A small, pure, reusable package that other repos import — chiefly the
`network_fmri` orchestrator, which wires QA verdicts into its data-selection layer
(`bids-filter-file` + `scans.tsv`). Kept free of Flywheel / orchestration so it
stays testable and reusable.

## What's here

| module | role |
|--------|------|
| `qa_runs` | flag short functional runs by per-task cohort-mean volume count (`flag_short_runs` is pure + unit-tested; `scan_bold_volumes` reads NIfTI headers) → CLI `nf-qa-runs` |

**`nf-qa-runs`** — scan a BIDS tree, compute each task's mean volume count across
the whole cohort, and flag any run under a fraction (default 0.5) of that mean:

```bash
nf-qa-runs /path/to/bids                 # TSV report to stdout
nf-qa-runs /path/to/bids --frac 0.5 --out qa_runs.tsv
```

Clear aborts fall out; two near-complete runs of a task both survive. Output feeds
the processing-selection layer; a human can override.

## Roadmap (consolidation)

This package is intended to absorb the QA scattered elsewhere so there's one QA
surface for the orchestrator to import: cohort lev1-outlier detection, motion /
FreeSurfer metrics, output-presence checks (currently in `neuro_workflow.qa`), and
the reliability-movie generation (the standalone `bold-reliability-movies`).

## Setup (Sherlock)

`$HOME` is quota'd — keep the venv/cache on `$SCRATCH`:

```bash
module load uv
export UV_PROJECT_ENVIRONMENT=$SCRATCH/network_qa_venv
export UV_CACHE_DIR=$SCRATCH/uv_cache
uv sync && uv run pytest
```
