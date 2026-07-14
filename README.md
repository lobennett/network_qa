# network_qa

Quality-assurance metrics and decisions for the **r01network** neuroimaging
pipeline. A small, pure, reusable package that other repos import — chiefly the
`network_fmri` orchestrator, which wires QA verdicts into its data-selection layer
(`bids-filter-file` + `scans.tsv`). Kept free of Flywheel / orchestration so it
stays testable and reusable.

## What's here

| module | role |
|--------|------|
| `qa_runs` | flag short functional runs by per-task cohort-mean volume count (`flag_short_runs` is pure + unit-tested; `scan_bold_volumes` reads NIfTI headers) → CLI `nf-qa-runs` (also reachable as `network-qa qa-runs`) |
| `exclusions.base` | generator registry (`register_generator`/`get_generator`/`list_generators`) + provenance helpers (`make_meta`, `_git_sha`) |
| `exclusions.behavioral` | wraps `network_events.qc.run_qc` (accuracy/RT/omission thresholds) as a generator |
| `exclusions.motion` | reads `motion_qa`'s `motion_metrics.tsv`, applies study FD/DVARS thresholds |
| `exclusions.lev1_outlier` | auto-excludes scans flagged by cohort lev1 QC (vif/outlier-pct rules); dormant until `network_glm` produces its input CSV |
| `exclusions.qa_decisions` | expands a manually-reviewed decisions TSV (`decisions.py`) into per-scan exclusions |
| `compile` | runs the registered generators, merges + dedupes, and writes a provenance-stamped lockfile |
| `render` | turns a compiled lockfile into the 3 data-selection channels: bids-filter-file, scans.tsv, .bidsignore |

**`nf-qa-runs`** — scan a BIDS tree, compute each task's mean volume count across
the whole cohort, and flag any run under a fraction (default 0.5) of that mean:

```bash
nf-qa-runs /path/to/bids                 # TSV report to stdout
nf-qa-runs /path/to/bids --frac 0.5 --out qa_runs.tsv
```

Clear aborts fall out; two near-complete runs of a task both survive. Output feeds
the processing-selection layer; a human can override.

## Exclusions integrator

`network_qa` is the study's exclusion **integrator**: it runs the registered
exclusion generators, compiles their output into one provenance-stamped
lockfile, and renders that lockfile into the three data-selection channels an
orchestrator (`network_fmri`) needs.

```bash
# 1. Compile every registered generator's output into a lockfile.
#    (each generator's CLI flags are attached to this same subcommand --
#    e.g. --motion-metrics-tsv, --decisions-tsv, --lev1-outliers-csv)
network-qa compile --dataset discovery --out lock.json \
    --motion-metrics-tsv /path/to/motion_metrics.tsv \
    --decisions-tsv config/manifests/qc_decisions.tsv

# 2. Render the 3 channels from that lockfile.
network-qa render bids-filter --anat-acquisition SagMPRAGE \
    --task goNogo --task nBack --task stroop --out fmriprep_filter.json

network-qa render scans-tsv --lockfile lock.json --bids-dir /path/to/bids

network-qa render bidsignore --lockfile lock.json --out /path/to/bids/.bidsignore
```

`bids-filter-file` is coarse and config-driven (canonical anat acquisition +
the task set a pipeline runs) — pybids filters can't express per-scan quality
exclusions, so those are NOT in it. `scans.tsv` carries the per-scan "why".
`.bidsignore` holds only genuinely-invalid scans (`source == "invalid"`);
everything else that's excluded for quality reasons is enforced downstream,
at lev1, by reading the lockfile directly.

The operational compile against real cohorts is gated on upstream inputs that
don't exist yet: `motion` needs fMRIPrep run on `discovery_v2`, and
`lev1_outlier` needs `network_glm`'s lev1 QC output. This package builds and
unit-tests the compile/render machinery on synthetic data; wiring
`network_fmri` to invoke it, and running it for real, are follow-ups.

## Roadmap (consolidation)

This package is intended to absorb the QA scattered elsewhere so there's one QA
surface for the orchestrator to import: FreeSurfer metrics, output-presence
checks, and the reliability-movie generation (the standalone
`bold-reliability-movies`).

## Setup (Sherlock)

`$HOME` is quota'd — keep the venv/cache on `$SCRATCH`:

```bash
module load uv
export UV_PROJECT_ENVIRONMENT=$SCRATCH/network_qa_venv
export UV_CACHE_DIR=$SCRATCH/uv_cache
uv sync && uv run pytest
```
