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
| `qa-motion` | MRIQC | `motion`, `behavioral` | what enters lev1 |
| `qa-lev1` | `glm-outliers` | `motion`, `behavioral`, `lev1_outlier` | what enters lev2 |

```bash
network-qa compile --dataset discovery --generators motion behavioral \
    --bids-dir <bids> --out lock.json --mriqc-dir <mriqc derivatives>
```

Motion comes from MRIQC rather than fMRIPrep confounds, so the exclusion set is known
before preprocessing. `fd_perc` counts frames above whatever `--fd_thres` MRIQC ran with,
so the generator refuses IQMs whose recorded threshold is not the expected 0.5 mm rather
than applying the wrong cutoff. The study's old *proportion of std_dvars > 1.5* criterion is
not applied: MRIQC publishes mean `dvars_std`, not a proportion, and a mean-based substitute
excluded nothing FD had not already caught on either cohort (0 additional runs in 291
discovery and 2308 validation acquisitions). `dvars_std` is still recorded in each entry's
metrics as evidence.

`glm-lev1 --exclusions-file lock.json` reads the result. The lockfile carries the package commit
and each entry's source and reason, so a model's exclusion set is traceable to the evidence.
After `qa-lev1`, refresh subject fixed effects with the final lock before running lev2.
Built-in generators emit BIDS-prefixed identities with unpadded numeric runs (`run-1`),
matching GLM's keys; subject and session labels retain their original zeros.

For API callers, `generator_names=None` selects all registered generators and `[]`
selects none. Selected generators require their declared inputs: an unknown generator,
missing motion directory, unreadable IQM, invalid scored motion metric, or unverified
MRIQC FD threshold stops compilation. Ambiguous duplicate IQM acquisitions also stop
compilation; multi-echo motion still uses echo-1. Threshold arguments must be finite
and in their numeric domains.
Outlier/decision tables require their schema even when empty. Decisions use explicit `-`
in all three scan fields for subject-level scope; partial identities and duplicates
disagreeing on the action are errors, while duplicates that agree keep every distinct
reason. `pass` and `review` do not override another exclusion.

An empty lock is **not evidence of complete QC coverage**. Empty valid input tables or
directories can produce no exclusions, behavioral sidecars retain their documented
missing/unreadable fallback, and empty/NaN lev1 metrics remain unscored. For API callers,
all generators honor the intersection of `subjects` and `subjects_file`, normalizing
bare and prefixed subject IDs. A configured selection must name at least one subject;
a missing roster, empty selection or non-overlapping intersection is an error, even
when no generators are selected. Absent/`None` selectors leave the dataset unrestricted.
The CLI's `--dataset` names the provenance record; it does not select a roster.
See [the audit record](docs/CODE-REVIEW.md) for tested cases and remaining limits.

## Generators

| Generator | Reads | Excludes a run when |
|---|---|---|
| `motion` | MRIQC's IQMs (`fd_mean`, `fd_perc`) | rest mean FD, or too many task frames over 0.5 mm |
| `behavioral` | `network_events`' `_desc-truncation.json` sidecars | truncation dropped more than half its test trials |
| `lev1_outlier` | `network_glm`'s `lev1_outliers.csv` | VIF or outlier-percentage rules flag it |
| `qa_decisions` | a hand-reviewed decisions TSV | a human said so |

`behavioral` is the decision half of a deliberate split: `network_events` truncates a run — at a
backward-clock glitch, or at the end of an aborted scan — and records what that cost, but makes no
exclusion decision. This is where the threshold is applied.

**Not implemented:** a DVARS criterion (measured and dropped, see above), and the
accuracy / RT / omission criteria this study previously applied. They lived in
`network_events.qc`, which was removed; the per-task thresholds survive as
`network_events.qc_globals` but the computation would need rewriting.

## Layout

```
src/network_qa/
  cli.py                    one subcommand: compile
  compile.py                run the generators, merge, dedupe, stamp provenance
  decisions.py              parse a hand-reviewed decisions TSV
  exclusions/base.py        generator registry + provenance helpers
  exclusions/motion.py      MRIQC motion evidence
  exclusions/behavioral.py  trial retention after truncation
  exclusions/lev1_outlier.py  VIF / outlier percentage
  exclusions/qa_decisions.py  manual exclusions
```

## Setup

See [CONTRIBUTING.md](CONTRIBUTING.md#setup) for environment setup and test commands.
The [test workflow](.github/workflows/tests.yml) defines the standalone CI checks;
the [audit record](docs/CODE-REVIEW.md#validation-and-reproduction) records coverage,
measured results, data-dependent skips and optional consumer integration instructions.

## Scan-review manifests

The review workflow has a separate entry point:

```bash
network-qa decisions generate --bids-dir <bids> --mriqc-dir <mriqc> \
    --output <bids>/code/network_fmri/scan_decisions.tsv
```

This compiles the reviewed functional, motion, and anatomical evidence into one row
per logical acquisition, plus missing-expected anatomical rows. Every review flag
sets `decision=review` and `approval_required=yes`. Clean rows default to `keep`;
no generated row is human-approved. Anatomical recommendations remain nonbinding.
The legacy `compile` command and its exclusion policy are unchanged.

Canonical behavioral exceptions are read from
`sourcedata/behavioral/behavioral_exceptions.tsv`. A reviewed absence is evidence,
not a drop or a reason to require events. For other task runs, missing or unreadable
behavior, event files, conversion-error tables, and truncation sidecars require
review. Both nonmonotonic and scan-length trial-loss metrics are retained in the
metadata's `behavioral_evidence`, keyed by full acquisition identity. Positive trial
loss requires review; this command applies no automatic behavioral drop threshold.

The `.meta.json` sidecar records the exact manifest SHA-256, resolved input roots,
BIDS and MRIQC content inventories and their digests, and available source/package
commits and MRIQC version/input-commit evidence. Unknown provenance is JSON `null`,
not an inferred commit. The deterministic generation timestamp uses the source
commit time (or `null`); an orchestrator's milestone receipt owns the wall-clock
execution time. `approved_manifest_sha256` remains `null` until a separate approval
workflow seals the decisions.

`compiler.inventory_records` and `compiler.inventory_digest` expose the BIDS
inventory contract for approval validation. It covers raw subject files, canonical
sourcedata, `dataset_description.json`, `participants.tsv`, `participants.json`, and
`.bidsignore`. Derivatives and `code/` are excluded from that digest; MRIQC JSON,
HTML, and TSV files have a separate inventory. Symlink targets and readable content
are hashed, and unavailable content is recorded explicitly. Outputs within BIDS
must be under `code/`, and outputs cannot be placed inside the MRIQC evidence root.

Both outputs are staged before publication. Each replacement is atomic; a caught
publication failure restores the prior manifest. As two directory entries cannot
be replaced in a single filesystem operation, readers must verify the metadata's
manifest digest to detect a process interruption between replacements. Generation
is a serial workflow stage; concurrent writers are not supported.
