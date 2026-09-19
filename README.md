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
  cli.py                    compile and decisions generate/approve/validate
  compile.py                run the generators, merge, dedupe, stamp provenance
  compiler.py               compile scan-review evidence and generation provenance
  approval.py               verify review decisions and seal metadata checksums
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

The study expects echoes 1, 2, and 3. Motion values require an observed, trusted
echo 2, including in incomplete groups; a lone echo 1 or 3 supplies only volume-count
evidence. Malformed motion IQMs retain the acquisition with `malformed_iqm` and no
usable motion values. MRIQC reports must be readable, nonempty regular files;
symlinks to such files remain valid. Invalid reports require review and contribute
no report path.

Anatomical files with malformed or unsupported identity entities remain explicit
observations under their physical subject/session parent and T1w/T2w suffix. Their
synthetic acquisition label is `invalid` plus the SHA-256 of the BIDS-relative file
path, with `x` appended as needed to avoid any observed acquisition-key collision.
Each malformed physical path counts toward anatomical review, including distinct
`.nii` and `.nii.gz` paths. These rows carry `invalid_identity`,
`untrusted_identity`, and `anatomical_count`; they cannot supply MRIQC evidence or
a selection recommendation. An invalid physical subject/session parent stops
compilation because it cannot support a manifest identity.

Canonical behavioral exceptions are read from
`sourcedata/behavioral/behavioral_exceptions.tsv`. A reviewed absence is evidence,
not a drop or a reason to require events. Exception rows require nonempty
`reviewed_by` and `reviewed_at` as well as identity, reason, and detail. For other task runs, missing or unreadable
behavior, event files, conversion-error tables, and truncation sidecars require
review. Both nonmonotonic and scan-length trial-loss metrics are retained in the
metadata's `behavioral_evidence`, keyed by full acquisition identity. Positive trial
loss requires review; this command applies no automatic behavioral drop threshold.

The `.meta.json` sidecar records the exact manifest SHA-256, resolved input roots,
BIDS and MRIQC content inventories and their digests, and available source/package
commits and MRIQC version/input-commit evidence. Unknown provenance is JSON `null`,
not an inferred commit. MRIQC input-commit provenance includes deterministic per-IQM
coverage (`valid`, `missing`, `malformed`, or `unreadable`) and a conflict indicator.
The aggregate commit is populated only when every acquisition IQM in the MRIQC root
has the same valid full Git commit; unrelated JSON cannot supply that evidence.
The deterministic generation timestamp uses the source
commit time (or `null`); an orchestrator's milestone receipt owns the wall-clock
execution time. `generation_metadata_sha256` binds all generation fields.
`approved_manifest_sha256` and `approved_metadata_sha256` are both explicitly
`null` until approval seals the decisions.

`compiler.inventory_records` and `compiler.inventory_digest` expose the BIDS
inventory contract for approval validation. It covers raw subject files, canonical
sourcedata, `dataset_description.json`, `participants.tsv`, `participants.json`, and
`.bidsignore`. Derivatives and `code/` are excluded from that digest; MRIQC JSON,
HTML, and TSV files have a separate inventory. Symlink targets and readable content
are hashed, and unavailable content is recorded explicitly. Directory symlinks in
either evidence scope are rejected before inspection; ordinary git-annex file
symlinks remain supported. VCS administration directories (`.git`, `.hg`, `.svn`,
`.bzr`, `.jj`, `.pijul`, `_darcs`, `CVS`, `RCS`, `SCCS`, and `.fossil-settings`) are
pruned before traversal in BIDS and MRIQC evidence scopes. Public annex IQM/report
symlinks retain their literal targets and resolved-content hashes; internal object
paths are not separate evidence records. The same exclusions apply to review
evidence discovery, MRIQC provenance, and committed-path comparisons. Outputs within BIDS
must be under `code/`, and outputs cannot be placed inside the MRIQC evidence root.

Both outputs are staged before publication. Each replacement is atomic; a caught
publication failure restores the prior manifest. As two directory entries cannot
be replaced in a single filesystem operation, readers must verify the metadata's
manifest digest to detect a process interruption between replacements. Generation
is a serial workflow stage; concurrent writers are not supported.

### Review, approve, and validate

After generation, edit only `decision`, `approved`, `reason_code`, `reason_detail`,
`reviewer`, and `reviewed_at` in the TSV. Every flagged or approval-required row must
be resolved to `keep` or `drop`, with `approved=yes`, a nonempty explanation,
reviewer, and review timestamp. Clean rows may remain `keep` and `approved=no`.
Any drop, including a manually dropped clean row, requires explicit approval and a
controlled reason: `excessive_motion`, `anatomical_quality`,
`incomplete_acquisition`, `severe_artifact`, `duplicate_lower_quality`, `aborted_run`,
`missing_required_metadata`, or `other`. Recommendations never approve a row.

```bash
network-qa decisions approve \
    --manifest <bids>/code/network_fmri/scan_decisions.tsv \
    --metadata <bids>/code/network_fmri/scan_decisions.meta.json --bids-dir <bids>
network-qa decisions validate \
    --manifest <bids>/code/network_fmri/scan_decisions.tsv \
    --metadata <bids>/code/network_fmri/scan_decisions.meta.json --bids-dir <bids>
```

Both commands return JSON with `ok`, `errors` (an array of diagnostic strings), and
`manifest_sha256`. Exit status is 0 for success, 1 for blocked approval or invalid /
unavailable inputs, and 2 for argparse command-usage errors. API callers use
`approval.seal_approval(manifest, metadata, bids_dir)` and
`approval.validate_approval(manifest, metadata, bids_dir)`, returning `ApprovalResult`.

The gate reconstructs the complete generated baseline through the read-only
`compiler.collect_decision_evidence` contract. Its canonical TSV must reproduce the
generation `manifest_sha256`; metadata generation fields must match current
inputs after verifying and retaining the stored generation commit and timestamp.
A projection of all TSV columns except the six human review fields must
equal that reconstructed baseline. This distinguishes legitimate review edits from
a mismatched manifest/metadata pair after interrupted publication. Identities,
evidence, row order, flags, approval requirements, and recommendations cannot be
edited during review. Regenerate manifests from earlier versions that lack the
generation metadata checksum or either explicit approval checksum field.

Approval requires available BIDS and MRIQC inventory content, the same resolved
roots, a known source DataLad/Git commit with no uncommitted inventory changes,
and unchanged package commit/version provenance. The network_qa source checkout
must have a successfully verified clean Git status; command failure or timeout
leaves cleanliness unknown and blocks approval. `package_provenance_basis`
distinguishes a source checkout from a VCS-installed wheel whose Git commit is
recorded by PEP 610 and whose checkout cleanliness is not applicable. The MRIQC version must
be known, and every acquisition IQM must record the same valid `provenance.input_commit`,
ancestral to the current BIDS HEAD and bound to identical raw content. Unknown,
missing, malformed, or conflicting MRIQC input
provenance blocks sealing; the gate never fills it with the current BIDS HEAD.
MRIQC's own dataset commit may be null when it has no independent Git dataset;
its input commit and content inventory still bind the evidence.

BIDS file contents are compared directly to Git blobs in the source commit, so
ignore rules and index flags cannot conceal uncommitted evidence. A file symlink
must resolve to another inventoried, committed file, or to an annex object with a
matching `SHA256` / `SHA256E` key, size, and content checksum. Other external-link or
annex-key backends are unbound for this gate and require a separately reviewed
verification contract. Git-clean filters and unlocked annex pointer files are not
interpreted as raw-content provenance.

DataLad descendant saves for `mriqc-complete`, `scan-decisions-generated`, and
`scan-decisions-approved` are allowed when the covered raw and MRIQC inventories
remain unchanged. The stored generation source commit and timestamp stay immutable;
the gate verifies that the generation and MRIQC input commits are reachable
ancestors of current BIDS HEAD and that all three commit snapshots bind the current
raw content. A separately versioned MRIQC dataset may similarly advance through
descendant commits that preserve its covered evidence. Missing or divergent
commits, changed inventory content, and altered evidence still block approval.
The generation timestamp is checked against its stored source commit, not replaced
by the latest milestone time. Inventory reconstruction reads all covered content
and can take time on a full dataset.

`approve` atomically updates only `approved_manifest_sha256` and
`approved_metadata_sha256` in the sidecar. It never changes the manifest or BIDS
evidence. `validate` writes nothing and requires an existing seal. Any later TSV
edit invalidates the seal, including changes to explanations or whitespace; edits
to metadata or current evidence also fail validation. Repeating approval of an
unchanged sealed pair is a read-only success. A changed sealed TSV cannot be
resealed directly; regenerate and review a fresh pair. These checksums detect stale
or edited artifacts; they do not authenticate reviewer identity or provide a
cryptographic signature. Keep generation, review, approval, and curation serial.
