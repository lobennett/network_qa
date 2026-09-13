# Package audit — 2026-09-13

Audited baseline: `d401e5f8dfa0f2feef27b1b470a8f06347691f2c`. This review read every
production module, the CLI, README/CONTRIBUTING contracts, existing tests, and dependency
manifest/lock. Corrections address software validation, identity and provenance; scientific
thresholds, behavioral-performance policy and the package/Slurm architecture are unchanged.

Read-only integration references were the managed `network_fmri` clone at
`4216d522729d26651abe96e968b29403a6211751` and `network_glm` at
`122fa29cd11e724a6cfc8160841bcfc537bb2c82`. No participant data, original research clone,
production checkout, campaign state or remote computation was changed.

## Findings and counterexamples

The executable cases live in
[`test_audit_regressions.py`](../tests/exclusions/test_audit_regressions.py).
Expectations below follow the input contracts and actual GLM consumers, not newly selected
exclusion criteria. Each correction was reproduced before implementation. Replaying the
final regression file against baseline source gives **57 failures and 17 passing controls**;
the corrected implementation passes all 74 cases.

| Path | Trigger and observed baseline symptom | Correction, proven path and disconfirming case |
|---|---|---|
| `compile.py`, selected motion | `--generators motoin` overwrote an existing lock with zero exclusions. Selecting motion without its directory also succeeded. API `[]` selected every registered generator. | Reject unknown names before execution; require motion's input only when selected, as lev1/manual generators already do. `None` selects all, `[]` selects none. A behavioral-only CLI compile still needs no motion inputs. Old tests had confused selecting motion without input with omitting motion from the subset. |
| Motion evidence | Broken JSON, absent/NaN `fd_perc`, or missing/NaN recorded `fd_thres` could yield a successful lock. A nested zero threshold could fall through to a different top-level threshold. | Reject unreadable IQMs and invalid scored metrics; validate the recorded threshold without truthiness fallback. The existing explicit mismatch guard remains. Valid single acquisitions still score; rest uses mean FD and tasks use percentage FD; DVARS remains evidence only. |
| Motion identity | Two files with the same subject/session/task/run silently selected the last path: a clean copy masked an over-threshold acquisition. | Reject ambiguous duplicate acquisitions, including acquisition-label collisions in the four-entity lock schema. A single high-FD file excludes; ordinary echo-1/2/3 input still yields one acquisition, and zero-padded echo labels select the same acquisition unpadded ones do rather than dropping it unscored. Subject filtering happens before duplicate/evidence validation. |
| Roster handling | An empty or missing configured roster disabled filtering; motion ignored `subjects_file` entirely. This admitted out-of-cohort exclusion entries. | A configured roster that names no subject is an operator error, not a zero-subject cohort: both it and a missing file raise; motion also honors the file. Each generator is exercised with empty, missing and one-member rosters. Existing no-roster behavior remains unrestricted. Motion's `subjects` set is normalized to the same `sub-` entity form before it is intersected with the roster, and a selection naming no subject raises instead of compiling an empty lock. |
| Manual decisions | Blank/partial scan fields could broaden a row into a subject decision; duplicate decisions depended on spelling/order. Subject expansion missed multi-echo/acquisition filenames, and a misfiled BOLD could exclude another subject. | Require complete scan identity or explicit `-/-/-`; reject normalized duplicate identities that disagree on the action, and join the distinct reasons of those that agree. Expand supported acquisition/echo variants, default absent run to 1, and verify directory identity. Identical decisions and echo files compile to one acquisition; other subjects/sessions remain unaffected. |
| Numeric run identity | QA emitted `run-01`, while GLM compared `run-1`. A supposedly excluded run remained in fixed effects. | All built-in generators emit unpadded numeric run entities, preserving subject/session labels. The real sibling integration fails on padded runs before correction and passes for both padded/unpadded inputs afterward. Queries remain exact; no loader rewrite or session relabeling was introduced. |
| Tables and lock reader | Empty/headerless files looked like valid empty evidence. Truncated CSV rows -- including truncation past the required columns, which crashed -- and arbitrary numeric text could become zero scores; blank run IDs made unusable exclusions. Non-list lock payloads were returned as though they were exclusion collections. | Validate required table columns, row shape and outlier identity; reject invalid scored numeric text and non-list/non-object lock structures. Header-only schema-valid tables remain empty. Empty/NaN lev1 metrics still score as zero; positive infinite VIF still fires the existing strict rule. |
| Thresholds and provenance | `nan` threshold arguments silently disabled comparisons. A wheel nested in an application's Git checkout could claim that application's commit. | Validate finite nonnegative thresholds and fraction/percentage domains. Git provenance is used only when its top-level directory equals QA's inferred source root; installed distribution and explicit environment fallbacks remain. A real temporary Git repository tests both nested-wheel rejection and source-root acceptance. |
| Dependency manifest | `nibabel`, `pandas` and `network_events` were declared as install-time requirements of a package that imports none of them; only the optional sibling integration module needs `nibabel`/`numpy`. | Declare no runtime dependencies and move `nibabel`/`numpy` to the `dev` group, refreshing the lock without unrelated upgrades. The built wheel still installs and its `network-qa` CLI still compiles a lock in an environment with no scientific stack. Coordinated with `network_fmri`'s queued QA pin/lock publication, which pins `network_events` directly. |
| Local-cache packaging | Building with `--cache-dir .uv-cache` included the build environment in the source archive; wheel-from-sdist failed on its external Python symlink. | Explicitly ignore `.uv-cache/` at the repository root. The same build then produces both distributions; archive inspection confirms the cache is absent. This keeps isolated-worktree builds self-contained. |

Boundary controls exercise rest FD `0.2` versus `0.20001`, task FD percentage `20` versus
`20.001`, behavioral retention exactly `0.5` versus `0.51`, combined VIF/outlier percentages
at and immediately below `10/10`, and strict VIF/outliers at and below `15`.
Motion and behavioral comparisons remain strict `>`; lev1 rules remain inclusive `>=`.
Default ignored contrasts and configurable overrides retain existing test coverage.

## Final-lock propagation

[`test_lock_consumers.py`](../tests/integration/test_lock_consumers.py) imports fMRI's
actual `STAGES` definitions and runs QA's real CLI for both motion and final compiles.
It passes the serialized locks to GLM's actual exclusion loader, computes fixed effects
from synthetic NIfTI effect/variance/Z maps, and runs the real lev2 file discovery.

For one synthetic subject, three runs become two after motion QC (mean effect **6**),
then one after a VIF of **15** (effect **9**). The final refresh retires the previously
eligible fixed-effects map and writes the below-minimum-runs variant, so lev2 drops the
subject. The unaffected subject remains eligible with mean effect **13/3**. The test runs
with both padded and unpadded run indices. Baseline passed the unpadded case but retained
all three runs in the padded case. Both now pass.

Read-only inspection of `network_fmri/qa/exclusions.py` and `workflow.py` confirms the final
compile is followed by `level1-finalize` with the final lock, then lev2. The integration
does not submit Slurm jobs, fit time-series GLMs, run permutations or validate surface
arithmetic. The two synthetic image cases verify the concrete consumer boundary.

## Validation and reproduction

Baseline package suite: **93 passed, 1 skipped**. Corrected frozen package suite:
**167 passed, 2 skipped**. The skips are the existing unavailable participant-data test
and the optional sibling integration module. Running the latter with the managed sibling
sources and GLM's frozen environment gives **2 passed**.
`uv build --cache-dir .uv-cache --out-dir .pytest_cache/dist` also passes. An extracted-wheel
CLI smoke test inside the worktree produces an empty behavioral lock with `dist:0.1.0`
provenance, rather than the enclosing checkout's commit.

```bash
uv sync --frozen
uv run --frozen pytest -q

# Optional integration: use an environment provisioned from GLM's uv.lock.
# Set these variables to local checkouts; imports are read-only.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/src:$GLM_REPO/src:$FMRI_REPO/src" \
MPLCONFIGDIR="$PWD/.pytest_cache/mpl" \
"$GLM_PYTHON" -m pytest tests/integration -q --basetemp .pytest_cache/integration
```

Local validation used `--cache-dir .uv-cache` and worktree-local pytest temporary
directories. The manifest and frozen dependency lock were preserved. No configured
lint/type-check or GitHub CI workflow exists in this package at baseline. Publication
review and shipping are delegated to Firstmate's no-mistakes stage after this commit.

## Practical limits and preserved policy

- This is synthetic software evidence, not a cohort reanalysis or proof of complete QC.
  A valid empty directory/table remains valid; missing individual acquisitions are not
  reconciled against a required inventory. Unsupported filenames are still skipped.
- Behavioral missing/unreadable sidecars and absent dropped-fraction metrics still yield
  no exclusion. Accuracy, RT and omission QC remain unimplemented. Restoring those rules,
  or requiring evidence before eligibility, needs a separate study-policy decision.
- Lev1 empty/NaN metrics remain unscored; ignored contrasts remain unscored. Singular VIF
  can still serialize as Python JSON `Infinity`, accepted by the current GLM reader but
  not strict JSON consumers. This audit does not introduce a replacement metric policy.
- Lockfiles record package identity, source, reason and exclusion metrics, not hashes of
  every input or a complete inventory of passing/missing acquisitions. `--dataset` is a
  metadata label; CLI callers must point at the intended cohort's evidence tree.
- The lock schema represents subject/session/task/run, not acquisition or echo as
  independent cohort dimensions. Ambiguous MRIQC acquisitions now require reconciliation
  instead of choosing one silently. Schema expansion would require coordinated consumers.
- Existing locks are not rewritten. After publication, `network_fmri` needs its immutable
  QA pin/lock updated and appropriate locks/fixed effects regenerated through its normal
  workflow. Deployment and participant recomputation were outside this worktree's scope.
