# Contributing

Shared conventions for the `network_*` repos (`network_fmri`, `network_events`, `network_glm`,
`network_qa`). This file is identical in each.

## Setup

```bash
uv sync
uv run pytest -q
```

On Sherlock, run both on a compute node (`sh_dev` or `sbatch`), never the login node.

Two things that will waste your time otherwise:

- **`uv run pytest`, not bare `pytest`.** Bare pytest can import an installed wheel from a venv
  instead of your working tree, so you end up testing the old code and debugging a fix that is
  already correct.
- **`ml load devel gcc/12.4.0` first.** CentOS 7's libstdc++ is too old for current numpy wheels,
  which fail at import with `CXXABI_1.3.9 not found`.

Use `uv sync`, never `uv pip install` — the latter resolves outside the lock and will silently
drift a pinned dependency.

## How the repos fit together

`network_fmri` is the launcher and pins the other three at immutable commits in
`[tool.uv.sources]`, so one `uv sync` provisions everything and every stage runs in the same venv.
A change to a dependency therefore needs three steps:

1. commit and push it there;
2. bump the `rev` in every repo that pins it (`network_qa` also pins `network_events`);
3. `uv lock && uv sync` in `network_fmri`.

Pins are commit SHAs, not branches, so a rebuild is reproducible.

## Changes

Keep them small and test-first. Anything that changes what lands in a BIDS tree, an
`events.tsv`, or an exclusion lockfile should come with a test — these failures are silent, they
produce no error and no visible artefact, only wrong models.

Commands stay pure and idempotent so an operator can wrap them in `datalad run`.
