# Contributing to network_qa

## Development setup: container + uv venv overlay

**Recommended setup** for extending a feature, adding a dependency, or otherwise
hacking on the package. The technique is the STAMPED *container venv overlay*
([example](https://examples.stamped-principles.org/examples/container-venv-overlay-development/)):
a **pinned container** provides the frozen heavy environment (pandas/nibabel + the
QA tool deps), and a **uv venv created with `--system-site-packages`** overlays
your *editable* checkout on top of it. You edit code on the host, it's live inside
the container immediately, and the container is never rebuilt — so the environment
stays reproducible while iteration is instant.

```bash
# One-time: create the overlay venv INSIDE the container (venv on $SCRATCH persists
# across runs; the container image stays immutable).
apptainer exec --cleanenv \
  -B "$PWD":/work -B "$SCRATCH":"$SCRATCH" --pwd /work \
  <base_container.sif> \
  bash -lc '
    uv venv --system-site-packages "$SCRATCH/nqa_dev_venv"   # overlay = container site-packages + editable pkg
    . "$SCRATCH/nqa_dev_venv/bin/activate"
    uv pip install -e .
  '

# Each dev run thereafter (no reinstall, no rebuild):
apptainer exec --cleanenv -B "$PWD":/work -B "$SCRATCH":"$SCRATCH" --pwd /work \
  <base_container.sif> \
  bash -lc '. "$SCRATCH/nqa_dev_venv/bin/activate" && python -m pytest -q'
```

`<base_container.sif>` is any container carrying the pinned heavy deps (a
scientific-python base, or a purpose-built image). Edit → the editable install
reflects it instantly → re-run in the container.

## Tests

```bash
python -m pytest -q
```

Keep changes small and test-first (TDD). network_qa integrates QA metrics +
decisions into the study exclusion set; new exclusion sources should come with a
generator test.
