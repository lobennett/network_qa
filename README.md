# network_qa

`network-qa` compiles MRIQC, imaging, and behavioral evidence into a reviewable scan
decision file. It records evidence and recommendations; a reviewer makes the final
keep/drop decisions before preprocessing continues.

## Setup

```bash
uv sync
```

## Generate scan decisions

```bash
network-qa decisions generate \
  --bids-dir /path/to/bids \
  --mriqc-dir /path/to/mriqc \
  --output /path/to/bids/code/network_fmri/scan_decisions.tsv
```

The command reads:

- BOLD and anatomical files from the BIDS dataset;
- in-scanner behavior from `sourcedata/behavioral/in_scanner`;
- event-conversion evidence from `sourcedata/events_qc`;
- MRIQC image-quality and motion metrics.

It flags missing echoes, short scans, motion, missing behavior, and ambiguous anatomy.
It also writes `scan_decisions.meta.json` with input inventories and checksums.

## Approve decisions

After reviewing every row that requires approval:

```bash
network-qa decisions approve \
  --manifest /path/to/bids/code/network_fmri/scan_decisions.tsv \
  --metadata /path/to/bids/code/network_fmri/scan_decisions.meta.json \
  --bids-dir /path/to/bids
```

Use `decisions validate` with the same arguments before applying curation. Validation
fails if the manifest, metadata, or governed dataset has changed since approval.

The older `network-qa compile` command remains available for creating exclusion
lockfiles used by downstream models. Run `network-qa compile --help` for its inputs.

## Development

```bash
uv lock --check
uv run pytest -q
uv build
git diff --check
```
