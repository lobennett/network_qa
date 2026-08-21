"""Tests for the network-qa CLI (compile + render subcommands)."""
import json

import nibabel as nib
import numpy as np

from network_qa import cli


def test_cli_compile_writes_lockfile(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "compile_exclusions",
                        lambda *a, **k: {"_meta": {"dataset": "discovery"}, "exclusions": []})
    out = tmp_path / "lock.json"
    cli.main(["compile", "--dataset", "discovery", "--out", str(out),
              "--mriqc-dir", "/x"])
    assert json.loads(out.read_text())["_meta"]["dataset"] == "discovery"


def test_cli_compile_passes_generators_and_bids_dir(tmp_path, monkeypatch):
    """`--generators` and `--bids-dir` reach compile_exclusions: the subset of
    names is forwarded verbatim and bids_dir lands in dataset_config."""
    captured = {}

    def _fake(dataset, dataset_config, args, generator_names=None):
        captured["dataset"] = dataset
        captured["dataset_config"] = dataset_config
        captured["generator_names"] = generator_names
        return {"_meta": {"dataset": dataset, "generators": generator_names},
                "exclusions": []}

    monkeypatch.setattr(cli, "compile_exclusions", _fake)
    out = tmp_path / "lock.json"
    cli.main(["compile", "--dataset", "discovery", "--out", str(out),
              "--generators", "motion", "behavioral",
              "--bids-dir", str(tmp_path / "bids")])
    assert captured["generator_names"] == ["motion", "behavioral"]
    assert captured["dataset_config"]["bids_dir"] == str(tmp_path / "bids")


def _write_bold(bids_dir, subject, session, task, run, n_vols):
    func = bids_dir / subject / session / "func"
    func.mkdir(parents=True, exist_ok=True)
    name = f"{subject}_{session}_task-{task}_run-{run}_bold.nii.gz"
    img = nib.Nifti1Image(np.zeros((2, 2, 2, n_vols), dtype=np.int16), np.eye(4))
    nib.save(img, str(func / name))


def test_cli_compile_subset_runs_only_named_generators(tmp_path):
    """End-to-end: `compile --generators short_run behavioral` runs ONLY those
    two — it succeeds without the motion/lev1_outlier/qa_decisions inputs those
    generators would otherwise require (motion Path(None)/lev1 FileNotFoundError),
    proving they were not invoked. behavioral
    no-ops (no sourcedata)."""
    bids = tmp_path / "bids"
    _write_bold(bids, "sub-s01", "ses-01", "stroop", "1", 100)
    _write_bold(bids, "sub-s01", "ses-01", "stroop", "2", 100)
    _write_bold(bids, "sub-s02", "ses-01", "stroop", "1", 40)   # aborted -> short-run

    out = tmp_path / "lock.json"
    cli.main(["compile", "--dataset", "discovery", "--out", str(out),
              "--generators", "motion", "behavioral",
              "--bids-dir", str(bids)])

    lock = json.loads(out.read_text())
    # The named subset is what ran; unnamed generators contributed nothing.
    assert lock["_meta"]["generators"] == ["motion", "behavioral"]
    assert {e["source"] for e in lock["exclusions"]} <= {"motion", "behavioral"}





