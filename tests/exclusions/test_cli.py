"""Tests for the network-qa compile CLI."""
import json

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


def test_cli_compile_subset_runs_only_named_generators(tmp_path):
    """A behavioral-only compile needs no inputs for unselected generators."""
    bids = tmp_path / "bids"
    bids.mkdir()

    out = tmp_path / "lock.json"
    cli.main(["compile", "--dataset", "discovery", "--out", str(out),
              "--generators", "behavioral",
              "--bids-dir", str(bids)])

    lock = json.loads(out.read_text())
    # The named subset is what ran; unnamed generators contributed nothing.
    assert lock["_meta"]["generators"] == ["behavioral"]
    assert lock["exclusions"] == []
