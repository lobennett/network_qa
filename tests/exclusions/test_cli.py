"""Tests for the network-qa CLI (compile + render subcommands)."""
import json
from network_qa import cli


def test_cli_compile_writes_lockfile(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "compile_exclusions",
                        lambda *a, **k: {"_meta": {"dataset": "discovery"}, "exclusions": []})
    out = tmp_path / "lock.json"
    cli.main(["compile", "--dataset", "discovery", "--out", str(out),
              "--motion-metrics-tsv", "/x.tsv"])
    assert json.loads(out.read_text())["_meta"]["dataset"] == "discovery"


def test_cli_render_bids_filter(tmp_path):
    out = tmp_path / "fmriprep.json"
    cli.main(["render", "bids-filter", "--anat-acquisition", "SagMPRAGE",
              "--task", "goNogo", "--task", "nBack", "--out", str(out)])
    assert json.loads(out.read_text())["bold"]["task"] == ["goNogo", "nBack"]


def test_cli_render_scans_tsv(tmp_path):
    lockfile = tmp_path / "lock.json"
    lockfile.write_text(json.dumps({
        "_meta": {"dataset": "discovery"},
        "exclusions": [
            {"subject": "s03", "session": "05", "task": "task-rest", "run": "run-1",
             "reason": "High FD", "source": "motion"},
        ],
    }))
    bids_dir = tmp_path / "bids"
    cli.main(["render", "scans-tsv", "--lockfile", str(lockfile), "--bids-dir", str(bids_dir)])
    tsv = bids_dir / "sub-s03" / "ses-05" / "sub-s03_ses-05_scans.tsv"
    assert tsv.is_file()
    assert "High FD" in tsv.read_text()


def test_cli_render_bidsignore(tmp_path):
    lockfile = tmp_path / "lock.json"
    lockfile.write_text(json.dumps({
        "_meta": {"dataset": "discovery"},
        "exclusions": [
            {"subject": "s03", "session": "01", "task": "task-x", "run": "run-1",
             "reason": "aborted dim4=1", "source": "invalid"},
        ],
    }))
    out = tmp_path / ".bidsignore"
    cli.main(["render", "bidsignore", "--lockfile", str(lockfile), "--out", str(out)])
    assert "sub-s03/ses-01/func/sub-s03_ses-01_task-x_run-1_bold.nii.gz" in out.read_text()


def test_cli_qa_runs_subcommand_reachable(tmp_path, capsys):
    """The pre-existing qa_runs entrypoint stays reachable through the
    network-qa CLI too (it also keeps its own nf-qa-runs console script)."""
    bids_root = tmp_path / "bids"
    func = bids_root / "sub-s1" / "ses-01" / "func"
    func.mkdir(parents=True)
    import nibabel as nib
    import numpy as np
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 10)), np.eye(4))
    nib.save(img, str(func / "sub-s1_ses-01_task-rest_run-1_bold.nii.gz"))

    cli.main(["qa-runs", str(bids_root)])
    captured = capsys.readouterr()
    assert "n_volumes" in captured.out
