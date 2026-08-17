"""Tests for network_qa.render — the 3 data-selection channels (NEW, no
monolith equivalent exists: neuro_workflow only has a docs-markdown renderer
(scripts/render_exclusions_md.py) tested by tests/scripts/test_render_exclusions_md.py,
not a .bidsignore/scans.tsv/bids-filter renderer)."""
import json
from pathlib import Path
from network_qa.render import render_bids_filter, render_scans_tsv, render_bidsignore


def test_render_bids_filter_is_coarse():
    cfg = {"anat_acquisition": "SagMPRAGE", "tasks": ["goNogo", "nBack", "stroop"]}
    f = render_bids_filter(cfg)
    assert f == {"t1w": {"acquisition": "SagMPRAGE", "suffix": "T1w"},
                 "bold": {"task": ["goNogo", "nBack", "stroop"]}}


def test_render_scans_tsv_writes_why_per_session(tmp_path):
    entries = [{"subject": "s03", "session": "05", "task": "task-rest", "run": "run-1",
                "reason": "Resting FD mean (0.30) > 0.2", "source": "motion"}]
    render_scans_tsv(entries, tmp_path)
    tsv = tmp_path / "sub-s03" / "ses-05" / "sub-s03_ses-05_scans.tsv"
    assert tsv.is_file()
    txt = tsv.read_text()
    assert "filename\t" in txt and "why" in txt
    assert "func/sub-s03_ses-05_task-rest_run-1_bold.nii.gz" in txt
    assert "Resting FD mean" in txt


def test_render_scans_tsv_multiecho_lists_real_files(tmp_path):
    """Multi-echo scans must list the REAL echo filenames, not a bare
    ``_bold.nii.gz`` (which fails bids-validator SCANS_FILENAME_NOT_MATCH_DATASET)."""
    func = tmp_path / "sub-s03" / "ses-11" / "func"
    func.mkdir(parents=True)
    base = "sub-s03_ses-11_task-stopSignalWDirectedForgetting_run-1"
    for echo in (1, 2, 3):
        (func / f"{base}_echo-{echo}_bold.nii.gz").touch()
    # a decoy run-10 file must NOT be matched by run-1
    (func / "sub-s03_ses-11_task-stopSignalWDirectedForgetting_run-10_echo-1_bold.nii.gz").touch()
    entries = [{"subject": "sub-s03", "session": "ses-11",
                "task": "task-stopSignalWDirectedForgetting", "run": "run-1",
                "reason": "non-monotonic", "source": "behavioral-qc"}]
    render_scans_tsv(entries, tmp_path)
    txt = (tmp_path / "sub-s03" / "ses-11" / "sub-s03_ses-11_scans.tsv").read_text()
    for echo in (1, 2, 3):
        assert f"func/{base}_echo-{echo}_bold.nii.gz" in txt
    assert f"func/{base}_bold.nii.gz" not in txt          # no bare (echo-less) name
    assert "run-10" not in txt                            # run-1 didn't swallow run-10


def test_render_scans_tsv_falls_back_when_no_file(tmp_path):
    """No matching file on disk -> still record the why under the constructed name."""
    entries = [{"subject": "s03", "session": "05", "task": "task-rest", "run": "run-1",
                "reason": "missing", "source": "behavioral-qc"}]
    render_scans_tsv(entries, tmp_path)
    txt = (tmp_path / "sub-s03" / "ses-05" / "sub-s03_ses-05_scans.tsv").read_text()
    assert "func/sub-s03_ses-05_task-rest_run-1_bold.nii.gz" in txt


def test_render_bidsignore_invalid_only(tmp_path):
    entries = [
        {"subject": "s03", "session": "05", "task": "task-rest", "run": "run-1",
         "reason": "motion", "source": "motion"},                    # quality -> NOT bidsignore
        {"subject": "s03", "session": "01", "task": "task-x", "run": "run-1",
         "reason": "aborted dim4=1", "source": "invalid"},            # invalid -> bidsignore
    ]
    out = tmp_path / ".bidsignore"
    lines = render_bidsignore(entries, out)
    assert lines == ["sub-s03/ses-01/func/sub-s03_ses-01_task-x_run-1_bold.nii.gz"]
    assert out.read_text().strip() == lines[0]


def test_render_robust_to_prefixed_entities(tmp_path):
    """Regression: entries carry BIDS-prefixed subject/session (sub-s03/ses-05)
    -- the normal case now that all generators emit prefixed. render must NOT
    double-prefix them into garbage paths (sub-sub-s03/ses-ses-05/...).
    """
    # scans.tsv from a prefixed entry.
    prefixed = [{"subject": "sub-s03", "session": "ses-05", "task": "task-rest",
                 "run": "run-1", "reason": "High FD", "source": "motion"}]
    render_scans_tsv(prefixed, tmp_path)
    tsv = tmp_path / "sub-s03" / "ses-05" / "sub-s03_ses-05_scans.tsv"
    assert tsv.is_file(), "must NOT write sub-sub-s03/ses-ses-05"
    assert "func/sub-s03_ses-05_task-rest_run-1_bold.nii.gz" in tsv.read_text()
    assert not (tmp_path / "sub-sub-s03").exists()

    # bidsignore from a prefixed invalid entry.
    inv = [{"subject": "sub-s03", "session": "ses-01", "task": "task-x",
            "run": "run-1", "reason": "aborted", "source": "invalid"}]
    out = tmp_path / ".bidsignore"
    lines = render_bidsignore(inv, out)
    assert lines == ["sub-s03/ses-01/func/sub-s03_ses-01_task-x_run-1_bold.nii.gz"]
    assert "sub-sub-s03" not in out.read_text()
