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
