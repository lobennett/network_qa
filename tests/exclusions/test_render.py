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


# --- per-session bids-filter -------------------------------------------------
#
# The coarse per-pipeline filter can't express a per-scan quality call: every
# excluded scan's *task* stays in the task list, so fMRIPrep still sees it. That
# matters under BABS, where a truncated run that crashes fMRIPrep fails the whole
# subject-session job. BABS fans out per subject-session, so a per-(sub,ses)
# filter CAN drop a bad scan -- as long as every run of that task in that session
# is excluded (a filter dict has no per-task run selectivity).
#
# These filters SUPERSEDE the one BABS generates itself (babs emits its own
# `--bids-filter-file` before our args; ours wins by argparse), so they must carry
# babs's full session scoping -- fmap/sbref/t1w/t2w/flair/roi -- not just bold.

def _touch_bold(bids_dir: Path, sub: str, ses: str, task: str, run: str, echoes=(1, 2, 3)):
    func = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
    func.mkdir(parents=True, exist_ok=True)
    for e in echoes:
        (func / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_echo-{e}_bold.nii.gz").touch()


CFG = {"anat_acquisition": "SagMPRAGE", "tasks": ["goNogo", "nBack", "rest"]}


def test_per_session_filter_scopes_func_to_the_session(tmp_path):
    """No exclusions: full task list, func scoped to the session with a BARE
    session label (babs strips `ses-`; pybids entity values are bare). Anat is
    deliberately unscoped -- see
    test_per_session_filter_does_not_session_scope_anat."""
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s10", "01", "goNogo", "1")
    filters, residual = render_bids_filter_per_session(CFG, [], tmp_path)

    assert residual == []
    f = filters[("sub-s10", "ses-01")]
    assert f["bold"] == {"datatype": "func", "session": "01",
                         "suffix": "bold", "task": ["goNogo", "nBack", "rest"]}
    assert f["t1w"] == {"datatype": "anat", "suffix": "T1w",
                        "acquisition": "SagMPRAGE"}
    # babs's other keys are carried through, session-scoped
    assert set(f) == {"fmap", "bold", "sbref", "flair", "t2w", "t1w", "roi"}
    assert f["fmap"] == {"datatype": "fmap", "session": "01"}


def test_per_session_filter_drops_task_when_its_only_run_is_excluded(tmp_path):
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s19", "02", "goNogo", "1")
    _touch_bold(tmp_path, "s19", "02", "nBack", "1")
    entries = [{"subject": "sub-s19", "session": "ses-02", "task": "task-goNogo",
                "run": "run-1", "source": "short-run", "action": "exclude"}]

    filters, residual = render_bids_filter_per_session(CFG, entries, tmp_path)

    assert filters[("sub-s19", "ses-02")]["bold"]["task"] == ["nBack", "rest"]
    assert residual == []


def test_per_session_filter_keeps_task_when_a_sibling_run_survives(tmp_path):
    """s10 ses-01 goNogo has runs 1+2 and only run-1 is excluded. Dropping the
    task would discard the good run-2, so the task stays and the unexpressed
    exclusion is reported (never silently dropped)."""
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s10", "01", "goNogo", "1")
    _touch_bold(tmp_path, "s10", "01", "goNogo", "2")
    entries = [{"subject": "sub-s10", "session": "ses-01", "task": "task-goNogo",
                "run": "run-1", "source": "short-run", "action": "exclude"}]

    filters, residual = render_bids_filter_per_session(CFG, entries, tmp_path)

    assert "goNogo" in filters[("sub-s10", "ses-01")]["bold"]["task"]
    assert len(residual) == 1
    assert residual[0]["task"] == "task-goNogo"
    assert residual[0]["run"] == "run-1"


def test_per_session_filter_ignores_sources_not_requested(tmp_path):
    """behavioral-qc means the events log is defective, not the BOLD -- that's a
    lev1 call, so it must not remove a scan from preprocessing."""
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s10", "01", "goNogo", "1")
    entries = [{"subject": "sub-s10", "session": "ses-01", "task": "task-goNogo",
                "run": "run-1", "source": "behavioral-qc", "action": "exclude"}]

    filters, residual = render_bids_filter_per_session(
        CFG, entries, tmp_path, exclude_sources=("short-run",)
    )

    assert "goNogo" in filters[("sub-s10", "ses-01")]["bold"]["task"]
    # not requested => not an unexpressed exclusion
    assert residual == []


def test_per_session_filter_covers_every_session_on_disk(tmp_path):
    """Every subject-session gets a file, so any BABS job can read its own."""
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s10", "01", "goNogo", "1")
    _touch_bold(tmp_path, "s10", "02", "nBack", "1")
    _touch_bold(tmp_path, "s29", "11", "rest", "1")

    filters, _ = render_bids_filter_per_session(CFG, [], tmp_path)

    assert set(filters) == {("sub-s10", "ses-01"), ("sub-s10", "ses-02"),
                            ("sub-s29", "ses-11")}


def test_write_per_session_filters_names_files_for_babs(tmp_path):
    """Filenames must be reconstructible from babs's ${subid}/${sesid} shell vars
    (which keep their BIDS prefixes)."""
    from network_qa.render import write_bids_filter_per_session

    _touch_bold(tmp_path, "s10", "01", "goNogo", "1")
    written = write_bids_filter_per_session(
        CFG, [], tmp_path, pipeline="fmriprep", out_dir=tmp_path / "code"
    )

    expected = tmp_path / "code" / "bids-filter_fmriprep_sub-s10_ses-01.json"
    assert expected in written
    loaded = json.loads(expected.read_text())
    assert loaded["bold"]["session"] == "01"


def test_per_session_filter_excluded_task_absent_from_only_that_session(tmp_path):
    """A task excluded in ses-11 stays selected in ses-12."""
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s29", "11", "goNogo", "1")
    _touch_bold(tmp_path, "s29", "12", "goNogo", "1")
    entries = [{"subject": "sub-s29", "session": "ses-11", "task": "task-goNogo",
                "run": "run-1", "source": "short-run", "action": "exclude"}]

    filters, residual = render_bids_filter_per_session(CFG, entries, tmp_path)

    assert "goNogo" not in filters[("sub-s29", "ses-11")]["bold"]["task"]
    assert "goNogo" in filters[("sub-s29", "ses-12")]["bold"]["task"]


def test_per_session_filter_does_not_session_scope_anat(tmp_path):
    """Anat must NOT be session-scoped by default.

    Real-data constraint: in the discovery cohort only 7 of 61 sessions contain a
    SagMPRAGE T1w (anat is acquired sparsely; most sessions have just a CubePromo
    T2w). babs's own generated filter session-scopes t1w/t2w/flair/roi, which would
    leave 54/61 session-level jobs with no anat at all. Func + fmap stay
    session-scoped -- those really are per-session acquisitions.
    """
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s03", "12", "goNogo", "1")
    filters, _ = render_bids_filter_per_session(CFG, [], tmp_path)
    f = filters[("sub-s03", "ses-12")]

    for anat_key in ("t1w", "t2w", "flair", "roi"):
        assert "session" not in f[anat_key], f"{anat_key} must not be session-scoped"
    for ses_key in ("bold", "sbref", "fmap"):
        assert f[ses_key]["session"] == "12"


def test_per_session_filter_can_session_scope_anat_on_request(tmp_path):
    """Cohorts that do acquire anat every session can opt back in."""
    from network_qa.render import render_bids_filter_per_session

    _touch_bold(tmp_path, "s03", "12", "goNogo", "1")
    filters, _ = render_bids_filter_per_session(
        CFG, [], tmp_path, anat_scope="session"
    )
    assert filters[("sub-s03", "ses-12")]["t1w"]["session"] == "12"
