"""Tests for network_qa.decisions — ported verbatim (import-repoint only) from
neuro_workflow/tests/qa/test_decisions.py."""
from network_qa.decisions import Decision, ScanKey, load_decisions


def _write_tsv(tmp_path, content):
    p = tmp_path / "decisions.tsv"
    p.write_text(content)
    return p


def test_load_decisions_scan_level(tmp_path):
    p = _write_tsv(tmp_path, (
        "subject\tsession\ttask\trun\taction\treason\n"
        "sub-s03\tses-11\tstopSignalWDF\t1\texclude\tnon-monotonic onsets\n"
    ))
    result = load_decisions(p)
    key = ScanKey("sub-s03", "ses-11", "stopSignalWDF", "1")
    assert key in result
    assert result[key].action == "exclude"
    assert result[key].reason == "non-monotonic onsets"


def test_load_decisions_subject_level(tmp_path):
    p = _write_tsv(tmp_path, (
        "subject\tsession\ttask\trun\taction\treason\n"
        "sub-s1351\t-\t-\t-\tpass\tvisually inspected\n"
    ))
    result = load_decisions(p)
    assert "sub-s1351" in result
    assert result["sub-s1351"].action == "pass"


def test_load_decisions_missing_file_returns_empty(tmp_path):
    result = load_decisions(tmp_path / "nonexistent.tsv")
    assert result == {}


def test_load_decisions_invalid_action_raises(tmp_path):
    p = _write_tsv(tmp_path, (
        "subject\tsession\ttask\trun\taction\treason\n"
        "sub-s03\t-\t-\t-\tnonsense\twhatever\n"
    ))
    import pytest
    with pytest.raises(ValueError, match="invalid action"):
        load_decisions(p)


def test_load_decisions_skips_blank_lines(tmp_path):
    p = _write_tsv(tmp_path, (
        "subject\tsession\ttask\trun\taction\treason\n"
        "\n"
        "sub-s03\tses-11\tstopSignalWDF\t1\texclude\twhy\n"
    ))
    result = load_decisions(p)
    assert len(result) == 1
