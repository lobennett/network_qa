from dataclasses import replace

import pytest

from network_qa.manifest import (
    AcquisitionKey,
    DecisionRow,
    anatomical_key,
    functional_key,
    read_manifest,
    write_manifest,
)


def test_manifest_round_trip_preserves_blank_bids_entities(tmp_path):
    row = DecisionRow.clean(anatomical_key("sub-s01", "ses-01", "T1w"))

    digest = write_manifest(tmp_path / "scan_decisions.tsv", [row])

    assert read_manifest(tmp_path / "scan_decisions.tsv") == (row,)
    assert len(digest) == 64


def test_manifest_rejects_duplicate_keys(tmp_path):
    row = DecisionRow.clean(functional_key("sub-s01", "ses-01", "nBack", "1"))

    with pytest.raises(ValueError, match="duplicate acquisition"):
        write_manifest(tmp_path / "scan_decisions.tsv", [row, row])


def test_manifest_uses_contract_encodings_for_evidence(tmp_path):
    row = DecisionRow(
        key=AcquisitionKey(
            record_type="acquisition",
            subject="sub-s01",
            session="ses-01",
            datatype="func",
            suffix="bold",
            task="nBack",
            run="1",
        ),
        expected_echoes=(1, 2, 3),
        observed_echoes=(1, 3),
        missing_echoes=(2,),
        representative_echo=1,
        tr_count=100,
        original_tr_count=107,
        expected_tr_count_mean=110.5,
        tr_count_fraction=107 / 110.5,
        fd_mean=0.21,
        fd_perc=20.0,
        fd_thres=0.5,
        dvars_std=1.4,
        mriqc_report_path="derivatives/mriqc/report.html",
        behavioral_status="present",
        event_status="converted",
        flags=("missing_echo", "motion"),
        recommendation="review",
        recommendation_status="clear",
        recommendation_rationale="Echo 2 is absent.",
        decision="review",
        approval_required=True,
        approved=False,
        reason_code="",
        reason_detail="",
        reviewer="",
        reviewed_at="",
    )
    path = tmp_path / "scan_decisions.tsv"

    write_manifest(path, [row])

    assert "expected_echoes\tobserved_echoes" in path.read_text()
    assert "1,2,3\t1,3\t2" in path.read_text()
    assert "\tyes\tno\t" in path.read_text()
    assert read_manifest(path) == (row,)


def test_manifest_quotes_free_text_fields(tmp_path):
    row = replace(
        DecisionRow.clean(anatomical_key("sub-s01", "ses-01", "T1w")),
        reason_detail="Reviewer note\twith a newline\nfor context.",
    )
    path = tmp_path / "scan_decisions.tsv"

    write_manifest(path, [row])

    assert read_manifest(path) == (row,)


def test_manifest_round_trips_missing_expected_subject_modality_row(tmp_path):
    row = DecisionRow.clean(
        AcquisitionKey("missing_expected", "sub-s01", "", "anat", "T2w"),
    )
    path = tmp_path / "scan_decisions.tsv"

    write_manifest(path, [row])

    assert read_manifest(path) == (row,)
