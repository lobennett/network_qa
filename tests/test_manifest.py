import hashlib
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


@pytest.mark.parametrize(
    "key_args",
    (
        ("acquisition", "s01", "ses-01", "anat", "T1w"),
        ("acquisition", "sub-s01", "session-01", "anat", "T1w"),
        ("acquisition", "sub-s01", "ses-01", "func", "bold", "nBack"),
        ("acquisition", "sub-s01", "ses-01", "func", "T1w", "nBack", "", "", "1"),
        ("missing_expected", "sub-s01", "", "func", "bold"),
        ("missing_expected", "sub-s01", "ses-01", "anat", "T2w"),
    ),
)
def test_manifest_rejects_noncanonical_or_contradictory_identities(key_args):
    with pytest.raises(ValueError, match="identity"):
        DecisionRow.clean(AcquisitionKey(*key_args))


def test_manifest_rejects_subject_alias_before_duplicate_detection():
    with pytest.raises(ValueError, match="subject identity"):
        AcquisitionKey("acquisition", "s01", "ses-01", "func", "bold", task="nBack", run="1")


def test_manifest_rejects_duplicate_acquisitions_while_reading(tmp_path):
    path = tmp_path / "scan_decisions.tsv"
    write_manifest(path, [DecisionRow.clean(functional_key("sub-s01", "ses-01", "nBack", "1"))])
    contents = path.read_text()
    path.write_text(contents + contents.splitlines()[1] + "\n")

    with pytest.raises(ValueError, match="duplicate acquisition"):
        read_manifest(path)


def test_manifest_checksum_is_hash_of_exact_written_bytes(tmp_path):
    path = tmp_path / "scan_decisions.tsv"

    digest = write_manifest(path, [DecisionRow.clean(anatomical_key("sub-s01", "ses-01", "T1w"))])

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_is_invariant_to_row_and_semantic_set_order(tmp_path):
    first = replace(
        DecisionRow.clean(functional_key("sub-s01", "ses-01", "nBack", "1")),
        expected_echoes=(1, 2, 3),
        observed_echoes=(1, 3),
        missing_echoes=(2,),
        flags=("missing_echo", "motion"),
    )
    permuted = replace(
        first,
        expected_echoes=(3, 1, 2),
        observed_echoes=(3, 1),
        missing_echoes=(2,),
        flags=("motion", "missing_echo"),
    )
    anatomical = DecisionRow.clean(anatomical_key("sub-s01", "ses-01", "T1w"))
    first_path = tmp_path / "first.tsv"
    second_path = tmp_path / "second.tsv"

    first_digest = write_manifest(first_path, [first, anatomical])
    second_digest = write_manifest(second_path, [anatomical, permuted])

    assert first_digest == second_digest
    assert first_path.read_bytes() == second_path.read_bytes()


@pytest.mark.parametrize("run", ("01", "000", "run-1", "one", "1a"))
def test_manifest_rejects_noncanonical_functional_run_entities(run):
    with pytest.raises(ValueError, match="run identity"):
        functional_key("sub-s01", "ses-01", "nBack", run)


def test_manifest_rejects_padded_run_alias_before_duplicate_detection():
    canonical = functional_key("sub-s01", "ses-01", "nBack", "1")

    assert canonical.run == "1"
    with pytest.raises(ValueError, match="run identity"):
        functional_key("sub-s01", "ses-01", "nBack", "01")


def test_read_manifest_rejects_padded_run_alias(tmp_path):
    path = tmp_path / "scan_decisions.tsv"
    write_manifest(path, [DecisionRow.clean(functional_key("sub-s01", "ses-01", "nBack", "1"))])
    rows = path.read_text().splitlines()
    columns = rows[0].split("\t")
    values = rows[1].split("\t")
    values[columns.index("run")] = "01"
    rows[1] = "\t".join(values)
    path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="run identity"):
        read_manifest(path)
