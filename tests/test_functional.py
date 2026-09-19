import numpy as np
from nibabel import Nifti1Image, save

from network_qa.functional import inspect_functionals


def write_bold_group(
    bids_dir, *, task, echoes, subject="sub-s01", session="ses-01", run="1",
    acquisition="", direction="", extension=".nii.gz", shape_prefix=(2, 2, 2),
    filename_subject=None, filename_session=None,
):
    """Write one logical BOLD acquisition with a hand-specified dim4 per echo."""
    func = bids_dir / subject / session / "func"
    func.mkdir(parents=True, exist_ok=True)
    entities = [filename_subject or subject, filename_session or session, f"task-{task}"]
    if acquisition:
        entities.append(f"acq-{acquisition}")
    if direction:
        entities.append(f"dir-{direction}")
    entities.append(f"run-{run}")
    for echo, count in echoes.items():
        name = "_".join([*entities, f"echo-{echo}", f"bold{extension}"])
        save(Nifti1Image(np.zeros((*shape_prefix, count)), np.eye(4)), func / name)


def write_task_counts(bids_dir, task, counts):
    for index, count in enumerate(counts, start=1):
        write_bold_group(bids_dir, task=task, run=str(index), echoes={2: count})


def write_unreadable_bold(bids_dir, *, task, run="1", echo=1):
    func = bids_dir / "sub-s01" / "ses-01" / "func"
    func.mkdir(parents=True, exist_ok=True)
    name = f"sub-s01_ses-01_task-{task}_run-{run}_echo-{echo}_bold.nii.gz"
    (func / name).write_bytes(b"not a NIfTI")


def test_missing_any_expected_echo_flags_review(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100, 3: 100})

    evidence, = inspect_functionals(tmp_path)

    assert evidence.missing_echoes == (2,)
    assert "missing_echo" in evidence.flags


def test_short_scan_boundary_is_strict_after_dummy_volumes(tmp_path):
    exact_boundary = tmp_path / "exact"
    just_below_boundary = tmp_path / "below"
    write_task_counts(exact_boundary, "nBack", [3, 23])
    write_task_counts(just_below_boundary, "nBack", [3, 24])

    exact_rows = inspect_functionals(exact_boundary)
    below_rows = inspect_functionals(just_below_boundary)

    assert exact_rows[0].original_tr_count == 10
    assert exact_rows[0].expected_tr_count_mean == 20.0
    assert "short_scan" not in exact_rows[0].flags
    assert below_rows[0].expected_tr_count_mean == 20.5
    assert "short_scan" in below_rows[0].flags


def test_inspection_uses_echo_two_and_task_level_original_count_mean(tmp_path):
    write_bold_group(tmp_path, task="nBack", run="1", echoes={1: 100, 2: 100, 3: 100})
    write_bold_group(tmp_path, task="nBack", run="2", echoes={1: 80, 2: 80, 3: 80})

    first, second = inspect_functionals(tmp_path)

    assert first.representative_echo == 2
    assert first.tr_count == 100
    assert first.original_tr_count == 107
    assert first.expected_tr_count_mean == 97.0
    assert first.tr_count_fraction == 107 / 97
    assert second.expected_tr_count_mean == 97.0


def test_unequal_echo_counts_are_reviewable_and_excluded_from_task_mean(tmp_path):
    write_bold_group(tmp_path, task="nBack", run="1", echoes={1: 100, 2: 90, 3: 100})
    write_bold_group(tmp_path, task="nBack", run="2", echoes={1: 110, 2: 110, 3: 110})

    unequal, complete = inspect_functionals(tmp_path)

    assert unequal.tr_count is None
    assert unequal.original_tr_count is None
    assert "unequal_echo_counts" in unequal.flags
    assert complete.expected_tr_count_mean == 117.0


def test_inspection_preserves_acquisition_and_direction_and_canonicalizes_run(tmp_path):
    write_bold_group(
        tmp_path, task="nBack", run="01", acquisition="mb4", direction="AP",
        echoes={1: 100, 2: 100, 3: 100},
    )

    evidence, = inspect_functionals(tmp_path)

    assert evidence.key.record_type == "acquisition"
    assert evidence.key.datatype == "func"
    assert evidence.key.suffix == "bold"
    assert evidence.key.task == "nBack"
    assert evidence.key.run == "1"
    assert evidence.key.acquisition == "mb4"
    assert evidence.key.direction == "AP"


def test_uncompressed_nifti_is_inspected(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100}, extension=".nii")

    evidence, = inspect_functionals(tmp_path)

    assert evidence.observed_echoes == (1,)
    assert evidence.tr_count == 100


def test_duplicate_compressed_and_uncompressed_echo_is_reviewable(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100}, extension=".nii")
    write_bold_group(tmp_path, task="nBack", echoes={1: 100}, extension=".nii.gz")

    evidence, = inspect_functionals(tmp_path)

    assert "ambiguous_echo" in evidence.flags
    assert evidence.tr_count is None


def test_unexpected_echo_is_reviewable_and_cannot_be_complete_multi_echo(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100, 2: 100, 3: 100, 4: 100})

    evidence, = inspect_functionals(tmp_path)

    assert evidence.observed_echoes == (1, 2, 3, 4)
    assert "unexpected_echo" in evidence.flags
    assert evidence.representative_echo is None


def test_lone_single_echo_is_the_representative_image(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100})

    evidence, = inspect_functionals(tmp_path)

    assert evidence.representative_echo == 1
    assert evidence.tr_count == 100


def test_lone_echo_two_is_the_representative_image(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={2: 100})

    evidence, = inspect_functionals(tmp_path)

    assert evidence.representative_echo == 2
    assert evidence.tr_count == 100


def test_incomplete_trusted_group_with_echo_two_uses_echo_two(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100, 2: 100})

    evidence, = inspect_functionals(tmp_path)

    assert evidence.missing_echoes == (3,)
    assert evidence.representative_echo == 2
    assert evidence.tr_count == 100


def test_incomplete_multi_echo_with_agreed_counts_has_no_echo_two_representative(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100, 3: 100})

    evidence, = inspect_functionals(tmp_path)

    assert evidence.tr_count == 100
    assert evidence.representative_echo is None


def test_functional_identity_separates_subject_session_acquisition_and_direction(tmp_path):
    write_bold_group(tmp_path, task="nBack", run="1", echoes={1: 100}, subject="sub-s01")
    write_bold_group(tmp_path, task="nBack", run="1", echoes={1: 100}, subject="sub-s02")
    write_bold_group(tmp_path, task="nBack", run="1", echoes={1: 100}, session="ses-02")
    write_bold_group(tmp_path, task="nBack", run="1", echoes={1: 100}, acquisition="mb4", direction="AP")

    rows = inspect_functionals(tmp_path)

    assert len(rows) == 4
    assert {
        (row.key.subject, row.key.session, row.key.acquisition, row.key.direction)
        for row in rows
    } == {
        ("sub-s01", "ses-01", "", ""),
        ("sub-s02", "ses-01", "", ""),
        ("sub-s01", "ses-02", "", ""),
        ("sub-s01", "ses-01", "mb4", "AP"),
    }


def test_task_means_use_the_exact_bids_task_label(tmp_path):
    write_bold_group(tmp_path, task="nBack", run="1", echoes={1: 100})
    write_bold_group(tmp_path, task="nBack", run="2", echoes={1: 80})
    write_bold_group(tmp_path, task="flanker", run="1", echoes={1: 30})

    rows = {row.key.task + row.key.run: row for row in inspect_functionals(tmp_path)}

    assert rows["nBack1"].expected_tr_count_mean == 97.0
    assert rows["nBack2"].expected_tr_count_mean == 97.0
    assert rows["flanker1"].expected_tr_count_mean == 37.0


def test_malformed_nifti_dimensionality_is_explicit_review_evidence(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100}, shape_prefix=(2, 2))

    evidence, = inspect_functionals(tmp_path)

    assert "invalid_nifti" in evidence.flags
    assert evidence.tr_count is None


def test_unreadable_nifti_is_explicit_evidence_and_cannot_affect_task_mean(tmp_path):
    write_unreadable_bold(tmp_path, task="nBack", run="1")
    write_bold_group(tmp_path, task="nBack", run="2", echoes={1: 100})

    malformed, valid = inspect_functionals(tmp_path)

    assert "invalid_nifti" in malformed.flags
    assert malformed.tr_count is None
    assert malformed.expected_tr_count_mean is None
    assert valid.expected_tr_count_mean == 107.0


def test_five_dimensional_nifti_is_explicit_review_evidence(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100}, shape_prefix=(2, 2, 2, 2))

    evidence, = inspect_functionals(tmp_path)

    assert "invalid_nifti" in evidence.flags
    assert evidence.tr_count is None
    assert evidence.expected_tr_count_mean is None


def test_zero_volume_nifti_is_explicit_review_evidence(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 0})

    evidence, = inspect_functionals(tmp_path)

    assert "invalid_nifti" in evidence.flags
    assert evidence.tr_count is None
    assert evidence.expected_tr_count_mean is None


def test_filename_parent_identity_mismatch_is_explicit_review_evidence(tmp_path):
    write_bold_group(
        tmp_path, task="nBack", echoes={1: 100},
        filename_subject="sub-s99", filename_session="ses-02",
    )

    evidence, = inspect_functionals(tmp_path)

    assert evidence.key.subject == "sub-s99"
    assert evidence.key.session == "ses-02"
    assert "identity_mismatch" in evidence.flags
