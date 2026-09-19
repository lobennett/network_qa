import numpy as np
from nibabel import Nifti1Image, save

from network_qa.functional import inspect_functionals


def write_bold_group(
    bids_dir, *, task, echoes, subject="sub-s01", session="ses-01", run="1",
    acquisition="", direction="",
):
    """Write one logical BOLD acquisition with a hand-specified dim4 per echo."""
    func = bids_dir / subject / session / "func"
    func.mkdir(parents=True, exist_ok=True)
    entities = [subject, session, f"task-{task}"]
    if acquisition:
        entities.append(f"acq-{acquisition}")
    if direction:
        entities.append(f"dir-{direction}")
    entities.append(f"run-{run}")
    for echo, count in echoes.items():
        name = "_".join([*entities, f"echo-{echo}", "bold.nii.gz"])
        save(Nifti1Image(np.zeros((2, 2, 2, count)), np.eye(4)), func / name)


def write_task_counts(bids_dir, task, counts):
    for index, count in enumerate(counts, start=1):
        write_bold_group(bids_dir, task=task, run=str(index), echoes={2: count})


def test_missing_any_expected_echo_flags_review(tmp_path):
    write_bold_group(tmp_path, task="nBack", echoes={1: 100, 3: 100})

    evidence, = inspect_functionals(tmp_path)

    assert evidence.missing_echoes == (2,)
    assert "missing_echo" in evidence.flags


def test_short_scan_boundary_is_strict(tmp_path):
    write_task_counts(tmp_path, "nBack", [50, 100, 100])

    rows = inspect_functionals(tmp_path)

    assert "short_scan" not in rows[0].flags


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
