import json

import pytest

from network_qa.anatomical import inspect_anatomicals


def write_anatomical(
    bids_dir, *, suffix, subject="sub-s01", session="ses-01", acquisition="", run="1",
):
    anat = bids_dir / subject / session / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    entities = [subject, session]
    if acquisition:
        entities.append(f"acq-{acquisition}")
    if run:
        entities.append(f"run-{run}")
    path = anat / ("_".join([*entities, suffix]) + ".nii.gz")
    path.write_bytes(b"anatomical inventory only")
    return path


def write_iqm(mriqc_dir, image, **metrics):
    relative = image.relative_to(image.parents[3])
    path = mriqc_dir / relative.with_suffix("").with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics))
    return path


def write_report(mriqc_dir, image):
    path = mriqc_dir / (image.name.removesuffix(".nii.gz") + ".html")
    path.write_text("MRIQC report")
    return path


def valid_metrics(**overrides):
    return {
        "cjv": 0.5,
        "cnr": 2.0,
        "snr_total": 10.0,
        "efc": 0.4,
        "fber": 100.0,
        "qi_2": 0.02,
        "wm2max": 0.7,
        **overrides,
    }


def anatomical_fixture(tmp_path, *, suffix, count):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    for index in range(count):
        image = write_anatomical(bids, suffix=suffix, run=str(index + 1))
        write_iqm(mriqc, image, **valid_metrics())
        write_report(mriqc, image)
    other_suffix = "T2w" if suffix == "T1w" else "T1w"
    other = write_anatomical(bids, suffix=other_suffix)
    write_iqm(mriqc, other, **valid_metrics())
    write_report(mriqc, other)
    return inspect_anatomicals(bids, mriqc, ["sub-s01"])


@pytest.mark.parametrize("suffix,count", [("T1w", 0), ("T1w", 2), ("T2w", 0), ("T2w", 2)])
def test_any_count_other_than_one_requires_review(suffix, count, tmp_path):
    rows = anatomical_fixture(tmp_path, suffix=suffix, count=count)

    assert any("anatomical_count" in row.flags for row in rows)


def duplicate_fixture(tmp_path, **metric_pairs):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    images = [
        write_anatomical(bids, suffix="T1w", acquisition="first", run="1"),
        write_anatomical(bids, suffix="T1w", acquisition="second", run="2"),
        write_anatomical(bids, suffix="T2w"),
    ]
    for index, image in enumerate(images):
        metrics = valid_metrics()
        if image.name.endswith("_T1w.nii.gz"):
            metrics.update({name: values[index] for name, values in metric_pairs.items()})
        write_iqm(mriqc, image, **metrics)
        write_report(mriqc, image)
    return inspect_anatomicals(bids, mriqc, ["sub-s01"])


def test_primary_metrics_agree_on_recommendation(tmp_path):
    rows = duplicate_fixture(tmp_path, cjv=(0.4, 0.7), cnr=(3.0, 2.0))
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]

    assert {row.recommendation for row in t1w_rows} == {"keep-first"}
    assert all(row.recommendation_status == "clear" for row in t1w_rows)


def test_secondary_majority_resolves_primary_disagreement(tmp_path):
    rows = duplicate_fixture(
        tmp_path,
        cjv=(0.4, 0.7), cnr=(2.0, 3.0), snr_total=(12.0, 10.0),
        efc=(0.4, 0.5), fber=(120.0, 100.0), qi_2=(0.03, 0.02), wm2max=(0.7, 0.9),
    )
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]

    assert {row.recommendation for row in t1w_rows} == {"keep-first"}
    assert all(row.recommendation_status == "clear" for row in t1w_rows)


def test_secondary_tie_or_insufficient_metrics_is_indeterminate(tmp_path):
    rows = duplicate_fixture(
        tmp_path,
        cjv=(0.4, 0.7), cnr=(2.0, 3.0), snr_total=(10.0, 10.0),
        efc=(0.4, 0.5), fber=(-1.0, -1.0), qi_2=(0.03, 0.02), wm2max=(0.7, 0.7),
    )
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]

    assert {row.recommendation for row in t1w_rows} == {""}
    assert {row.recommendation_status for row in t1w_rows} == {"indeterminate"}


def test_wm2max_uses_distance_to_documented_interval_and_fber_minus_one_is_ignored(tmp_path):
    rows = duplicate_fixture(
        tmp_path,
        cjv=(0.4, 0.7), cnr=(2.0, 3.0), snr_total=(10.0, 10.0),
        efc=(0.4, 0.4), fber=(-1.0, 999.0), qi_2=(0.02, 0.02), wm2max=(0.82, 0.7),
    )
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]

    assert {row.recommendation for row in t1w_rows} == {"keep-second"}
    assert all(row.recommendation_status == "clear" for row in t1w_rows)


def test_observed_identity_includes_session_acquisition_and_canonical_run(tmp_path):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    image = write_anatomical(
        bids, suffix="T1w", subject="sub-s02", session="ses-03", acquisition="mprage", run="01",
    )
    t2w = write_anatomical(bids, suffix="T2w", subject="sub-s02", session="ses-03")
    for path in (image, t2w):
        write_iqm(mriqc, path, **valid_metrics())
        write_report(mriqc, path)

    rows = inspect_anatomicals(bids, mriqc, ["s02"])
    t1w, = [row for row in rows if row.key.suffix == "T1w"]

    assert (t1w.key.subject, t1w.key.session, t1w.key.acquisition, t1w.key.run) == (
        "sub-s02", "ses-03", "mprage", "1",
    )


def test_missing_malformed_ambiguous_and_mismatched_mriqc_evidence_stays_explicit(tmp_path):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    first = write_anatomical(bids, suffix="T1w", acquisition="first", run="1")
    second = write_anatomical(bids, suffix="T1w", acquisition="second", run="2")
    t2w = write_anatomical(bids, suffix="T2w")
    write_iqm(mriqc, first, **valid_metrics())
    write_report(mriqc, first)
    duplicate = mriqc / "other" / first.name.removesuffix(".nii.gz")
    duplicate.parent.mkdir(parents=True)
    duplicate.with_suffix(".json").write_text(json.dumps(valid_metrics()))
    malformed = mriqc / second.relative_to(bids).with_suffix("").with_suffix(".json")
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{")
    wrong_parent = mriqc / "sub-s99" / "ses-01" / "anat" / t2w.name
    wrong_parent.parent.mkdir(parents=True)
    wrong_parent.with_suffix("").with_suffix(".json").write_text(json.dumps(valid_metrics()))

    rows = {row.key.acquisition or row.key.suffix: row for row in inspect_anatomicals(bids, mriqc, ["sub-s01"])}

    assert "ambiguous_iqm" in rows["first"].flags
    assert "malformed_iqm" in rows["second"].flags
    assert "identity_mismatch" in rows["T2w"].flags
    assert "missing_report" in rows["second"].flags
    assert rows["second"].recommendation_status == "indeterminate"


def test_evidence_never_approves_anatomicals(tmp_path):
    rows = anatomical_fixture(tmp_path, suffix="T1w", count=2)

    assert all(not hasattr(row, "decision") for row in rows)
