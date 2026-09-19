import json

import pytest

from network_qa.anatomical import inspect_anatomicals


def write_anatomical(
    bids_dir, *, suffix, subject="sub-s01", session="ses-01", acquisition="", run="1",
    extension=".nii.gz", filename_subject=None, filename_session=None,
):
    anat = bids_dir / subject / session / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    entities = [filename_subject or subject, filename_session or session]
    if acquisition:
        entities.append(f"acq-{acquisition}")
    if run:
        entities.append(f"run-{run}")
    path = anat / ("_".join([*entities, suffix]) + extension)
    path.write_bytes(b"anatomical inventory only")
    return path


def write_iqm(mriqc_dir, image, **metrics):
    relative = image.relative_to(image.parents[3])
    path = mriqc_dir / relative.with_suffix("").with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics))
    return path


def write_report(mriqc_dir, image):
    stem = image.name.removesuffix(".nii.gz").removesuffix(".nii")
    path = mriqc_dir / f"{stem}.html"
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


def test_duplicate_compressed_and_uncompressed_encodings_are_one_untrusted_review_row(tmp_path):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    compressed = write_anatomical(bids, suffix="T1w", extension=".nii.gz")
    uncompressed = write_anatomical(bids, suffix="T1w", extension=".nii")
    t2w = write_anatomical(bids, suffix="T2w")
    for path in (compressed, t2w):
        write_iqm(mriqc, path, **valid_metrics())
        write_report(mriqc, path)

    rows = inspect_anatomicals(bids, mriqc, ["sub-s01"])
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]

    assert len(t1w_rows) == 1
    assert len({row.key for row in rows}) == len(rows)
    assert {"ambiguous_image", "anatomical_count", "untrusted_image"} <= set(t1w_rows[0].flags)
    assert all(value is None for value in t1w_rows[0].metrics.values())
    assert t1w_rows[0].recommendation_status == "indeterminate"


@pytest.mark.parametrize(
    "filename_subject,filename_session",
    [("sub-s99", "ses-01"), ("sub-s01", "ses-02")],
)
def test_selected_physical_parent_identity_mismatch_is_retained_as_invalid_observation(
    tmp_path, filename_subject, filename_session,
):
    bids = tmp_path / "bids"
    image = write_anatomical(
        bids, suffix="T1w", filename_subject=filename_subject, filename_session=filename_session,
    )
    write_anatomical(bids, suffix="T2w")

    rows = inspect_anatomicals(bids, tmp_path / "mriqc", ["sub-s01"])
    t1w, = [row for row in rows if row.key.suffix == "T1w"]

    assert (t1w.key.subject, t1w.key.session) == ("sub-s01", "ses-01")
    assert {"identity_mismatch", "untrusted_identity", "anatomical_count"} <= set(t1w.flags)
    assert t1w.key.record_type == "acquisition"
    assert t1w.metrics == {name: None for name in t1w.metrics}
    assert not any(row.key.record_type == "missing_expected" and row.key.suffix == "T1w" for row in rows)


@pytest.mark.parametrize("payload", [b"\xff", None])
def test_non_utf8_and_huge_numeric_iqms_are_malformed_evidence_not_errors(tmp_path, payload):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    t1w = write_anatomical(bids, suffix="T1w")
    t2w = write_anatomical(bids, suffix="T2w")
    iqm_path = write_iqm(mriqc, t1w, **valid_metrics())
    if payload is None:
        iqm_path.write_text(json.dumps(valid_metrics(cjv=10 ** 400)))
    else:
        iqm_path.write_bytes(payload)
    write_report(mriqc, t1w)
    write_iqm(mriqc, t2w, **valid_metrics())
    write_report(mriqc, t2w)

    rows = inspect_anatomicals(bids, mriqc, ["sub-s01"])
    t1w_row, = [row for row in rows if row.key.suffix == "T1w"]

    assert "malformed_iqm" in t1w_row.flags
    assert t1w_row.metrics["cjv"] is None


@pytest.mark.parametrize("kind", ["empty", "directory", "broken_symlink", "unreadable"])
def test_invalid_mriqc_reports_are_explicit_and_cannot_support_recommendations(tmp_path, kind):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    first = write_anatomical(bids, suffix="T1w", acquisition="first", run="1")
    second = write_anatomical(bids, suffix="T1w", acquisition="second", run="2")
    t2w = write_anatomical(bids, suffix="T2w")
    for path in (first, second, t2w):
        write_iqm(mriqc, path, **valid_metrics())
    invalid = mriqc / f"{first.name.removesuffix('.nii.gz')}.html"
    if kind == "empty":
        invalid.write_text("")
    elif kind == "directory":
        invalid.mkdir()
    elif kind == "broken_symlink":
        invalid.symlink_to(mriqc / "missing-report.html")
    else:
        invalid.write_text("MRIQC report")
        invalid.chmod(0)
    write_report(mriqc, second)
    write_report(mriqc, t2w)

    try:
        rows = inspect_anatomicals(bids, mriqc, ["sub-s01"])
    finally:
        if kind == "unreadable":
            invalid.chmod(0o644)
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]
    first_row, = [row for row in t1w_rows if row.key.acquisition == "first"]

    assert first_row.report_path is None
    assert "invalid_report" in first_row.flags
    assert {row.recommendation_status for row in t1w_rows} == {"clear"}
    assert {row.recommendation for row in t1w_rows} == {"keep-second"}


def test_counts_anatomicals_across_sessions_for_each_subject(tmp_path):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    first = write_anatomical(bids, suffix="T1w", session="ses-01")
    second = write_anatomical(bids, suffix="T1w", session="ses-02")
    t2w = write_anatomical(bids, suffix="T2w", session="ses-01")
    for path in (first, second, t2w):
        write_iqm(mriqc, path, **valid_metrics())
        write_report(mriqc, path)

    rows = inspect_anatomicals(bids, mriqc, ["sub-s01"])
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]

    assert len(t1w_rows) == 2
    assert all("anatomical_count" in row.flags for row in t1w_rows)


def test_ambiguous_report_is_explicit_and_selects_only_the_complete_duplicate(tmp_path):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    first = write_anatomical(bids, suffix="T1w", acquisition="first", run="1")
    second = write_anatomical(bids, suffix="T1w", acquisition="second", run="2")
    t2w = write_anatomical(bids, suffix="T2w")
    for path in (first, second, t2w):
        write_iqm(mriqc, path, **valid_metrics(cjv=0.5, cnr=2.0))
        write_report(mriqc, path)
    duplicate_report = mriqc / first.name.removesuffix("run-1_T1w.nii.gz")
    duplicate_report = duplicate_report.with_name(f"{duplicate_report.name}run-01_T1w.html")
    duplicate_report.write_text("MRIQC report")

    rows = inspect_anatomicals(bids, mriqc, ["sub-s01"])
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]
    first_row, = [row for row in t1w_rows if row.key.acquisition == "first"]

    assert "ambiguous_report" in first_row.flags
    assert {row.recommendation for row in t1w_rows} == {"keep-second"}


def test_primary_metric_ties_are_indeterminate(tmp_path):
    rows = duplicate_fixture(tmp_path, cjv=(0.5, 0.5), cnr=(2.0, 2.0))
    t1w_rows = [row for row in rows if row.key.suffix == "T1w"]

    assert {row.recommendation_status for row in t1w_rows} == {"indeterminate"}


@pytest.mark.parametrize("bad_value", [True, "0.5", [], float("nan")])
def test_wrong_or_nonfinite_metric_types_are_malformed_evidence(tmp_path, bad_value):
    bids = tmp_path / "bids"
    mriqc = tmp_path / "mriqc"
    t1w = write_anatomical(bids, suffix="T1w")
    t2w = write_anatomical(bids, suffix="T2w")
    write_iqm(mriqc, t1w, **valid_metrics(cjv=bad_value))
    write_report(mriqc, t1w)
    write_iqm(mriqc, t2w, **valid_metrics())
    write_report(mriqc, t2w)

    rows = inspect_anatomicals(bids, mriqc, ["sub-s01"])
    t1w_row, = [row for row in rows if row.key.suffix == "T1w"]

    assert "malformed_iqm" in t1w_row.flags
