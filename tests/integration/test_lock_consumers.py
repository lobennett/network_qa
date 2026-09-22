"""Optional integration coverage for installed sibling packages."""
import json

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("network_glm")
pytest.importorskip("network_fmri")

from network_glm.exclusions import load_exclusions
from network_glm.lev1.processing.fixed_effects import FixedEffectsAnalyzer
from network_glm.lev2.run import discover_input_files
from network_fmri.qa.exclusions import STAGES
from network_qa.cli import main


@pytest.mark.parametrize("padded", [False, True])
def test_final_lock_changes_fixed_effects_and_level2_cohort(tmp_path, padded):
    """Three runs -> motion leaves two -> final VIF rule leaves one/ineligible.

    The unchanged subject stays eligible. Use actual image arithmetic and file
    discovery, including retirement of the formerly eligible fixed-effects map.
    """
    mriqc = tmp_path / "mriqc"
    mriqc.mkdir()
    lev1 = tmp_path / "lev1"
    mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4))
    contrast = "incongruent-congruent"
    for subject in ("sub-s03", "sub-s10"):
        indiv = lev1 / subject / "task-flanker/indiv_contrasts"
        indiv.mkdir(parents=True)
        for run, value in [(1, 1), (2, 3), (3, 9)]:
            index = f"{run:02}" if padded else str(run)
            stem = f"{subject}_ses-01_task-flanker_run-{index}"
            (mriqc / f"{stem}_bold.json").write_text(json.dumps({
                "fd_perc": 30 if (subject, run) == ("sub-s03", 1) else 0,
                "fd_mean": 0.1, "fd_thres": 0.5,
            }))
            for stat, data in [("effect-size", value), ("variance", 1), ("z_score", value)]:
                nib.save(nib.Nifti1Image(np.full((3, 3, 3), data, dtype=np.float32),
                                        np.eye(4)),
                         indiv / f"{stem}_contrast-{contrast}_rtmodel-RTDur_stat-{stat}.nii.gz")

    outliers = tmp_path / "lev1_outliers.csv"
    index = "02" if padded else "2"
    outliers.write_text("subject,session,task,run,contrast,vif,outlier_pct\n"
                        f"sub-s03,ses-01,flanker,{index},{contrast},15,0\n")
    for stage, retained, eligible in [("motion", 2, 2), ("lev1", 1, 1)]:
        lock = tmp_path / f"{stage}.json"
        main(["compile", "--dataset", "discovery", "--generators", *STAGES[stage],
              "--bids-dir", str(tmp_path), "--mriqc-dir", str(mriqc),
              "--lev1-outliers-csv", str(outliers), "--out", str(lock)])
        exclusions = load_exclusions(lock)
        for subject in ("sub-s03", "sub-s10"):
            analyzer = FixedEffectsAnalyzer(subject, "flanker", mask_img=mask, min_runs=2)
            task_dir = lev1 / subject / "task-flanker"
            analyzer.compute_all_task_fixed_effects(
                task_dir / "indiv_contrasts", task_dir / "fixed_effects", exclusions,
                contrasts={contrast: "incongruent - congruent"})
            result = analyzer.contrast_results[contrast]
            assert result["n_runs"] == (retained if subject == "sub-s03" else 3)
            expected = (6 if stage == "motion" else 9) if subject == "sub-s03" else 13 / 3
            np.testing.assert_allclose(result["fixed_effect"].get_fdata(), expected, rtol=1e-6)
        inputs = discover_input_files([lev1], f"task-flanker_contrast-{contrast}")
        assert len(inputs) == eligible
        if stage == "lev1":
            assert all("sub-s10" in path for path in inputs)
