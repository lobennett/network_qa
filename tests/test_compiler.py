import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from network_qa import compiler
from network_qa.manifest import read_manifest


STEM = 'sub-s01_ses-01_task-nBack_run-1'


def fixture(tmp_path, *, exception=False):
    bids, mriqc = tmp_path / 'bids', tmp_path / 'mriqc'
    func = bids / 'sub-s01/ses-01/func'
    func.mkdir(parents=True)
    mriqc.mkdir()
    for echo in (1, 2, 3):
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 20)), np.eye(4)),
                 func / f'{STEM}_echo-{echo}_bold.nii.gz')
    iqm = mriqc / 'sub-s01/ses-01/func' / f'{STEM}_echo-2_bold.json'
    iqm.parent.mkdir(parents=True)
    iqm.write_text(json.dumps({'fd_mean': .1, 'fd_perc': 1, 'dvars_std': 1,
                              'provenance': {'settings': {'fd_thres': .5}, 'version': '24.0.2'}}))
    (mriqc / f'{STEM}_bold.html').write_text('MRIQC report')
    beh = bids / 'sourcedata/behavioral'
    beh.mkdir(parents=True)
    (beh / 'behavioral_exceptions.tsv').write_text(
        'subject\tsession\ttask\trun\treason\tdetail\n' +
        ('sub-s01\tses-01\tnBack\t1\tmissing\treviewed source absence\n' if exception else ''))
    qc = bids / 'sourcedata/events_qc'
    qc.mkdir()
    (qc / 'conversion_errors.tsv').write_text(
        'subject\tsession\ttask\trun\tsource_path\texception_class\tmessage\n')
    if not exception:
        raw = beh / 'sub-s01/ses-01/beh' / f'{STEM}_beh.csv'
        raw.parent.mkdir(parents=True)
        raw.write_text('trial,onset\n1,1\n')
        (func / f'{STEM}_events.tsv').write_text('onset\tduration\n1\t1\n')
        sidecar = qc / 'sub-s01/ses-01' / f'{STEM}_desc-truncation.json'
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({'NTestTrialsExpected': 10, 'NTestTrialsRetained': 10,
            'FractionTestTrialsDropped': 0, 'ScanDurationSeconds': 29.8,
            'NScanTestTrialsDropped': 0, 'FractionScanTestTrialsDropped': 0}))
    return bids, mriqc, tmp_path / 'out/scan_decisions.tsv'


def run(paths):
    compiler.compile_decisions(*paths)
    return next(row for row in read_manifest(paths[2]) if row.key.datatype == 'func')


def test_known_behavior_exception_is_evidence_not_drop(tmp_path):
    row = run(fixture(tmp_path, exception=True))
    assert row.decision == 'keep'
    assert row.behavioral_status == 'reviewed_exception'
    assert not row.approval_required and not row.approved


def test_clean_task_and_missing_anatomy(tmp_path):
    paths = fixture(tmp_path)
    row = run(paths)
    assert row.decision == 'keep' and not row.approved
    assert row.tr_count == 20 and row.original_tr_count == 27
    assert row.fd_mean == .1
    absent = [r for r in read_manifest(paths[2]) if r.key.record_type == 'missing_expected']
    assert {r.key.suffix for r in absent} == {'T1w', 'T2w'}
    assert all(r.decision == 'review' and r.approval_required and not r.approved for r in absent)


def test_conversion_error_requires_review_even_with_stale_events(tmp_path):
    paths = fixture(tmp_path)
    errors = paths[0] / 'sourcedata/events_qc/conversion_errors.tsv'
    with errors.open('a') as f:
        f.write('sub-s01\tses-01\tnBack\t1\traw.csv\tValueError\tbad timing\n')
    row = run(paths)
    assert row.decision == 'review' and row.approval_required
    assert row.event_status == 'failed'
    assert 'event_conversion_failed' in row.flags


@pytest.mark.parametrize('relative,flag', [
    ('sourcedata/behavioral/behavioral_exceptions.tsv', 'behavioral_evidence_unknown'),
    ('sourcedata/events_qc/conversion_errors.tsv', 'event_conversion_unknown'),
    (f'sourcedata/behavioral/sub-s01/ses-01/beh/{STEM}_beh.csv', 'behavioral_evidence_unknown'),
    (f'sourcedata/events_qc/sub-s01/ses-01/{STEM}_desc-truncation.json', 'truncation_unknown'),
    (f'sub-s01/ses-01/func/{STEM}_events.tsv', 'events_unknown'),
])
@pytest.mark.parametrize('damage', ['missing', 'unreadable'])
def test_missing_or_unreadable_evidence_requires_review(tmp_path, relative, flag, damage):
    paths = fixture(tmp_path)
    path = paths[0] / relative
    if damage == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'\xff')
    row = run(paths)
    assert row.decision == 'review' and row.approval_required and not row.approved
    assert flag in row.flags


def test_both_trial_losses_are_preserved_without_auto_drop(tmp_path):
    paths = fixture(tmp_path)
    sidecar = paths[0] / f'sourcedata/events_qc/sub-s01/ses-01/{STEM}_desc-truncation.json'
    data = json.loads(sidecar.read_text())
    data.update(NTestTrialsRetained=2, FractionTestTrialsDropped=.8,
                NScanTestTrialsDropped=1, FractionScanTestTrialsDropped=.5)
    sidecar.write_text(json.dumps(data))
    row = run(paths)
    assert row.decision == 'review' and row.approval_required
    assert {'nonmonotonic_trial_loss', 'scan_length_trial_loss'} <= set(row.flags)
    metadata = json.loads(paths[2].with_suffix('.meta.json').read_text())
    evidence, = metadata['behavioral_evidence']
    assert evidence['truncation_metrics'] == data


def test_metadata_deterministic_and_binds_inputs(tmp_path):
    paths = fixture(tmp_path)
    run(paths)
    out = paths[2]
    original = out.read_bytes(), out.with_suffix('.meta.json').read_bytes()
    metadata = json.loads(original[1])
    assert metadata['manifest_sha256'] == hashlib.sha256(original[0]).hexdigest()
    assert metadata['input_roots'] == {'bids_dir': str(paths[0]), 'mriqc_dir': str(paths[1])}
    assert metadata['approved_manifest_sha256'] is None
    assert metadata['provenance']['source_datalad_commit'] is None
    assert metadata['provenance']['mriqc_versions'] == ['24.0.2']
    run(paths)
    assert original == (out.read_bytes(), out.with_suffix('.meta.json').read_bytes())
    image = next(paths[0].glob('sub-*/ses-*/func/*nii.gz'))
    image.write_bytes(image.read_bytes() + b'changed')
    run(paths)
    assert json.loads(out.with_suffix('.meta.json').read_text())['inventory_sha256'] != metadata['inventory_sha256']


def test_metadata_publish_failure_restores_old_pair(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    run(paths)
    out = paths[2]
    old = out.read_bytes(), out.with_suffix('.meta.json').read_bytes()
    (paths[0] / f'sub-s01/ses-01/func/{STEM}_events.tsv').unlink()
    replace = Path.replace
    def fail_meta(self, target):
        if Path(target) == out.with_suffix('.meta.json'):
            raise OSError('injected publication failure')
        return replace(self, target)
    monkeypatch.setattr(Path, 'replace', fail_meta)
    with pytest.raises(OSError, match='publication'):
        run(paths)
    assert old == (out.read_bytes(), out.with_suffix('.meta.json').read_bytes())


def test_nonexistent_input_does_not_publish(tmp_path):
    out = tmp_path / 'scan_decisions.tsv'
    with pytest.raises(ValueError):
        compiler.compile_decisions(tmp_path / 'absent', tmp_path, out)
    assert not out.exists()


def test_anatomical_recommendation_is_nonbinding(tmp_path):
    paths = fixture(tmp_path)
    bids, mriqc, _ = paths
    anat = bids / 'sub-s01/ses-01/anat'
    anat.mkdir()
    for index, cjv, cnr in [(1, .4, 3), (2, .7, 2)]:
        stem = f'sub-s01_ses-01_run-{index}_T1w'
        (anat / f'{stem}.nii.gz').write_bytes(b'anatomical inventory')
        iqm = mriqc / 'sub-s01/ses-01/anat' / f'{stem}.json'
        iqm.parent.mkdir(exist_ok=True)
        iqm.write_text(json.dumps(dict(cjv=cjv, cnr=cnr, snr_total=10, efc=.4,
                                       fber=100, qi_2=.02, wm2max=.7)))
        (mriqc / f'{stem}.html').write_text('MRIQC report')
    run(paths)
    rows = [r for r in read_manifest(paths[2]) if r.key.suffix == 'T1w']
    assert len(rows) == 2 and all(r.recommendation for r in rows)
    assert all(r.decision == 'review' and r.approval_required and not r.approved for r in rows)


def test_outputs_cannot_overwrite_evidence(tmp_path):
    paths = fixture(tmp_path)
    events = paths[0] / f'sub-s01/ses-01/func/{STEM}_events.tsv'
    old = events.read_bytes()
    with pytest.raises(ValueError, match='output'):
        compiler.compile_decisions(paths[0], paths[1], events)
    assert events.read_bytes() == old


def test_failed_first_publication_leaves_no_pair(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    replace = Path.replace
    def fail_meta(self, target):
        if Path(target) == paths[2].with_suffix('.meta.json'):
            raise OSError('publication failed')
        return replace(self, target)
    monkeypatch.setattr(Path, 'replace', fail_meta)
    with pytest.raises(OSError):
        run(paths)
    assert not paths[2].exists() and not paths[2].with_suffix('.meta.json').exists()


def test_roster_only_subjects_have_missing_expected_rows(tmp_path):
    paths = fixture(tmp_path)
    (paths[0] / 'participants.tsv').write_text('participant_id\nsub-s01\nsub-s02\n')
    run(paths)
    rows = [r for r in read_manifest(paths[2]) if r.key.subject == 'sub-s02']
    assert {r.key.suffix for r in rows} == {'T1w', 'T2w'}
    assert all(r.decision == 'review' for r in rows)


def test_package_provenance_does_not_borrow_enclosing_repo(tmp_path, monkeypatch):
    import subprocess
    paths = fixture(tmp_path)
    application = tmp_path / 'app'
    application.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(application), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.org',
        'commit', '--allow-empty', '-m', 'application')
    installed = application / '.venv/lib/python/site-packages/network_qa/compiler.py'
    installed.parent.mkdir(parents=True)
    installed.write_text('installed wheel')
    monkeypatch.setattr(compiler, '__file__', str(installed))
    run(paths)
    metadata = json.loads(paths[2].with_suffix('.meta.json').read_text())
    assert metadata['provenance']['package_commits']['network_qa'] is None


def test_source_commit_and_mriqc_provenance_are_retained(tmp_path):
    import subprocess
    paths = fixture(tmp_path)
    def git(*args):
        return subprocess.run(['git', '-C', str(paths[0]), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-m', 'inputs')
    source_commit = git('rev-parse', 'HEAD')
    iqm = next(paths[1].rglob('*_bold.json'))
    data = json.loads(iqm.read_text())
    data['provenance']['input_commit'] = source_commit
    iqm.write_text(json.dumps(data))
    run(paths)
    metadata = json.loads(paths[2].with_suffix('.meta.json').read_text())
    assert metadata['provenance']['source_datalad_commit'] == source_commit
    assert metadata['provenance']['mriqc_input_commit'] == source_commit
    assert metadata['generation_timestamp'] == git('show', '-s', '--format=%cI', 'HEAD')
