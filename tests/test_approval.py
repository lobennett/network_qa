"""Approval integration tests use real inventories, Git commits and scan evidence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from network_qa import compiler
from network_qa.approval import seal_approval, validate_approval
from network_qa.cli import main
from network_qa.manifest import read_manifest, write_manifest
from test_compiler import fixture as evidence_fixture


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def generated(tmp_path, monkeypatch):
    bids, mriqc, manifest = evidence_fixture(tmp_path, exception=True)
    git(bids, 'init', '-q')
    git(bids, 'add', '.')
    git(bids, '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
        'commit', '-qm', 'input')
    commit = git(bids, 'rev-parse', 'HEAD')
    iqm = next(mriqc.rglob('*_bold.json'))
    data = json.loads(iqm.read_text())
    data['provenance']['input_commit'] = commit
    iqm.write_text(json.dumps(data))
    # The test suite runs against an intentionally dirty development checkout.
    # Only the package Git boundary is isolated; dataset Git/provenance is real.
    original_git = compiler._git
    package = Path(compiler.__file__).resolve().parents[2]
    def package_git(root, *args):
        if Path(root) == package and args == ('status', '--porcelain'):
            return None
        return original_git(root, *args)
    monkeypatch.setattr(compiler, '_git', package_git)
    compiler.compile_decisions(bids, mriqc, manifest)
    return manifest, manifest.with_suffix('.meta.json'), bids


def resolve(manifest, **changes):
    rows = read_manifest(manifest)
    write_manifest(manifest, [replace(row, decision='keep', approved=True,
        reason_code='other', reason_detail='Reviewed source evidence', reviewer='Alice',
        reviewed_at='2026-09-19T12:00:00Z', **changes) if row.flags else row for row in rows])


def change_flagged(manifest, **changes):
    write_manifest(manifest, [replace(row, **changes) if row.flags else row
                              for row in read_manifest(manifest)])


def meta_edit(metadata, change):
    data = json.loads(metadata.read_text())
    change(data)
    metadata.write_text(json.dumps(data))


def test_review_row_blocks_approval(generated):
    result = seal_approval(*generated)
    assert not result.ok
    assert any('unresolved review' in error for error in result.errors)


def test_seal_records_checksum_and_only_changes_metadata_approval_fields(generated):
    manifest, metadata, bids = generated
    resolve(manifest)
    before = manifest.read_bytes()
    meta_before = json.loads(metadata.read_text())
    inventory = compiler.inventory_records(bids)
    result = seal_approval(*generated)
    assert result.ok, result.errors
    assert result.manifest_sha256 == hashlib.sha256(before).hexdigest()
    meta_after = json.loads(metadata.read_text())
    assert meta_after.pop('approved_manifest_sha256') == result.manifest_sha256
    assert meta_after.pop('approved_metadata_sha256')
    meta_before.pop('approved_manifest_sha256')
    assert meta_after == meta_before
    assert manifest.read_bytes() == before
    assert compiler.inventory_records(bids) == inventory
    sealed = metadata.read_bytes()
    assert validate_approval(*generated).ok
    assert seal_approval(*generated).ok
    assert metadata.read_bytes() == sealed


def test_validate_requires_seal_and_never_writes(generated):
    resolve(generated[0])
    before = generated[1].read_bytes()
    result = validate_approval(*generated)
    assert not result.ok
    assert any('not sealed' in e for e in result.errors)
    assert generated[1].read_bytes() == before


@pytest.mark.parametrize('change', [
    {'decision': 'review'}, {'approved': False}, {'reason_detail': '  '},
    {'reviewer': ''}, {'reviewed_at': ''},
    {'decision': 'drop', 'reason_code': 'made_up'}, {'decision': 'drop', 'reason_code': ''},
])
def test_incomplete_review_blocks_sealing(generated, change):
    resolve(generated[0])
    change_flagged(generated[0], **change)
    before = generated[1].read_bytes()
    assert not seal_approval(*generated).ok
    assert generated[1].read_bytes() == before


def test_controlled_drop_is_allowed(generated):
    resolve(generated[0])
    change_flagged(generated[0], decision='drop', reason_code='incomplete_acquisition')
    assert seal_approval(*generated).ok


@pytest.mark.parametrize('change', [
    {'flags': ()}, {'approval_required': False}, {'recommendation': 'keep'},
    {'tr_count': 42}, {'mriqc_report_path': '/tmp/forged.html'},
])
def test_evidence_edits_cannot_be_sealed(generated, change):
    resolve(generated[0])
    change_flagged(generated[0], **change)
    result = seal_approval(*generated)
    assert not result.ok
    assert any('evidence' in e for e in result.errors)


def test_removed_row_cannot_be_sealed(generated):
    resolve(generated[0])
    write_manifest(generated[0], read_manifest(generated[0])[:-1])
    assert not seal_approval(*generated).ok


def test_crash_mismatched_generation_pair_cannot_be_silently_sealed(generated):
    resolve(generated[0])
    meta_edit(generated[1], lambda m: m.update(manifest_sha256='0' * 64))
    assert not seal_approval(*generated).ok


@pytest.mark.parametrize('operation', [seal_approval, validate_approval])
def test_any_post_seal_tsv_edit_invalidates_approval(generated, operation):
    resolve(generated[0])
    assert seal_approval(*generated).ok
    generated[0].write_bytes(generated[0].read_bytes().replace(b'Alice', b'Bob'))
    result = operation(*generated)
    assert not result.ok
    assert any('checksum' in e for e in result.errors)


@pytest.mark.parametrize('target', ['bids', 'mriqc', 'source_commit', 'metadata', 'root'])
@pytest.mark.parametrize('sealed', [False, True])
def test_stale_inputs_and_metadata_are_rejected(generated, target, sealed, tmp_path):
    manifest, metadata, bids = generated
    resolve(manifest)
    if sealed:
        assert seal_approval(*generated).ok
    if target == 'bids':
        (bids / 'participants.tsv').write_text('participant_id\nsub-s01\n')
    elif target == 'mriqc':
        mriqc = Path(json.loads(metadata.read_text())['input_roots']['mriqc_dir'])
        next(mriqc.glob('*.html')).write_text('changed report')
    elif target == 'source_commit':
        git(bids, '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
            'commit', '--allow-empty', '-qm', 'new milestone')
    elif target == 'metadata':
        meta_edit(metadata, lambda m: m.update(behavioral_evidence=[]))
    else:
        meta_edit(metadata, lambda m: m['input_roots'].update(bids_dir=str(tmp_path)))
    before = metadata.read_bytes()
    result = (validate_approval if sealed else seal_approval)(*generated)
    assert not result.ok
    assert metadata.read_bytes() == before


@pytest.mark.parametrize('provenance', [None, '0' * 40])
def test_unknown_or_wrong_mriqc_input_commit_blocks_even_regenerated_pair(generated, provenance):
    manifest, metadata, bids = generated
    mriqc = Path(json.loads(metadata.read_text())['input_roots']['mriqc_dir'])
    iqm = next(mriqc.rglob('*_bold.json'))
    data = json.loads(iqm.read_text())
    data['provenance']['input_commit'] = provenance
    iqm.write_text(json.dumps(data))
    compiler.compile_decisions(bids, mriqc, manifest)
    resolve(manifest)
    result = seal_approval(*generated)
    assert not result.ok
    assert any('MRIQC input commit' in e for e in result.errors)


def test_unavailable_inventory_content_blocks_even_regenerated_pair(generated):
    manifest, metadata, bids = generated
    (bids / 'sourcedata/missing.txt').symlink_to('missing-annex-object')
    mriqc = Path(json.loads(metadata.read_text())['input_roots']['mriqc_dir'])
    compiler.compile_decisions(bids, mriqc, manifest)
    resolve(manifest)
    result = seal_approval(*generated)
    assert not result.ok
    assert any('unavailable' in e for e in result.errors)


@pytest.mark.parametrize('content', ['{', '[]', '{}', '{"schema_version": 99}'])
def test_malformed_metadata_returns_structured_failure(generated, content):
    generated[1].write_text(content)
    assert not seal_approval(*generated).ok
    assert not validate_approval(*generated).ok


def test_cli_json_and_stable_exit_codes(generated, capsys):
    manifest, metadata, bids = generated
    options = ['--manifest', str(manifest), '--metadata', str(metadata), '--bids-dir', str(bids)]
    with pytest.raises(SystemExit) as error:
        main(['decisions', 'approve', *options])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)['ok'] is False
    resolve(manifest)
    main(['decisions', 'approve', *options])
    assert json.loads(capsys.readouterr().out)['ok'] is True
    main(['decisions', 'validate', *options])
    assert json.loads(capsys.readouterr().out)['ok'] is True
    manifest.unlink()
    with pytest.raises(SystemExit) as error:
        main(['decisions', 'validate', *options])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)['errors']


def test_atomic_publication_failure_preserves_both_artifacts(generated, monkeypatch):
    resolve(generated[0])
    before = [p.read_bytes() for p in generated[:2]]
    original_replace = Path.replace
    def fail_approval_replace(path, target):
        if path.name.startswith('.approval-'):
            raise OSError('simulated publication failure')
        return original_replace(path, target)
    monkeypatch.setattr(Path, 'replace', fail_approval_replace)
    result = seal_approval(*generated)
    assert not result.ok
    assert 'publication' in result.errors[0]
    assert [p.read_bytes() for p in generated[:2]] == before
    assert not list(generated[1].parent.glob('.approval-*'))


def test_uncommitted_source_inventory_blocks_regenerated_pair(generated):
    manifest, metadata, bids = generated
    (bids / 'sourcedata/new.txt').write_text('uncommitted evidence')
    mriqc = Path(json.loads(metadata.read_text())['input_roots']['mriqc_dir'])
    compiler.compile_decisions(bids, mriqc, manifest)
    resolve(manifest)
    result = seal_approval(*generated)
    assert not result.ok
    assert any('not bound to source commit' in error for error in result.errors)


@pytest.mark.parametrize('state', ['dirty', 'unbound', 'changed'])
def test_package_provenance_blocks_approval(generated, monkeypatch, state):
    resolve(generated[0])
    package = Path(compiler.__file__).resolve().parents[2]
    original = compiler._git
    def changed_package(root, *args):
        if Path(root) == package:
            if args == ('status', '--porcelain') and state == 'dirty':
                return ' M src/network_qa/approval.py'
            if args == ('rev-parse', 'HEAD') and state in {'unbound', 'changed'}:
                return None if state == 'unbound' else '0' * 40
        return original(root, *args)
    monkeypatch.setattr(compiler, '_git', changed_package)
    assert not seal_approval(*generated).ok


def test_clean_manual_drop_still_needs_human_approval(generated):
    manifest = generated[0]
    resolve(manifest)
    write_manifest(manifest, [replace(row, decision='drop', reason_code='other') if not row.flags else row
                              for row in read_manifest(manifest)])
    assert not seal_approval(*generated).ok


def test_unknown_source_commit_is_not_accepted(generated, monkeypatch):
    manifest, metadata, bids = generated
    original = compiler._dataset_commit
    monkeypatch.setattr(compiler, '_dataset_commit',
                        lambda root: None if root == bids else original(root))
    mriqc = Path(json.loads(metadata.read_text())['input_roots']['mriqc_dir'])
    compiler.compile_decisions(bids, mriqc, manifest)
    resolve(manifest)
    result = seal_approval(*generated)
    assert not result.ok
    assert any('source DataLad/Git commit' in e for e in result.errors)


@pytest.mark.parametrize('hidden', ['ignored', 'assume_unchanged'])
def test_git_status_cannot_hide_unbound_inventory_changes(generated, hidden):
    manifest, metadata, bids = generated
    if hidden == 'ignored':
        (bids / '.git/info/exclude').write_text('sourcedata/ignored.txt\n')
        (bids / 'sourcedata/ignored.txt').write_text('not recorded in source commit')
    else:
        relative = 'sourcedata/behavioral/behavioral_exceptions.tsv'
        git(bids, 'update-index', '--assume-unchanged', relative)
        path = bids / relative
        path.write_text(path.read_text() + '\n')
    mriqc = Path(json.loads(metadata.read_text())['input_roots']['mriqc_dir'])
    compiler.compile_decisions(bids, mriqc, manifest)
    resolve(manifest)
    result = seal_approval(*generated)
    assert not result.ok
    assert any('not bound to source commit' in error for error in result.errors)


@pytest.mark.parametrize('kind', ['annex_valid', 'annex_corrupt', 'external', 'loop'])
def test_source_commit_must_bind_symlink_content(generated, tmp_path, kind):
    manifest, metadata, bids = generated
    content = b'committed annex evidence'
    if kind.startswith('annex'):
        key = f'SHA256E-s{len(content)}--{hashlib.sha256(content).hexdigest()}.txt'
        target = bids / '.git/annex/objects/aa/bb' / key / key
        target.parent.mkdir(parents=True)
        target.write_bytes(content if kind == 'annex_valid' else b'corrupted content')
    elif kind == 'loop':
        target = bids / 'sourcedata/link.txt'
    else:
        target = tmp_path / 'external.txt'
        target.write_bytes(content)
    (bids / 'sourcedata/link.txt').symlink_to(target)
    git(bids, 'add', 'sourcedata/link.txt')
    git(bids, '-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'link')
    mriqc = Path(json.loads(metadata.read_text())['input_roots']['mriqc_dir'])
    iqm = next(mriqc.rglob('*_bold.json'))
    data = json.loads(iqm.read_text())
    data['provenance']['input_commit'] = git(bids, 'rev-parse', 'HEAD')
    iqm.write_text(json.dumps(data))
    compiler.compile_decisions(bids, mriqc, manifest)
    resolve(manifest)
    result = seal_approval(*generated)
    assert result.ok is (kind == 'annex_valid'), result.errors
