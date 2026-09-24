"""Compile review evidence and provenance into an unapproved scan manifest."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import re
import subprocess
import stat
import zipfile
import tempfile

from network_qa.anatomical import inspect_anatomicals
from network_qa._evidence_paths import (
    is_vcs_administration_path, iter_evidence_files,
    reject_directory_symlink as _reject_directory_symlink,
)
from network_qa.exclusions.behavioral import behavioral_evidence
from network_qa.exclusions.motion import inspect_motion
from network_qa.functional import inspect_functionals
from network_qa.manifest import DecisionRow, manifest_bytes, write_manifest


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def dataset_subjects(bids_dir: Path) -> tuple[str, ...]:
    """Include observed subject directories and roster-only subjects."""
    import csv
    from network_qa.manifest import AcquisitionKey
    subjects = {path.name for path in bids_dir.glob('sub-*') if path.is_dir()}
    roster = bids_dir / 'participants.tsv'
    if roster.exists():
        with roster.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t', strict=True)
            if 'participant_id' not in (reader.fieldnames or ()):
                raise ValueError('participants.tsv is missing participant_id')
            for row in reader:
                if None in row or not row.get('participant_id'):
                    raise ValueError('invalid participants.tsv row')
                subjects.add(row['participant_id'])
    for subject in subjects:
        AcquisitionKey('missing_expected', subject, '', 'anat', 'T1w')
    return tuple(sorted(subjects))


def _file_record(root: Path, path: Path) -> dict:
    _reject_directory_symlink(path)
    record = {'path': path.relative_to(root).as_posix()}
    if path.is_symlink():
        record['symlink'] = str(path.readlink())
    try:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        record.update(sha256=digest.hexdigest(), status='available')
    except OSError:
        record.update(sha256=None, status='unreadable')
    return record


def inventory_records(bids_dir: Path) -> list[dict]:
    """Inventory raw subjects, canonical sourcedata, and root BIDS metadata.

    Derivatives and code are excluded so publishing or approving the manifest cannot
    change its inventory. Symlink targets and content are both bound; unavailable
    annex content is explicitly recorded. Root metadata names are listed below.
    """
    roots = [*bids_dir.glob('sub-*'), bids_dir / 'sourcedata']
    nested = set()
    for root in roots:
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root):
            if '.git' in directories or '.git' in files:
                candidate = Path(current)
                if _dataset_commit(candidate) is not None:
                    nested.add(candidate)
                directories[:] = []
    paths = set()
    for root in roots:
        paths.update(iter_evidence_files(root))
    for name in ('dataset_description.json', 'participants.tsv', 'participants.json', '.bidsignore'):
        path = bids_dir / name
        if path.exists() or path.is_symlink():
            paths.add(path)
    paths = {path for path in paths if not any(path.is_relative_to(dataset) for dataset in nested)}
    records = [_file_record(bids_dir, path) for path in sorted(paths)]
    for dataset in sorted(nested):
        commit = _dataset_commit(dataset)
        status = _git(dataset, 'status', '--porcelain', '--untracked-files=all')
        state = 'unreadable' if commit is None or status is None else 'dirty' if status else 'available'
        records.append({'path': dataset.relative_to(bids_dir).as_posix(),
                        'gitlink': commit, 'status': state})
    return sorted(records, key=lambda record: record['path'])


def inventory_digest(records: list[dict]) -> str:
    """SHA-256 of UTF-8 sorted-key, two-space JSON plus final newline."""
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def _mriqc_records(mriqc_dir: Path) -> list[dict]:
    return [_file_record(mriqc_dir, path) for path in sorted(iter_evidence_files(mriqc_dir))
            if path.suffix in {'.json', '.html', '.tsv'}]


def mriqc_inventory_records(mriqc_dir: Path) -> list[dict]:
    """Public MRIQC JSON/HTML/TSV content inventory contract."""
    return _mriqc_records(mriqc_dir)


def generation_metadata_digest(metadata: dict) -> str:
    """Bind every generated field, excluding the digest and later approval fields."""
    excluded = {'generation_metadata_sha256', 'approved_manifest_sha256',
                'approved_metadata_sha256'}
    return hashlib.sha256(_json_bytes({k: v for k, v in metadata.items()
                                      if k not in excluded})).hexdigest()


def _git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(['git', '--no-optional-locks', '-C', str(root), *arguments], capture_output=True,
                                text=True, check=True, timeout=10)
        # Empty successful output (notably git status) is distinct from failure.
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _dataset_commit(root: Path) -> str | None:
    top = _git(root, 'rev-parse', '--show-toplevel')
    return _git(root, 'rev-parse', 'HEAD') if top and Path(top).resolve() == root else None


def _receipt_input_commit(mriqc_dir: Path, coverage: list[dict]) -> str | None:
    """Read the input commit from a complete set of network_fmri MRIQC receipts."""
    root = mriqc_dir / 'code/network_fmri/run-receipts/mriqc'
    subjects = set()
    for record in coverage:
        match = re.search(r'(?:^|/)sub-([^_/]+)', record['path'])
        if match:
            subjects.add(match.group(1))
    expected = {root / f'sub-{subject}.json' for subject in subjects} | {root / 'group.json'}
    if not subjects or not all(path.is_file() for path in expected):
        return None
    commits = set()
    for path in expected:
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError):
            return None
        expected_subject = 'group' if path.name == 'group.json' else path.stem.removeprefix('sub-')
        commit = receipt.get('input_datalad_commit') if isinstance(receipt, dict) else None
        if (receipt.get('schema_version') != 1 or receipt.get('status') != 'success'
                or receipt.get('subject') != expected_subject or not isinstance(commit, str)
                or re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', commit) is None):
            return None
        commits.add(commit)
    return next(iter(commits)) if len(commits) == 1 else None


def _archive_member_records(archive: Path) -> list[dict]:
    """Hash the subject evidence under the single MRIQC archive directory."""
    subject = archive.name.split('_', 1)[0]
    prefix, seen, records = None, set(), []
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            parts = member.filename.rstrip('/').split('/')
            if ('\\' in member.filename or '\x00' in member.orig_filename
                    or any(part in {'', '.', '..'} for part in parts)
                    or is_vcs_administration_path(Path(*parts))
                    or stat.S_IFMT(member.external_attr >> 16) not in {0, stat.S_IFREG, stat.S_IFDIR}
                    or member.flag_bits & 1 or member.filename in seen):
                raise ValueError('unsafe or duplicate archive member')
            seen.add(member.filename)
            prefix = prefix or parts[0]
            if parts[0] != prefix or not archive.name.startswith(subject + '_' + prefix):
                raise ValueError('inconsistent archive directory prefix')
            if member.is_dir():
                continue
            if len(parts) < 2:
                raise ValueError('archive member lacks MRIQC directory prefix')
            relative = Path(*parts[1:])
            subjects = {match.group() for part in parts[1:]
                        for match in re.finditer(r'sub-[A-Za-z0-9]+', part)}
            if subjects and subjects != {subject}:
                raise ValueError('archive member has wrong subject')
            if relative == Path('dataset_description.json') or relative.suffix not in {'.json', '.html', '.tsv'}:
                continue
            if parts[1] == 'code':
                raise ValueError('reserved archive evidence member')
            if not subjects:
                continue
            digest = hashlib.sha256()
            with stream.open(member) as content:
                for chunk in iter(lambda: content.read(1024 * 1024), b''):
                    digest.update(chunk)
            records.append({'path': relative.as_posix(), 'sha256': digest.hexdigest()})
    return records


def _archive_evidence_commit(mriqc_dir: Path) -> str | None:
    """Verify the explicit archive handoff against its committed BABS snapshot.

    This is extraction provenance, not a participant/group execution receipt.
    The BABS raw gitlink, rather than a later raw HEAD, identifies MRIQC input.
    """
    receipt_path = mriqc_dir / 'code/network_fmri/mriqc-evidence.json'
    try:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (not isinstance(receipt, dict) or receipt.get('schema_version') != 1
                or receipt.get('kind') != 'babs-mriqc-evidence'
                or receipt.get('raw_gitlink') != 'sourcedata/raw'):
            return None
        source_name = receipt['source_dataset_name']
        if (not isinstance(source_name, str) or source_name in {'.', '..'}
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*', source_name)):
            return None
        source = mriqc_dir.parent / source_name
        commit, input_commit = receipt['source_dataset_commit'], receipt['input_datalad_commit']
        if (not source.is_absolute() or source.is_symlink() or source.resolve() != source
                or _dataset_commit(source) is None
                or any(not isinstance(value, str) or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', value)
                       for value in (commit, input_commit))
                or _git(source, 'merge-base', '--is-ancestor', commit, 'HEAD') is None):
            return None
        identity = _git(source, 'config', '--blob', f'{commit}:.datalad/config', '--get', 'datalad.dataset.id')
        if not identity or identity != receipt['source_dataset_id']:
            return None
        if _git(source, 'ls-tree', commit, '--', 'sourcedata/raw') != f'160000 commit {input_commit}\tsourcedata/raw':
            return None
        archives = receipt['archives']
        if not isinstance(archives, list) or not archives:
            return None
        seen, archive_evidence = set(), []
        for record in archives:
            name = record['path']
            if (not isinstance(name, str) or not re.fullmatch(r'sub-[A-Za-z0-9]+_[A-Za-z0-9_.+-]+\.zip', name)
                    or name in seen):
                return None
            seen.add(name)
            archive = source / name
            current = _file_record(source, archive)
            if current['status'] != 'available' or current['sha256'] != record['sha256']:
                return None
            listing = _git(source, 'ls-tree', commit, '--', name)
            header, separator, stored_name = (listing or '').partition('\t')
            fields = header.split()
            if not separator or stored_name != name or len(fields) != 3 or fields[1] != 'blob':
                return None
            mode, _, oid = fields
            if (mode == '120000') != archive.is_symlink():
                return None
            if archive.is_symlink():
                target = os.fsencode(archive.readlink())
                blob = hashlib.new('sha1' if len(commit) == 40 else 'sha256',
                                   f'blob {len(target)}\0'.encode() + target).hexdigest()
                key = re.fullmatch(r'(SHA256E?|MD5E?)-s([0-9]+)--([a-f0-9]+)(?:\..*)?', archive.readlink().name)
                if key is None or archive.stat().st_size != int(key[2]):
                    return None
                digest = hashlib.sha256() if key[1].startswith('SHA256') else hashlib.md5(usedforsecurity=False)
                with archive.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != key[3]:
                    return None
            else:
                blob = _git(source, 'hash-object', '--no-filters', '--', name)
            if blob != oid:
                return None
            archive_evidence.extend(_archive_member_records(archive))
        tracked = _git(source, 'ls-tree', '-r', '--name-only', commit)
        committed_archives = {name for name in (tracked or '').splitlines()
                              if '/' not in name and name.startswith('sub-') and name.endswith('.zip')}
        if seen != committed_archives:
            return None
        records = mriqc_inventory_records(mriqc_dir)
        expected = sorted(receipt['evidence'], key=lambda row: row['path'])
        actual = [{'path': row['path'], 'sha256': row['sha256']} for row in records
                  if row['path'] != receipt_path.relative_to(mriqc_dir).as_posix()]
        if any(row['status'] != 'available' for row in records) or expected != actual:
            return None
        # The receipt is editable: the immutable archive members must independently
        # reproduce every retained evidence path and byte hash. Only the generated
        # root derivative description is intentionally outside the ZIP projection.
        retained = [row for row in actual if row['path'] != 'dataset_description.json']
        if sorted(archive_evidence, key=lambda row: row['path']) != retained:
            return None
        return input_commit
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, zipfile.BadZipFile, EOFError):
        return None


def _provenance(bids_dir, mriqc_dir):
    package_root = Path(__file__).resolve().parents[2]
    versions = set()
    coverage = []
    # Cover every acquisition IQM in the supplied MRIQC evidence root, including
    # unused echoes and ambiguous candidates, so no potential contributor is lost.
    # Dataset descriptions and unrelated JSON cannot establish input provenance.
    for path in sorted(path for path in iter_evidence_files(mriqc_dir) if path.suffix == '.json'):
        record = None
        if path.name.endswith(('_bold.json', '_T1w.json', '_T2w.json')):
            record = {'path': path.relative_to(mriqc_dir).as_posix(),
                      'state': 'malformed', 'input_commit': None}
            coverage.append(record)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                continue
            provenance = data.get('provenance', {})
            if record is not None and isinstance(provenance, dict):
                if 'input_commit' not in provenance:
                    record['state'] = 'missing'
                else:
                    candidate = provenance['input_commit']
                    if isinstance(candidate, str) and re.fullmatch(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})', candidate):
                        record.update(state='valid', input_commit=candidate.lower())
            if isinstance(provenance, dict):
                if isinstance(provenance.get('version'), str):
                    versions.add(provenance['version'])
            generators = data.get('GeneratedBy', [])
            if isinstance(generators, list):
                for generator in generators:
                    if isinstance(generator, dict) and str(generator.get('Name', '')).lower() == 'mriqc':
                        if isinstance(generator.get('Version'), str):
                            versions.add(generator['Version'])
        except OSError:
            if record is not None:
                record['state'] = 'unreadable'
        except (UnicodeError, ValueError, TypeError):
            continue
    input_commits = {record['input_commit'] for record in coverage if record['state'] == 'valid'}
    complete = bool(coverage) and all(record['state'] == 'valid' for record in coverage)
    receipt_commit = _receipt_input_commit(mriqc_dir, coverage)
    mriqc_input_commit = (next(iter(input_commits)) if complete and len(input_commits) == 1
                          else receipt_commit)
    input_basis = ('iqm-provenance' if complete and len(input_commits) == 1 else
                   'network_fmri-run-receipts' if receipt_commit else 'unavailable')
    archive_receipt = mriqc_dir / 'code/network_fmri/mriqc-evidence.json'
    archive_conflict = False
    if archive_receipt.exists() or archive_receipt.is_symlink():
        archive_commit = _archive_evidence_commit(mriqc_dir)
        archive_conflict = bool(input_commits and input_commits != {archive_commit})
        malformed = any(record['state'] not in {'valid', 'missing'} for record in coverage)
        mriqc_input_commit = archive_commit if coverage and not archive_conflict and not malformed else None
        input_basis = 'network_fmri-archive-evidence' if mriqc_input_commit else 'invalid-archive-evidence'
    package_commit = _dataset_commit(package_root)
    package_basis = 'source-checkout' if package_commit else 'unknown'
    package_status = _git(package_root, 'status', '--porcelain') if package_commit else None
    package_dirty = None if package_status is None else bool(package_status)
    if package_commit is None:
        # PEP 610 retains the exact source revision for VCS-installed wheels.
        try:
            direct_url = distribution('network_qa').read_text('direct_url.json')
            install = json.loads(direct_url) if direct_url else {}
            vcs = install.get('vcs_info', {}) if isinstance(install, dict) else {}
            candidate = vcs.get('commit_id') if isinstance(vcs, dict) else None
            if isinstance(vcs, dict) and vcs.get('vcs') == 'git' and isinstance(candidate, str) and re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', candidate):
                package_commit = candidate
                package_basis = 'pep610-vcs'
        except (OSError, ValueError):
            package_commit = None
    commit = _dataset_commit(bids_dir)
    return {
        'source_datalad_commit': commit,
        'source_commit_time': _git(bids_dir, 'show', '-s', '--format=%cI', 'HEAD') if commit else None,
        'mriqc_dataset_commit': _dataset_commit(mriqc_dir),
        'mriqc_input_commit': mriqc_input_commit,
        'mriqc_input_commit_basis': input_basis,
        'mriqc_input_commit_candidates': sorted(input_commits),
        'mriqc_input_commit_coverage': coverage,
        'mriqc_input_commit_coverage_scope': 'all-acquisition-iqms-in-mriqc-root',
        'mriqc_input_commit_conflict': len(input_commits) > 1 or archive_conflict,
        'mriqc_versions': sorted(versions),
        'package_commits': {'network_qa': package_commit},
        'package_versions': {'network_qa': version('network_qa')},
        'package_dirty': package_dirty,
        'package_provenance_basis': package_basis,
    }


def _publish_pair(output: Path, manifest: Path, metadata: Path) -> None:
    """Replace files atomically and restore the old manifest on a caught failure.

    Two directory entries cannot be replaced as one filesystem transaction. Metadata
    is published last as the completion marker; its digest detects interruption
    between replacements. Readers must validate that digest before trusting a pair.
    """
    meta_output = output.with_suffix('.meta.json')
    old = output.read_bytes() if output.exists() else None
    manifest.replace(output)
    try:
        metadata.replace(meta_output)
    except BaseException:
        if old is None:
            output.unlink()
        else:
            rollback = manifest.parent / 'rollback.tsv'
            rollback.write_bytes(old)
            rollback.replace(output)
        raise


def collect_decision_evidence(bids_dir: Path, mriqc_dir: Path) -> tuple[tuple[DecisionRow, ...], dict]:
    """Recompute the exact generated baseline and metadata without writing files."""
    bids_dir, mriqc_dir = Path(bids_dir), Path(mriqc_dir)
    _reject_directory_symlink(bids_dir)
    _reject_directory_symlink(mriqc_dir)
    bids_dir, mriqc_dir = bids_dir.resolve(), mriqc_dir.resolve()
    if not bids_dir.is_dir() or not mriqc_dir.is_dir():
        raise ValueError('BIDS and MRIQC inputs must be existing directories')
    before = inventory_records(bids_dir)
    mriqc_before = _mriqc_records(mriqc_dir)
    functionals = inspect_functionals(bids_dir)
    motion = {row.key: row for row in inspect_motion(functionals, mriqc_dir)}
    behavior = behavioral_evidence(bids_dir, functionals)
    by_key = {row.key: row for row in behavior}
    rows = []
    for functional in functionals:
        values = asdict(functional)
        values.pop('key')
        moving, behaving = motion[functional.key], by_key[functional.key]
        flags = tuple(sorted(set((*functional.flags, *moving.flags, *behaving.flags))))
        values.update(flags=flags, fd_mean=moving.fd_mean, fd_perc=moving.fd_perc,
                      fd_thres=moving.fd_thres, dvars_std=moving.dvars_std,
                      mriqc_report_path=str(moving.report_path or ''),
                      behavioral_status=behaving.behavioral_status, event_status=behaving.event_status)
        rows.append(DecisionRow(key=functional.key, **values, decision='review' if flags else 'keep',
                                approval_required=bool(flags), approved=False))
    for anatomical in inspect_anatomicals(bids_dir, mriqc_dir, dataset_subjects(bids_dir)):
        metrics = {f'mriqc_{"qi2" if name == "qi_2" else name}': value
                   for name, value in anatomical.metrics.items()}
        rows.append(DecisionRow(key=anatomical.key, **metrics, flags=anatomical.flags,
            mriqc_report_path=str(anatomical.report_path or ''),
            recommendation=anatomical.recommendation, recommendation_status=anatomical.recommendation_status,
            recommendation_rationale=anatomical.recommendation_rationale,
            decision='review' if anatomical.flags else 'keep',
            approval_required=bool(anatomical.flags), approved=False))
    provenance = _provenance(bids_dir, mriqc_dir)
    inventory, mriqc_inventory = inventory_records(bids_dir), _mriqc_records(mriqc_dir)
    if inventory != before or mriqc_inventory != mriqc_before:
        raise ValueError('inputs changed during compilation')
    metadata = {
        'schema_version': 1,
        'input_roots': {'bids_dir': str(bids_dir), 'mriqc_dir': str(mriqc_dir)},
        'inventory': inventory, 'inventory_sha256': inventory_digest(inventory),
        'inventory_policy': 'raw-subjects+sourcedata+named-root-metadata; sha256-content-and-symlink; v1',
        'inventory_digest_encoding': 'json-sort-keys-indent-2-utf8-final-newline',
        'mriqc_inventory': mriqc_inventory, 'mriqc_inventory_sha256': inventory_digest(mriqc_inventory),
        'provenance': provenance,
        'generation_timestamp': provenance['source_commit_time'],
        'generation_timestamp_basis': 'source-commit-time; null when unavailable',
        'approved_manifest_sha256': None,
        'approved_metadata_sha256': None,
        'behavioral_evidence': [asdict(row) for row in behavior],
        'motion_calculations': [
            {'key': asdict(row.key), 'method': row.fd_method, 'source': row.fd_source,
             'original_fd_thres': row.fd_original_thres, 'fd_thres': row.fd_thres,
             'analyzed_volumes': row.fd_n_volumes, 'mriqc_dummy_trs': row.fd_dummy_trs}
            for row in sorted(motion.values(), key=lambda row: row.key)
        ],
        'row_count': len(rows),
    }
    metadata['manifest_sha256'] = hashlib.sha256(manifest_bytes(rows)).hexdigest()
    # Round-trip to the JSON representation: behavioral dataclasses contain tuples.
    metadata = json.loads(_json_bytes(metadata))
    metadata['generation_metadata_sha256'] = generation_metadata_digest(metadata)
    return tuple(sorted(rows, key=lambda row: row.key)), metadata


def compile_decisions(bids_dir: Path, mriqc_dir: Path, output: Path) -> Path:
    """Generate unapproved decisions and deterministic, explicitly unsealed metadata."""
    bids_dir, mriqc_dir, output = Path(bids_dir), Path(mriqc_dir), Path(output).absolute()
    if output.suffix != '.tsv':
        raise ValueError('manifest output must have a .tsv suffix')
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(mriqc_dir.resolve()):
        raise ValueError('output must be outside MRIQC evidence')
    if resolved_output.is_relative_to(bids_dir.resolve()):
        relative = resolved_output.relative_to(bids_dir.resolve())
        if len(relative.parts) == 1 or relative.parts[0] != 'code':
            raise ValueError('output inside BIDS must be under code/')
    rows, metadata = collect_decision_evidence(bids_dir, mriqc_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.scan-decisions-', dir=output.parent) as temporary:
        staging = Path(temporary)
        manifest = staging / 'manifest.tsv'
        metadata['manifest_sha256'] = write_manifest(manifest, rows)
        meta_file = staging / 'metadata.json'
        meta_file.write_bytes(_json_bytes(metadata))
        _publish_pair(output, manifest, meta_file)
    return output
