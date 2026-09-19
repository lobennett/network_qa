"""Compile review evidence and provenance into an unapproved scan manifest."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import re
import subprocess
import tempfile

from network_qa.anatomical import inspect_anatomicals
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


def _reject_directory_symlink(path: Path) -> None:
    if path.is_symlink() and path.is_dir():
        raise ValueError(f'evidence directory symlink is not supported: {path}')


def _reject_directory_symlinks(root: Path) -> None:
    # rglob yields directory links without following them. Check the scope root
    # too: a subject or sourcedata root may itself be linked outside the dataset.
    _reject_directory_symlink(root)
    for path in root.rglob('*'):
        _reject_directory_symlink(path)


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
    paths = set()
    for root in [*bids_dir.glob('sub-*'), bids_dir / 'sourcedata']:
        _reject_directory_symlinks(root)
        if root.is_dir():
            paths.update(path for path in root.rglob('*') if path.is_file() or path.is_symlink())
    for name in ('dataset_description.json', 'participants.tsv', 'participants.json', '.bidsignore'):
        path = bids_dir / name
        if path.exists() or path.is_symlink():
            paths.add(path)
    return [_file_record(bids_dir, path) for path in sorted(paths)]


def inventory_digest(records: list[dict]) -> str:
    """SHA-256 of UTF-8 sorted-key, two-space JSON plus final newline."""
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def _mriqc_records(mriqc_dir: Path) -> list[dict]:
    _reject_directory_symlinks(mriqc_dir)
    return [_file_record(mriqc_dir, path) for path in sorted(mriqc_dir.rglob('*'))
            if path.suffix in {'.json', '.html', '.tsv'} and (path.is_file() or path.is_symlink())]


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
        result = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True,
                                text=True, check=True, timeout=10)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _dataset_commit(root: Path) -> str | None:
    top = _git(root, 'rev-parse', '--show-toplevel')
    return _git(root, 'rev-parse', 'HEAD') if top and Path(top).resolve() == root else None


def _provenance(bids_dir, mriqc_dir):
    package_root = Path(__file__).resolve().parents[2]
    versions = set()
    coverage = []
    # Cover every acquisition IQM in the supplied MRIQC evidence root, including
    # unused echoes and ambiguous candidates, so no potential contributor is lost.
    # Dataset descriptions and unrelated JSON cannot establish input provenance.
    for path in sorted(mriqc_dir.rglob('*.json')):
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
    package_commit = _dataset_commit(package_root)
    package_dirty = bool(_git(package_root, 'status', '--porcelain')) if package_commit else None
    if package_commit is None:
        # PEP 610 retains the exact source revision for VCS-installed wheels.
        try:
            direct_url = distribution('network_qa').read_text('direct_url.json')
            install = json.loads(direct_url) if direct_url else {}
            vcs = install.get('vcs_info', {}) if isinstance(install, dict) else {}
            candidate = vcs.get('commit_id') if isinstance(vcs, dict) else None
            package_commit = candidate if isinstance(candidate, str) else None
        except (OSError, ValueError):
            package_commit = None
    commit = _dataset_commit(bids_dir)
    return {
        'source_datalad_commit': commit,
        'source_commit_time': _git(bids_dir, 'show', '-s', '--format=%cI', 'HEAD') if commit else None,
        'mriqc_dataset_commit': _dataset_commit(mriqc_dir),
        'mriqc_input_commit': next(iter(input_commits)) if complete and len(input_commits) == 1 else None,
        'mriqc_input_commit_candidates': sorted(input_commits),
        'mriqc_input_commit_coverage': coverage,
        'mriqc_input_commit_coverage_scope': 'all-acquisition-iqms-in-mriqc-root',
        'mriqc_input_commit_conflict': len(input_commits) > 1,
        'mriqc_versions': sorted(versions),
        'package_commits': {'network_qa': package_commit},
        'package_versions': {'network_qa': version('network_qa')},
        'package_dirty': package_dirty,
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
        'behavioral_evidence': [asdict(row) for row in behavior],
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
