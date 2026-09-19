"""Read-only evidence verification and atomic sealing of human scan decisions.

Checksums detect stale or edited artifacts; they are not reviewer authentication or
cryptographic signatures. Generation, review and sealing are serial operations.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from network_qa.compiler import (
    collect_decision_evidence, generation_metadata_digest, inventory_records,
    mriqc_inventory_records,
)
from network_qa.manifest import manifest_bytes, read_manifest
from network_qa._evidence_paths import is_vcs_administration_path


DROP_REASONS = frozenset({
    'excessive_motion', 'anatomical_quality', 'incomplete_acquisition',
    'severe_artifact', 'duplicate_lower_quality', 'aborted_run',
    'missing_required_metadata', 'other',
})
_REVIEW_FIELDS = frozenset({
    'decision', 'approved', 'reason_code', 'reason_detail', 'reviewer', 'reviewed_at',
})
_APPROVAL_FIELDS = frozenset({'approved_manifest_sha256', 'approved_metadata_sha256'})
_COMMIT = re.compile(r'(?:[0-9a-f]{40}|[0-9a-f]{64})\Z')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')


@dataclass(frozen=True)
class ApprovalResult:
    ok: bool
    errors: tuple[str, ...]
    manifest_sha256: str


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _metadata_seal(metadata: dict) -> str:
    return hashlib.sha256(_json_bytes({k: v for k, v in metadata.items()
                                      if k != 'approved_metadata_sha256'})).hexdigest()


def _evidence_projection(data: bytes) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in _REVIEW_FIELDS}
            for row in csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''), delimiter='\t')]


def _review_errors(rows) -> list[str]:
    errors = []
    if not rows:
        errors.append('evidence: empty acquisition inventory cannot be approved')
    for row in rows:
        label = '/'.join(str(v) for v in asdict(row.key).values() if v)
        if row.decision == 'review':
            errors.append(f'review: unresolved review for {label}')
        needs_review = bool(row.flags) or row.approval_required or row.decision == 'drop' or row.approved
        if needs_review:
            if not row.approved:
                errors.append(f'review: approved=yes required for {label}')
            for field in ('reason_detail', 'reviewer', 'reviewed_at'):
                if not getattr(row, field).strip():
                    errors.append(f'review: nonempty {field} required for {label}')
        if row.decision == 'drop' and row.reason_code not in DROP_REASONS:
            errors.append(f'review: valid controlled drop reason_code required for {label}')
    return errors


def _valid_commit(value) -> bool:
    return isinstance(value, str) and _COMMIT.fullmatch(value) is not None


def _source_inventory_matches(bids_dir: Path, commit: str, records: list[dict], *, mriqc=False) -> bool:
    """Compare content to the commit, independent of index flags and ignore rules."""
    if any(record['status'] != 'available' for record in records):
        return False
    listing = subprocess.run([
        'git', '--no-optional-locks', '-C', str(bids_dir), 'ls-tree', '-r', '-z', commit,
    ], capture_output=True, check=True, timeout=30).stdout
    committed = {}
    for entry in listing.split(b'\0'):
        if not entry:
            continue
        header, raw_path = entry.split(b'\t', 1)
        path = os.fsdecode(raw_path)
        if is_vcs_administration_path(Path(path)):
            continue
        parts = Path(path).parts
        in_scope = (Path(path).suffix in {'.json', '.html', '.tsv'} if mriqc else
                    ((len(parts) > 1 and (parts[0].startswith('sub-') or parts[0] == 'sourcedata'))
                     or path in {'dataset_description.json', 'participants.tsv', 'participants.json', '.bidsignore'}))
        if in_scope:
            mode, kind, oid = header.decode('ascii').split()
            committed[path] = mode, kind, oid
    if set(committed) != {record['path'] for record in records}:
        return False
    records_by_path = {record['path']: record for record in records}
    for record in records:
        path = bids_dir / record['path']
        mode, kind, oid = committed[record['path']]
        if kind != 'blob' or (mode == '120000') != path.is_symlink():
            return False
        digest = hashlib.new('sha1' if len(commit) == 40 else 'sha256')
        if path.is_symlink():
            data = os.fsencode(path.readlink())
            digest.update(f'blob {len(data)}\0'.encode('ascii'))
            digest.update(data)
            target = path.resolve()
            relative = target.relative_to(bids_dir).as_posix() if target.is_relative_to(bids_dir) else None
            if relative not in records_by_path:
                # A Git symlink binds only its literal target. For external annex
                # objects the committed key must also bind the available bytes.
                key = re.fullmatch(r'SHA256E?-s([0-9]+)--([a-f0-9]{64})(?:\..*)?', path.readlink().name)
                if key is None or record['sha256'] != key[2] or path.stat().st_size != int(key[1]):
                    return False
        else:
            digest.update(f'blob {path.stat().st_size}\0'.encode('ascii'))
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
        if digest.hexdigest() != oid:
            return False
    return True


def _is_ancestor(root: Path, ancestor, head) -> bool:
    if not _valid_commit(ancestor) or not _valid_commit(head):
        return False
    result = subprocess.run([
        'git', '--no-optional-locks', '-C', str(root), 'merge-base', '--is-ancestor', ancestor, head,
    ], capture_output=True, timeout=30)
    return result.returncode == 0


def _restore_generation_snapshot(meta: dict, current: dict, bids_dir: Path, mriqc_dir: Path) -> list[str]:
    """Bind stored commit snapshots to live content before restoring their identity.

    Milestone-only descendants change HEAD/time but not the generated evidence.
    Everything except these verified snapshot fields remains freshly recomputed.
    """
    errors = []
    stored, live = meta['provenance'], current['provenance']
    head = live['source_datalad_commit']
    source = stored['source_datalad_commit']
    input_commit = live['mriqc_input_commit']
    source_bound = False
    checked_content = {}
    for label, commit in (('source DataLad/Git commit', source), ('MRIQC input commit', input_commit),
                          ('current source commit', head)):
        if not _is_ancestor(bids_dir, commit, head):
            errors.append(f'provenance: {label} is not a known reachable ancestor of current source HEAD')
            continue
        if commit not in checked_content:
            checked_content[commit] = _source_inventory_matches(bids_dir, commit, current['inventory'])
        if not checked_content[commit]:
            errors.append(f'provenance: BIDS inventory content not bound to source commit ({label})')
        elif label == 'source DataLad/Git commit':
            source_bound = True
    if source_bound:
        timestamp = subprocess.run([
            'git', '--no-optional-locks', '-C', str(bids_dir), 'show', '-s', '--format=%cI', source,
        ], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
        live['source_datalad_commit'] = source
        live['source_commit_time'] = timestamp
        current['generation_timestamp'] = timestamp
    stored_mriqc, live_mriqc = stored['mriqc_dataset_commit'], live['mriqc_dataset_commit']
    if stored_mriqc is not None:
        if not _is_ancestor(mriqc_dir, stored_mriqc, live_mriqc):
            errors.append('provenance: MRIQC dataset commit is not a reachable ancestor of current MRIQC HEAD')
        elif not all(_source_inventory_matches(mriqc_dir, commit, current['mriqc_inventory'], mriqc=True)
                     for commit in {stored_mriqc, live_mriqc}):
            errors.append('provenance: MRIQC inventory content is not bound to its dataset commits')
        else:
            live['mriqc_dataset_commit'] = stored_mriqc
    current['generation_metadata_sha256'] = generation_metadata_digest(current)
    return errors


def _provenance_errors(provenance: dict) -> list[str]:
    errors = []
    source = provenance['source_datalad_commit']
    if not _valid_commit(source):
        errors.append('provenance: source DataLad/Git commit is unavailable or unbound')
    mriqc_input = provenance['mriqc_input_commit']
    if not _valid_commit(mriqc_input):
        errors.append('provenance: MRIQC input commit is unknown, incomplete, or conflicting')
    if not provenance['mriqc_versions']:
        errors.append('provenance: MRIQC version is unavailable')
    if not _valid_commit(provenance['package_commits']['network_qa']):
        errors.append('provenance: network_qa package commit is unavailable or unbound')
    if provenance['package_dirty']:
        errors.append('provenance: network_qa package checkout is dirty')
    elif provenance['package_provenance_basis'] == 'source-checkout' and provenance['package_dirty'] is not False:
        errors.append('provenance: network_qa package cleanliness is unknown')
    elif provenance['package_provenance_basis'] not in {'source-checkout', 'pep610-vcs'}:
        errors.append('provenance: network_qa package provenance is unavailable or unbound')
    return errors


def _check(manifest: Path, metadata: Path, bids_dir: Path, *, require_seal: bool):
    digest = ''
    try:
        manifest, metadata, bids_dir = Path(manifest).absolute(), Path(metadata).absolute(), Path(bids_dir)
        if manifest.is_symlink() or metadata.is_symlink():
            raise ValueError('manifest and metadata must be regular files, not symlinks')
        if manifest.suffix != '.tsv' or metadata != manifest.with_suffix('.meta.json'):
            raise ValueError('metadata must be the manifest .meta.json sidecar')
        data, meta_bytes = manifest.read_bytes(), metadata.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        rows = read_manifest(manifest)
        meta = json.loads(meta_bytes)
        if not isinstance(meta, dict) or meta.get('schema_version') != 1:
            raise ValueError('unsupported or malformed metadata schema')
        if not _APPROVAL_FIELDS.issubset(meta):
            raise ValueError('metadata must contain both approval checksum fields')
        approval_values = [meta[field] for field in _APPROVAL_FIELDS]
        if not (all(value is None for value in approval_values) or
                all(isinstance(value, str) and _SHA256.fullmatch(value) for value in approval_values)):
            raise ValueError('approval checksums must both be null or both be SHA-256 strings')
        errors = _review_errors(rows)
        if meta.get('generation_metadata_sha256') != generation_metadata_digest(meta):
            errors.append('metadata: generation metadata checksum mismatch; regenerate the pair')
        sealed = meta.get('approved_manifest_sha256')
        if sealed is not None:
            if sealed != digest:
                errors.append('checksum: approved manifest checksum mismatch')
            if meta.get('approved_metadata_sha256') != _metadata_seal(meta):
                errors.append('checksum: approved metadata checksum mismatch')
        elif require_seal:
            errors.append('approval: manifest is not sealed')
        roots = meta['input_roots']
        if not isinstance(roots, dict) or set(roots) != {'bids_dir', 'mriqc_dir'}:
            raise ValueError('invalid input roots')
        if any(not isinstance(value, str) or not Path(value).is_absolute() for value in roots.values()):
            raise ValueError('input roots must be absolute paths')
        mriqc_dir = Path(roots['mriqc_dir'])
        if bids_dir.is_symlink() or bids_dir.resolve() != Path(roots['bids_dir']):
            raise ValueError('BIDS input root mismatch or symlink')
        if mriqc_dir.is_symlink() or mriqc_dir.resolve() != mriqc_dir:
            raise ValueError('MRIQC input root mismatch or symlink')
        # Approval must never replace evidence, even when supplied a malicious path.
        for path in (manifest.resolve(), metadata.resolve()):
            if path.is_relative_to(mriqc_dir):
                raise ValueError('approval outputs must be outside MRIQC evidence')
            if path.is_relative_to(bids_dir.resolve()):
                relative = path.relative_to(bids_dir.resolve())
                if len(relative.parts) == 1 or relative.parts[0] != 'code':
                    raise ValueError('approval outputs inside BIDS must be under code/')
        baseline_rows, current = collect_decision_evidence(bids_dir, mriqc_dir)
        errors.extend(_restore_generation_snapshot(meta, current, bids_dir.resolve(), mriqc_dir))
        # Full reconstruction proves generation identity as well as its evidence.
        # Human review changes the TSV digest, so comparing its whole digest to
        # the generation digest would reject every legitimate reviewed manifest.
        if _evidence_projection(data) != _evidence_projection(manifest_bytes(baseline_rows)):
            errors.append('evidence: row identities or immutable evidence fields changed')
        immutable_meta = {k: v for k, v in meta.items() if k not in _APPROVAL_FIELDS}
        immutable_current = {k: v for k, v in current.items() if k not in _APPROVAL_FIELDS}
        if immutable_meta != immutable_current:
            errors.append('metadata: generated baseline, inputs, inventory, or provenance mismatch')
        for kind in ('inventory', 'mriqc_inventory'):
            if any(record['status'] != 'available' for record in current[kind]):
                errors.append(f'evidence: unavailable content in {kind}')
        errors.extend(_provenance_errors(current['provenance']))
        # Detect edits during inspection, including manifest edits during parsing.
        if data != manifest.read_bytes() or meta_bytes != metadata.read_bytes():
            errors.append('concurrency: manifest or metadata changed during validation')
        if current['inventory'] != inventory_records(bids_dir) or current['mriqc_inventory'] != mriqc_inventory_records(mriqc_dir):
            errors.append('concurrency: evidence inventory changed during validation')
        return ApprovalResult(not errors, tuple(errors), digest), meta, meta_bytes, data
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, csv.Error,
            subprocess.SubprocessError) as exc:
        return ApprovalResult(False, (f'input: {exc}',), digest), None, None, None


def validate_approval(manifest: Path, metadata: Path, bids_dir: Path) -> ApprovalResult:
    """Read-only verification of a sealed TSV against current evidence/provenance."""
    return _check(manifest, metadata, bids_dir, require_seal=True)[0]


def seal_approval(manifest: Path, metadata: Path, bids_dir: Path) -> ApprovalResult:
    """Validate reviewed decisions and atomically update only approval metadata."""
    result, meta, original_meta, original_manifest = _check(manifest, metadata, bids_dir, require_seal=False)
    if not result.ok or meta['approved_manifest_sha256'] is not None:
        return result
    metadata, manifest = Path(metadata), Path(manifest)
    meta['approved_manifest_sha256'] = result.manifest_sha256
    meta['approved_metadata_sha256'] = _metadata_seal(meta)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.approval-', dir=metadata.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(meta))
        if metadata.read_bytes() != original_meta or manifest.read_bytes() != original_manifest:
            raise ValueError('manifest or metadata changed before sealing')
        temporary.replace(metadata)
        return result
    except (OSError, ValueError) as exc:
        return ApprovalResult(False, (f'publication: {exc}',), result.manifest_sha256)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
