"""Track pretraining data as part of the detector's validation contract."""
import json
from pathlib import Path
from utils.split_contract import read_split, sha256_file


def verify_stage1_contract(cfg, root, split_summary):
    image = cfg.get('MODEL', {}).get('IMG_CLS', {})
    policy = image.get('LINEAGE_POLICY', 'unverified')
    if policy == 'unverified':
        return {'policy': policy, 'formal_holdout_certified': False}
    if policy not in ('shared_legacy', 'strict_holdout'):
        raise ValueError('unknown Stage 1 LINEAGE_POLICY')
    root = Path(root)
    def resolve(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path
    checkpoint = resolve(image['MODEL_PATH'])
    manifest_path = resolve(image['PROVENANCE_PATH'])
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Stage 1 checkpoint hash differs from provenance')
    if image['NAME'] != manifest['architecture']:
        raise ValueError('Stage 1 architecture differs from provenance')
    summary = {'policy': policy, 'checkpoint_sha256': manifest['checkpoint_sha256'],
               'manifest_sha256': sha256_file(manifest_path),
               'formal_holdout_certified': False, 'provenance': manifest}
    if policy == 'shared_legacy':
        # Explicitly allowed for controlled development comparisons, without
        # converting unknown/test-selected upstream provenance to a clean claim.
        if bool(cfg.VAL.get('RUN_CONDITIONAL_AT_END', False)):
            raise ValueError('shared-legacy development runs must not trigger automatic test evaluation')
        summary['limitation'] = 'Stage 1 fit/selection splits are not certified; shared checkpoint controls front-end differences only.'
        return summary
    if manifest.get('lineage_status') != 'documented':
        raise ValueError('strict_holdout requires documented Stage 1 lineage')
    if not manifest.get('fit_splits') or not manifest.get('evidence_files'):
        raise ValueError('strict_holdout requires fit splits and supporting run evidence')
    if 'selection_splits' not in manifest:
        raise ValueError('selection_splits must be recorded, including an explicit empty list')
    for record in manifest['evidence_files']:
        if sha256_file(resolve(record['path'])) != record['sha256']:
            raise ValueError('Stage 1 run evidence hash mismatch')
    rows = {}
    for role in ('fit_splits', 'selection_splits'):
        rows[role] = set()
        for record in manifest[role]:
            path = resolve(record['path'])
            if sha256_file(path) != record['sha256']:
                raise ValueError('Stage 1 split hash mismatch')
            rows[role].update(read_split(path))
    official_train = set(read_split(Path(split_summary['paths']['official_train'])))
    official_test = set(read_split(Path(split_summary['paths']['official_test'])))
    used = rows['fit_splits'] | rows['selection_splits']
    if not used.issubset(official_train) or used & official_test:
        raise ValueError('Stage 1 fit/selection used frames outside official train')
    if split_summary['protocol_phase'] == 'stage2_selection':
        detector_val = set(read_split(Path(split_summary['paths']['val'])))
        if used & detector_val:
            raise ValueError('Stage 1 fit/selection overlaps outer detector validation')
    summary['formal_holdout_certified'] = True
    summary['fit_count'] = len(rows['fit_splits'])
    summary['selection_count'] = len(rows['selection_splits'])
    return summary
