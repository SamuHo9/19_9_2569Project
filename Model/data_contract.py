"""Shared data contract for coefficient and XYZ models.

Training can use the pre-generated balanced set while evaluation always uses the
untouched test set.  The balanced set is marked exploratory because its rows are
synthetic/interpolated feature rows and its extraction provenance is not certified
as a fixed-reference clinical pipeline.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parent
PROTOCOL = 'balanced-preaugmented-train-original-test-v1'
RAW_COEF_PROTOCOL = 'original-coef-no-pls-v1'
META = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType']


def subject_key(name):
    value = str(name)
    value = re.sub(r'\.(nii\.gz|nii|vtk|coef)$', '', value, flags=re.I)
    value = re.sub(r'^(left_|right_|lh_|rh_)', '', value, flags=re.I)
    value = re.sub(r'_SPHARM(?:_realigned|_procalign|_ellalign)?$', '', value, flags=re.I)
    value = re.sub(r'_aligned$', '', value, flags=re.I)
    value = re.sub(r'_hippocampus(?:_lh|_rh)?$', '', value, flags=re.I)
    return value


def patient_group_id(name):
    """Return a dataset-independent patient identifier for grouped CV.

    The feature CSV subject names contain side, class and processing suffixes.
    The stable ``sub-*`` token is the patient identity; the fallback keeps the
    normalized subject name when a legacy file has no such token.
    """
    value = str(name).split('/', 1)[-1]
    match = re.search(r'(?i)(?:^|[_-])(sub[-_]?[a-z0-9]+|subject[-_]?[a-z0-9]+|case[-_]?[a-z0-9]+)(?:_|$)', value)
    if match:
        return match.group(1).replace('_', '-').lower()
    return subject_key(value).lower()


def canonical_frame(path, dataset, allow_augmented=False):
    frame = pd.read_csv(path)
    frame = frame.rename(columns={'Group_Name': 'Group', 'Group_Label': 'Class'})
    if 'Subject' not in frame or frame.Subject.isna().any():
        raise ValueError(f'Missing subject identity: {path}')
    augmented_mask = frame.Subject.astype(str).str.contains(r'aug|interp|synth', case=False, regex=True)
    if augmented_mask.any() and not allow_augmented:
        raise ValueError(f'Pre-augmented rows are not allowed in evaluation input: {path}')
    if 'DataType' in frame and not allow_augmented and not frame.DataType.fillna('Original').str.lower().isin(['original', 'real']).all():
        raise ValueError(f'Non-original data in {path}')
    if 'Class' in frame and not frame.Class.isin([0, 1, 2]).all():
        raise ValueError(f'Unknown or invalid class in {path}')
    if 'BinaryClass' not in frame:
        if 'Class' not in frame:
            raise ValueError(f'No labels in {path}')
        frame['BinaryClass'] = (frame.Class == 1).astype(int)
    if not frame.BinaryClass.isin([0, 1]).all():
        raise ValueError(f'Unknown or invalid binary labels in {path}')
    frame['BinaryClass'] = frame.BinaryClass.astype(np.int64)
    if 'Class' in frame and not np.array_equal(frame.BinaryClass, (frame.Class == 1).astype(int)):
        raise ValueError(f'Class/BinaryClass disagreement in {path}')
    frame['Subject'] = dataset + '/' + frame.Subject.map(subject_key)
    if frame.Subject.duplicated().any():
        raise ValueError(f'Duplicate subject in {path}')
    feature_cols = [c for c in frame if c not in META]
    if not feature_cols or not all(re.match(r'^(Coef_\d+|[xyz]_\d+)$', c) for c in feature_cols):
        raise ValueError(f'Unexpected feature columns in {path}')
    if not np.isfinite(frame[feature_cols].to_numpy(dtype=float)).all():
        raise ValueError(f'Nonfinite features in {path}')
    frame['Group'] = frame.get('Group', frame.BinaryClass.map({0: 'Class 0', 1: 'Class 1'}))
    frame['Class'] = frame.get('Class', frame.BinaryClass)
    # Keep the provenance visible to downstream manifests and prevent an
    # augmented row from being mistaken for an original subject.
    if 'DataType' not in frame:
        frame['DataType'] = np.where(augmented_mask.to_numpy(), 'Augmented', 'Original')
    else:
        frame['DataType'] = frame['DataType'].fillna('Original').astype(str)
        frame.loc[augmented_mask, 'DataType'] = 'Augmented'
    return frame[META + feature_cols].sort_values('Subject').reset_index(drop=True)


def validate_output_manifest(path, kind):
    """Reject a canonical output that was extracted with another variant.

    The CSVs in ``Model/Output_Dataset`` are shared by every model.  Checking
    the sidecar here prevents a stale SPHARM or mesh variant from being mixed
    into a run even when the CSV columns happen to look compatible.
    """
    path = Path(path)
    if path.parent.name != 'Output_Dataset':
        return
    sidecar = Path(str(path) + '.json')
    if not sidecar.is_file():
        raise FileNotFoundError(f'Missing output manifest: {sidecar}')
    try:
        payload = json.loads(sidecar.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid output manifest: {sidecar}') from exc
    contract = payload.get('feature_contract')
    if not isinstance(contract, dict):
        raise ValueError(f'Missing feature_contract in output manifest: {sidecar}')
    key = 'coefficient_variant' if kind == 'coef' else 'mesh_variant'
    if contract.get(key) != 'ellalign':
        raise ValueError(f'Expected ellalign {kind} output, got {contract.get(key)!r}: {path}')


def dataset_pair(dataset, side, kind, root=MODEL_ROOT):
    folder = Path(root) / dataset / side
    stem = dataset + '_' + side.capitalize()
    pairs = {}
    for mode in ('coef', 'xyz'):
        suffix = 'coef_features' if mode == 'coef' else 'xyz_coords'
        # Balanced files are produced centrally in Output_Dataset.  Fall back
        # to the legacy per-dataset train file only when a balanced file is not
        # present, so old cohorts remain readable.
        balanced = Path(root) / 'Output_Dataset' / f'{stem}_balanced_{suffix}.csv'
        train_path = balanced if balanced.is_file() else folder / f'{stem}_train_{suffix}.csv'
        test_path = folder / f'{stem}_test_{suffix}.csv'
        # Prefer the newly extracted test file in Output_Dataset as well; this
        # avoids silently mixing a new balanced train with stale test features.
        output_test = Path(root) / 'Output_Dataset' / f'{stem}_test_{suffix}.csv'
        if output_test.is_file():
            test_path = output_test
        validate_output_manifest(train_path, mode)
        validate_output_manifest(test_path, mode)
        pairs[mode] = [
            canonical_frame(train_path, dataset, allow_augmented=True),
            canonical_frame(test_path, dataset, allow_augmented=False),
        ]
    for split_index in (0, 1):
        a, b = pairs['coef'][split_index], pairs['xyz'][split_index]
        if not a.Subject.equals(b.Subject) or not a.BinaryClass.equals(b.BinaryClass):
            raise ValueError(f'Coefficient/XYZ cohort or label mismatch: {dataset}/{side}')
    train, test = pairs[kind]
    if set(train.Subject) & set(test.Subject):
        raise ValueError(f'Train/test subject overlap in {dataset}/{side}')
    return train, test


RAW_COEF_SOURCES = {
    'All_coef': 'ALL',
    'All_coef_Ds004469': 'Ds004469',
    'All_coef_Ds005602': 'Ds005602',
}

RAW_XYZ_SOURCES = RAW_COEF_SOURCES.copy()


def raw_coef_pair(cohort, side, root=MODEL_ROOT):
    """Load the untouched coefficient train/test pair for an All_coef cohort.

    This deliberately bypasses the balanced files.  The balanced folders are
    generated by the PLS-DA interpolation script and therefore cannot be used
    for the no-PLS evaluation protocol.
    """
    if cohort not in RAW_COEF_SOURCES or side not in ('left', 'right'):
        raise ValueError(f'Unsupported raw coefficient cohort or side: {cohort}/{side}')
    source = RAW_COEF_SOURCES[cohort]
    # Read the explicitly built no-PLS cohort.  This keeps the new protocol
    # independent from the shared Output_Dataset directory, which also contains
    # balanced PLS-DA files used by the legacy exploratory protocol.
    cohort_dir = Path(root) / cohort / side
    stem = f'{cohort}_{side.capitalize()}'
    train_path = cohort_dir / f'{stem}_train_coef_features.csv'
    test_path = cohort_dir / f'{stem}_test_coef_features.csv'
    for path in (train_path, test_path):
        sidecar = Path(str(path) + '.json')
        if not sidecar.is_file():
            raise FileNotFoundError(f'Missing no-PLS dataset manifest: {sidecar}')
        try:
            payload = json.loads(sidecar.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Invalid no-PLS dataset manifest: {sidecar}') from exc
        if payload.get('protocol') != RAW_COEF_PROTOCOL:
            raise ValueError(f'Unexpected raw coefficient protocol in {sidecar}')
        contract = payload.get('feature_contract')
        if not isinstance(contract, dict) or contract.get('coefficient_variant') != 'ellalign':
            raise ValueError(f'Expected ellalign coefficient output: {sidecar}')
        if payload.get('pls_da_applied_to_model_input') is not False:
            raise ValueError(f'PLS-DA flag is not false: {sidecar}')
        if int(payload.get('synthetic_rows', 0)) != 0:
            raise ValueError(f'Synthetic rows found in no-PLS manifest: {sidecar}')
    train = canonical_frame(train_path, source, allow_augmented=False)
    test = canonical_frame(test_path, source, allow_augmented=False)
    for name, frame in (('train', train), ('test', test)):
        if not frame['DataType'].fillna('Original').str.lower().isin(['original', 'real']).all():
            raise ValueError(f'All_coef {cohort}/{side} {name} contains non-original rows')
    if set(train.Subject) & set(test.Subject):
        raise ValueError(f'All_coef train/test subject overlap: {cohort}/{side}')
    train_groups = {patient_group_id(v) for v in train.Subject}
    test_groups = {patient_group_id(v) for v in test.Subject}
    if train_groups & test_groups:
        raise ValueError(f'All_coef patient overlap: {cohort}/{side}: {sorted(train_groups & test_groups)[:5]}')
    return train, test


def load_raw_cohort(cohort, side, kind='coef', root=MODEL_ROOT):
    """Load a no-PLS original coefficient cohort for the shared runner."""
    if kind != 'coef':
        raise ValueError('All_coef cohorts contain coefficient features only')
    train, test = raw_coef_pair(cohort, side, root)
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError(f'All_coef train/test schemas differ: {cohort}/{side}')
    feature_cols = [c for c in train if c not in META]
    if len(feature_cols) != 507 or not all(re.match(r'^Coef_\d+$', c) for c in feature_cols):
        raise ValueError(f'All_coef requires 507 coefficient features: {cohort}/{side}')
    if train.BinaryClass.nunique() != 2 or test.BinaryClass.nunique() != 2:
        raise ValueError(f'All_coef requires both classes: {cohort}/{side}')
    return train, test


def raw_xyz_pair(cohort, side, root=MODEL_ROOT):
    """Load original XYZ point clouds for a leakage-free PointNet run."""
    if cohort not in RAW_XYZ_SOURCES or side not in ('left', 'right'):
        raise ValueError(f'Unsupported raw XYZ cohort or side: {cohort}/{side}')
    source = RAW_XYZ_SOURCES[cohort]
    output = Path(root) / 'Output_Dataset'
    stem = f'{source}_{side.capitalize()}'
    train_path = output / f'{stem}_train_xyz_coords.csv'
    test_path = output / f'{stem}_test_xyz_coords.csv'
    validate_output_manifest(train_path, 'xyz')
    validate_output_manifest(test_path, 'xyz')
    train = canonical_frame(train_path, source, allow_augmented=False)
    test = canonical_frame(test_path, source, allow_augmented=False)
    feature_cols = [c for c in train if re.match(r'^[xyz]_\d+$', c)]
    expected = [axis + '_' + str(i) for i in range(1002) for axis in ('x', 'y', 'z')]
    if feature_cols != expected or [c for c in test if re.match(r'^[xyz]_\d+$', c)] != expected:
        raise ValueError(f'Expected x_0,y_0,z_0..x_1001,y_1001,z_1001: {cohort}/{side}')
    if not train['DataType'].str.lower().isin(['original', 'real']).all() or not test['DataType'].str.lower().isin(['original', 'real']).all():
        raise ValueError(f'Raw XYZ pair contains non-original rows: {cohort}/{side}')
    if set(train.Subject) & set(test.Subject):
        raise ValueError(f'Raw XYZ train/test subject overlap: {cohort}/{side}')
    train_groups = {patient_group_id(v) for v in train.Subject}
    test_groups = {patient_group_id(v) for v in test.Subject}
    if train_groups & test_groups:
        raise ValueError(f'Raw XYZ patient overlap: {cohort}/{side}')
    train_hash = set(pd.util.hash_pandas_object(train[feature_cols], index=False).tolist())
    test_hash = set(pd.util.hash_pandas_object(test[feature_cols], index=False).tolist())
    if train_hash & test_hash:
        raise ValueError(f'Identical raw XYZ feature rows across train/test: {cohort}/{side}')
    return train, test


def load_raw_xyz_cohort(cohort, side, root=MODEL_ROOT):
    train, test = raw_xyz_pair(cohort, side, root)
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError(f'Raw XYZ train/test schemas differ: {cohort}/{side}')
    if train.BinaryClass.nunique() != 2 or test.BinaryClass.nunique() != 2:
        raise ValueError(f'Raw XYZ requires both classes: {cohort}/{side}')
    return train, test


def load_cohort(cohort, side, kind='coef', root=MODEL_ROOT):
    if kind not in ('coef', 'xyz') or side not in ('left', 'right'):
        raise ValueError('Invalid feature kind or side')
    if cohort in ('Ds004469', 'Ds005602'):
        train, test = dataset_pair(cohort, side, kind, root)
    elif cohort == 'All_Augment_tain':
        # Use the current ALL_Left/ALL_Right collection. The old code
        # concatenated stale source-dataset files (264/67 rows).
        train, test = dataset_pair('ALL', side, kind, root)
    elif cohort == 'combind_methode':
        pairs = [dataset_pair(ds, side, kind, root) for ds in ('Ds004469', 'Ds005602')]
        train, test = [pd.concat([p[i] for p in pairs], ignore_index=True) for i in (0, 1)]
    elif cohort in ('Ds004469Train_Ds005602test', 'Ds005602Train_Ds004469test'):
        train_ds, test_ds = cohort.split('Train_')
        test_ds = test_ds.removesuffix('test')
        train = pd.concat(dataset_pair(train_ds, side, kind, root), ignore_index=True)
        test = pd.concat(dataset_pair(test_ds, side, kind, root), ignore_index=True)
    else:
        raise ValueError(f'Unsupported cohort: {cohort}')
    train, test = [df.sort_values('Subject').reset_index(drop=True) for df in (train, test)]
    if set(train.Subject) & set(test.Subject):
        raise ValueError('Train/test subjects overlap')
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError('Train/test feature schemas differ')
    feature_cols = [c for c in train if c not in META]
    if set(pd.util.hash_pandas_object(train[feature_cols], index=False)) & set(pd.util.hash_pandas_object(test[feature_cols], index=False)):
        raise ValueError('Identical feature rows across train/test')
    if min(train.BinaryClass.value_counts()) < 5 or train.BinaryClass.nunique() != 2 or test.BinaryClass.nunique() != 2:
        raise ValueError('Need both classes and at least 5 training subjects per class')
    return train, test


def cohort_identity(frame):
    records = frame[['Subject', 'BinaryClass']].sort_values('Subject').to_dict('records')
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def load_training_data(script_file, kind='coef'):
    path = Path(script_file).resolve()
    relative = path.relative_to(MODEL_ROOT)
    cohort = relative.parts[0]
    side = next((p.lower() for p in relative.parts if p.lower() in ('left', 'right')), None)
    train, test = load_cohort(cohort, side, kind)
    # Isolate every new run: historical predictions/plots are never overwritten.
    destination = MODEL_ROOT / 'validated_runs' / cohort / side / path.parent.name
    destination.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_') + path.stem + '_', dir=destination))
    manifest = {'protocol': PROTOCOL, 'script': str(relative), 'kind': kind,
                'train_n': len(train), 'test_n': len(test),
                'train_identity': cohort_identity(train), 'test_identity': cohort_identity(test),
                'train_subjects': train.Subject.tolist(), 'test_subjects': test.Subject.tolist(),
                'train_features_sha256': hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
                'test_features_sha256': hashlib.sha256(test.to_csv(index=False).encode()).hexdigest(),
                'augmentation': 'balanced pre-generated training rows; test remains original/unaugmented',
                'train_data_type_counts': train['DataType'].value_counts().to_dict(),
                'image_preprocessing_provenance': 'exploratory extracted features; fixed-reference independence not certified'}
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.chdir(run_dir)
    (run_dir / 'plots').mkdir()
    (run_dir / 'results').mkdir()
    counts = train['DataType'].value_counts().to_dict()
    print(f'[PROTOCOL] {PROTOCOL}: train={len(train)} ({counts}), test={len(test)} (original only)')
    print(f'[RUN_DIR] {run_dir}')
    return train, test
