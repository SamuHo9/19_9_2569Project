"""Build one consistent ML feature contract for every split.

The exporter runs the XYZ and coefficient extractors with the same SPHARM
variant, normalizes their metadata columns, checks that their subject/label
sets are identical, and writes the canonical files consumed by ``Model``.
Raw SPHARM/ICP files are never modified.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SETS = ('ALL_Left', 'ALL_Right', 'Ds004469_Left', 'Ds004469_Right',
        'Ds005602_Left', 'Ds005602_Right')
SPLITS = ('train', 'test', 'balanced')
META = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_extractor(python, script, source, args, processing):
    command = [str(python), str(script), '--spharm_dir', str(source)] + list(args)
    result = subprocess.run(command, cwd=str(processing), capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    if result.stdout:
        print(result.stdout, end='')
    if result.returncode:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end='')
        raise RuntimeError(f'Extractor failed ({result.returncode}): {" ".join(command)}')
    return result


def _as_label(value, name):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f'Non-integer {name} label: {value!r}')


def normalize_frame(frame, kind, split):
    if kind == 'xyz':
        frame = frame.rename(columns={'Group_Name': 'Group', 'Group_Label': 'Class'})
    required = {'Subject', 'Group', 'Class'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{kind} output is missing columns: {sorted(missing)}')
    frame = frame.copy()
    frame['Class'] = frame['Class'].map(lambda x: _as_label(x, 'Class'))
    if not frame['Class'].isin((0, 1, 2)).all():
        raise ValueError(f'{kind}/{split} contains an unknown Class label')
    expected_binary = (frame['Class'] == 1).astype(np.int64)
    if 'BinaryClass' in frame:
        frame['BinaryClass'] = frame['BinaryClass'].map(lambda x: _as_label(x, 'BinaryClass'))
        if not frame['BinaryClass'].isin((0, 1)).all() or not np.array_equal(frame['BinaryClass'], expected_binary):
            raise ValueError(f'{kind}/{split} Class/BinaryClass mismatch')
    else:
        frame['BinaryClass'] = expected_binary
    if frame['Subject'].isna().any():
        raise ValueError(f'{kind}/{split} has missing Subject values')
    frame['Subject'] = frame['Subject'].astype(str).str.strip()
    if (frame['Subject'] == '').any() or frame['Subject'].duplicated().any():
        raise ValueError(f'{kind}/{split} has missing or duplicate Subject values')
    augmented = frame['Subject'].str.contains(r'aug|interp|synth', case=False, regex=True)
    if split == 'test' and augmented.any():
        raise ValueError(f'Augmented subjects found in test {kind} output')
    frame['DataType'] = np.where(augmented, 'Augmented', 'Original')
    feature_pattern = r'^([xyz]_\d+)$' if kind == 'xyz' else r'^(Coef_\d+)$'
    features = [c for c in frame.columns if re.match(feature_pattern, str(c))]
    if not features or len(features) != len(set(features)):
        raise ValueError(f'{kind}/{split} has no valid feature columns')
    values = frame[features].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'{kind}/{split} contains non-finite feature values')
    frame[features] = values
    return frame[META + features].sort_values('Subject').reset_index(drop=True), features


def read_contract(path):
    sidecar = Path(str(path) + '.json')
    if not sidecar.is_file():
        raise FileNotFoundError(f'Missing extractor manifest: {sidecar}')
    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    contract = payload.get('feature_contract')
    if not isinstance(contract, dict):
        raise ValueError(f'Missing feature_contract in {sidecar}')
    return contract


def main():
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument('--project_root', type=Path, default=default_root)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--strict_provenance', action='store_true',
                        help='Do not allow historical inputs without _processing.json sidecars')
    args = parser.parse_args()

    project = args.project_root.resolve()
    split_root = project / 'SPHARM' / 'split_data'
    output = project / 'Model' / 'Output_Dataset'
    processing = project / 'Data_Processing'
    xyz_script = processing / 'extract_ml_features.py'
    coef_script = processing / 'extract_ml_features_coef.py'
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for name in SETS:
        set_root = split_root / name
        if not set_root.is_dir():
            raise FileNotFoundError(f'Missing split directory: {set_root}')
        for split in SPLITS:
            source = set_root / 'train' / 'balanced' if split == 'balanced' else set_root / split
            if not source.is_dir():
                raise FileNotFoundError(f'Missing {name}/{split} directory: {source}')
            generated_root = set_root / 'train' / 'ml_features' if split == 'balanced' else set_root / 'ml_features'
            run_extractor(args.python, xyz_script, source,
                          ['--mesh_variant', 'ellalign'], processing)
            coef_args = ['--coef_variant', 'ellalign']
            if not args.strict_provenance:
                coef_args.append('--allow-missing-provenance')
            run_extractor(args.python, coef_script, source, coef_args, processing)

            xyz_raw = generated_root / 'spharm_xyz_coords.csv'
            coef_raw = generated_root / f'{split}_coef_features.csv'
            edges_raw = generated_root / 'mesh_edges.csv'
            if not xyz_raw.is_file() or not coef_raw.is_file():
                raise FileNotFoundError(f'Extractor did not create both outputs for {name}/{split}')
            xyz_contract = read_contract(xyz_raw)
            coef_contract = read_contract(coef_raw)
            if xyz_contract.get('mesh_variant') != 'ellalign':
                raise ValueError(f'XYZ variant mismatch for {name}/{split}: {xyz_contract}')
            if coef_contract.get('coefficient_variant') != 'ellalign':
                raise ValueError(f'Coefficient variant mismatch for {name}/{split}: {coef_contract}')

            xyz, xyz_features = normalize_frame(pd.read_csv(xyz_raw), 'xyz', split)
            coef, coef_features = normalize_frame(pd.read_csv(coef_raw), 'coef', split)
            if not xyz['Subject'].equals(coef['Subject']) or not xyz['BinaryClass'].equals(coef['BinaryClass']):
                raise ValueError(f'XYZ/coefficient subject or label mismatch for {name}/{split}')
            if xyz_contract.get('num_points') != len(xyz_features) // 3:
                raise ValueError(f'XYZ point count mismatch for {name}/{split}')
            if len(coef_features) != 507:
                raise ValueError(f'Expected 507 SPHARM coefficient values, got {len(coef_features)} for {name}/{split}')

            xyz_path = output / f'{name}_{split}_xyz_coords.csv'
            coef_path = output / f'{name}_{split}_coef_features.csv'
            xyz.to_csv(xyz_path, index=False, float_format='%.8f')
            coef.to_csv(coef_path, index=False, float_format='%.8f')
            # Edges are topology metadata; keep a split-labelled copy for
            # compatibility with the existing output layout.
            if edges_raw.is_file():
                shutil.copy2(edges_raw, output / f'{name}_{split}_mesh_edges.csv')

            xyz_manifest = {
                'csv_sha256': sha256(xyz_path),
                'feature_contract': dict(xyz_contract),
                'dataset': name, 'split': split,
                'source_dir': str(source), 'rows': len(xyz),
                'schema': META + xyz_features,
            }
            coef_manifest = {
                'csv_sha256': sha256(coef_path),
                'feature_contract': dict(coef_contract),
                'dataset': name, 'split': split,
                'source_dir': str(source), 'rows': len(coef),
                'schema': META + coef_features,
            }
            Path(str(xyz_path) + '.json').write_text(json.dumps(xyz_manifest, indent=2), encoding='utf-8')
            Path(str(coef_path) + '.json').write_text(json.dumps(coef_manifest, indent=2), encoding='utf-8')
            records.append({
                'Name': f'{name}_{split}', 'Source': str(source),
                'Rows': len(xyz), 'OriginalRows': int((xyz.DataType == 'Original').sum()),
                'AugmentedRows': int((xyz.DataType == 'Augmented').sum()),
                'XYZFeatures': len(xyz_features), 'CoefFeatures': len(coef_features),
                'MeshVariant': xyz_contract['mesh_variant'],
                'CoefficientVariant': coef_contract['coefficient_variant'],
                'XYZFile': xyz_path.name, 'CoefFile': coef_path.name,
                'ProvenanceStatus': coef_contract.get('provenance_status', 'certified'),
            })
            print(f'[OK] {name}/{split}: rows={len(xyz)} xyz={len(xyz_features)} coef={len(coef_features)}')

    manifest_path = output / 'feature_extraction_manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / 'export_contract.json').write_text(json.dumps({
        'version': 'ml-output-v2',
        'metadata': META,
        'sets': list(SETS), 'splits': list(SPLITS),
        'mesh_variant': 'ellalign', 'coefficient_variant': 'ellalign',
        'point_order': 'x,y,z', 'coefficient_values': 507,
        'provenance_mode': 'strict' if args.strict_provenance else 'allow_missing_exploratory',
    }, indent=2), encoding='utf-8')
    print(f'Wrote {len(records)} paired datasets to {output}')


if __name__ == '__main__':
    main()
