"""Verify that every input label has a complete SPHARM result set."""
import argparse
import json
from pathlib import Path
import sys


REQUIRED_SUFFIXES = (
    '_SPHARM.coef',
    '_SPHARM.vtk',
    '_SPHARM_grid.vtk',
    '_SPHARM_ellalign.coef',
)


def _usable(path):
    """Return False for missing or zero/truncated artifacts."""
    try:
        return path.is_file() and path.stat().st_size > 100
    except OSError:
        return False


def _vtk_geometry_issues(path):
    """Return geometry errors for a VTK polydata artifact."""
    try:
        import vtk
    except ImportError as exc:
        raise RuntimeError('Deep SPHARM verification requires Slicer/Python with vtk') from exc
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None:
        return ['no_polydata']
    issues = []
    if poly.GetNumberOfPoints() < 10:
        issues.append(f'points={poly.GetNumberOfPoints()}')
    if poly.GetNumberOfCells() == 0:
        issues.append('cells=0')
    return issues


def verify(input_dir, output_dir, deep=False):
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    result_root = output_root / 'spharm_results'
    inputs = sorted({
        p.name[:-len(ext)]
        for p in input_root.rglob('*')
        if p.is_file()
        for ext in ('.nii.gz', '.nii', '.hdr')
        if p.name.lower().endswith(ext)
    })
    missing = []
    invalid_geometry = []
    for subject in inputs:
        absent = [suffix for suffix in REQUIRED_SUFFIXES
                  if not _usable(result_root / f'{subject}{suffix}')]
        if absent:
            missing.append({'subject': subject, 'missing': absent})
            continue
        if deep:
            geometry = {}
            for suffix in ('_SPHARM.vtk', '_SPHARM_grid.vtk'):
                issues = _vtk_geometry_issues(result_root / f'{subject}{suffix}')
                if issues:
                    geometry[suffix] = issues
            if geometry:
                invalid_geometry.append({'subject': subject, 'issues': geometry})
    failed = len(missing) + len(invalid_geometry)
    status = {
        'input_dir': str(input_root),
        'output_dir': str(output_root),
        'total_inputs': len(inputs),
        'complete': len(inputs) - failed,
        'failed': failed,
        'deep_geometry_check': bool(deep),
        'required_suffixes': list(REQUIRED_SUFFIXES),
        'missing': missing,
        'invalid_geometry': invalid_geometry,
    }
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--report', default=None)
    parser.add_argument('--deep', action='store_true',
                        help='Read SPHARM and grid VTK files and validate points/cells')
    args = parser.parse_args()
    status = verify(args.input_dir, args.output_dir, deep=args.deep)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).resolve().write_text(json.dumps(status, indent=2), encoding='utf-8')
    if status['failed']:
        print(f"SPHARM incomplete: {status['failed']}/{status['total_inputs']} subjects missing outputs.", file=sys.stderr)
        return 1
    print(f"SPHARM complete: {status['complete']} subjects.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
