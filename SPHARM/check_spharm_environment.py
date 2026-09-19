"""Run inside SlicerSALT to verify the dependencies used by the SPHARM pipeline."""
import importlib
import json
from pathlib import Path
import sys

import slicer


def check_environment():
    checks = {}
    for name in ('numpy', 'scipy.ndimage', 'scipy.special', 'vtk'):
        try:
            importlib.import_module(name)
            checks[name] = True
        except Exception as exc:
            checks[name] = str(exc)
    for name in ('segpostprocessclp', 'genparameshclp', 'paratospharmmeshclp'):
        checks[name] = getattr(slicer.modules, name, None) is not None
    result = {'success': all(value is True for value in checks.values()),
              'checks': checks, 'python': sys.version,
              'scope': 'Runtime imports and CLI module registration; no MRI processing performed'}
    script = next((Path(arg).resolve() for arg in sys.argv
                   if arg.endswith('check_spharm_environment.py')), None)
    if script is None:
        script = Path(__file__).resolve()
    path = script.with_name('environment_check.json')
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print('Environment check saved to: ' + str(path), flush=True)
    slicer.util.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    check_environment()
