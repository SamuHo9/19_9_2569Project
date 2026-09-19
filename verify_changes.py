"""Current regression runner; preserves the original audit_results.json evidence."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ignored = {'FastSurfer', 'Backup', 'build', 'venv', '.git', 'maintenance_backup_20260918', 'validated_runs'}
sources = [p for p in ROOT.rglob('*.py') if not any(part in ignored for part in p.parts)]
errors = []
for path in sources:
    try:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        errors.append({'file': str(path.relative_to(ROOT)), 'error': str(exc)})
env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONWARNINGS='ignore::DeprecationWarning')
result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), '-v'],
                         cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
log = result.stdout + result.stderr
(ROOT/'regression_results.txt').write_text(log, encoding='utf-8')
summary = {'syntax_files': len(sources), 'syntax_errors': errors, 'regression_exit_code': result.returncode,
           'runtime': sys.version, 'verification': 'unit/integration tests; MRI/Qt/torch execution not covered',
           'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(ROOT/'regression_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(log)
print(f'Syntax: {len(sources)} files, {len(errors)} errors')
sys.exit(1 if errors or result.returncode else 0)
