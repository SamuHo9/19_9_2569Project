"""Run current regression checks; the pre-fix script is in maintenance_backup_20260918.

Historical audit_results.json remains unchanged.
"""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('verify_changes.py')), run_name='__main__')
