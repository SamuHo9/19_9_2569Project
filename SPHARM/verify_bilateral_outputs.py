"""Verify left/right SPHARM outputs with one shared contract."""
import argparse
import json
from pathlib import Path
import re
import sys

from verify_spharm_outputs import verify


def _input_stems(input_dir):
    root = Path(input_dir).resolve()
    stems = set()
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        for ext in ('.nii.gz', '.nii', '.hdr'):
            if path.name.lower().endswith(ext):
                stems.add(path.name[:-len(ext)])
                break
    return stems


def _subject_record(stem):
    """Return side-independent subject identity and the side label.

    ``Healthy``/``TLE`` is deliberately removed from the identity: for a
    patient with unilateral epilepsy the left and right hippocampi can have
    different anatomical labels.  The labels are returned separately so the
    report can expose that expected side-specific difference.
    """
    pattern = (r'^(left|right)_(Healthy|TLE)_(.+)_hippocampus_'
               r'(lh|rh)_aligned$')
    match = re.match(pattern, stem, flags=re.IGNORECASE)
    if match:
        return match.group(3), match.group(1).lower(), match.group(2).lower()
    key = re.sub(r'^(?:left|right)_', '', stem, flags=re.IGNORECASE)
    key = re.sub(r'^(?:Healthy|TLE)_', '', key, flags=re.IGNORECASE)
    key = re.sub(r'_hippocampus_(?:lh|rh)_aligned$', '', key,
                 flags=re.IGNORECASE)
    return key, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left_input_dir', required=True)
    parser.add_argument('--left_output_dir', required=True)
    parser.add_argument('--right_input_dir', required=True)
    parser.add_argument('--right_output_dir', required=True)
    parser.add_argument('--report', default=None)
    parser.add_argument('--deep', action='store_true',
                        help='Read both SPHARM VTK artifacts and validate points/cells')
    args = parser.parse_args()

    left = verify(args.left_input_dir, args.left_output_dir, deep=args.deep)
    right = verify(args.right_input_dir, args.right_output_dir, deep=args.deep)
    left_ids = _input_stems(args.left_input_dir)
    right_ids = _input_stems(args.right_input_dir)
    left_records = {_subject_record(stem)[0]: _subject_record(stem)
                    for stem in left_ids}
    right_records = {_subject_record(stem)[0]: _subject_record(stem)
                     for stem in right_ids}
    left_pair_ids = set(left_records)
    right_pair_ids = set(right_records)
    label_mismatches = [
        {
            'subject': subject,
            'left_label': left_records[subject][2],
            'right_label': right_records[subject][2],
        }
        for subject in sorted(left_pair_ids & right_pair_ids)
        if left_records[subject][2] != right_records[subject][2]
    ]
    report = {
        'contract': 'shared_spharm_v1',
        'left': left,
        'right': right,
        'input_subjects_only_left': sorted(left_pair_ids - right_pair_ids),
        'input_subjects_only_right': sorted(right_pair_ids - left_pair_ids),
        'paired_input_subjects': len(left_pair_ids & right_pair_ids),
        'raw_input_subjects_left': len(left_ids),
        'raw_input_subjects_right': len(right_ids),
        'side_label_mismatch_count': len(label_mismatches),
        'side_label_mismatches': label_mismatches,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).resolve().write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if (left['failed'] or right['failed'] or left_pair_ids != right_pair_ids):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
