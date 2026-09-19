"""Verify the current patient-level SPHARM split from its manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "SPHARM" / "split_data"


def verify(split_root: Path) -> Dict[str, object]:
    manifest_path = split_root / "current_split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"current split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = list(manifest["required_suffixes"])
    subjects = list(manifest["subjects"])
    source_dirs = {side: Path(manifest["source"][side]) for side in ("left", "right")}
    source_files = {
        side: {entry.name for entry in source_dir.iterdir() if entry.is_file()}
        for side, source_dir in source_dirs.items()
    }
    errors: List[dict] = []
    split_counts = {"train": 0, "test": 0}
    side_counts = {"left": {"train": 0, "test": 0}, "right": {"train": 0, "test": 0}}

    for item in subjects:
        subject_id = item["subject_id"]
        split = item["split"]
        if split not in split_counts:
            errors.append({"subject_id": subject_id, "error": f"invalid split {split!r}"})
            continue
        split_counts[split] += 1
        for side in ("left", "right"):
            label_key = f"{side}_label"
            if not item.get(label_key):
                continue
            side_counts[side][split] += 1
            stem = f"{side}_{item[label_key]}_{subject_id}_hippocampus_{'lh' if side == 'left' else 'rh'}_aligned"
            destination = split_root / f"ALL_{side.capitalize()}" / split
            missing = [suffix for suffix in required if not (destination / f"{stem}{suffix}").is_file()]
            if missing:
                errors.append({"subject_id": subject_id, "side": side, "split": split, "missing": missing})
            expected_artifacts = {
                name for name in source_files[side] if name.startswith(stem + "_")
            }
            destination_names = {
                entry.name for entry in destination.iterdir() if entry.is_file()
            } if destination.is_dir() else set()
            missing_artifacts = sorted(expected_artifacts - destination_names)
            if missing_artifacts:
                errors.append({
                    "subject_id": subject_id,
                    "side": side,
                    "split": split,
                    "missing_artifacts": missing_artifacts,
                })

    result = {
        "schema": "spharm_current_split_verification_v1",
        "split_root": str(split_root),
        "manifest": str(manifest_path),
        "required_suffixes": required,
        "subjects_total": len(subjects),
        "split_counts": split_counts,
        "side_counts": side_counts,
        "source_artifact_files": {side: len(files) for side, files in source_files.items()},
        "errors": errors,
        "complete": not errors,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    result = verify(args.split_root.resolve())
    report_path = (args.report or (args.split_root.resolve() / "current_split_completeness.json")).resolve()
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    if result["errors"]:
        print(f"Current split incomplete: {len(result['errors'])} errors.", file=sys.stderr)
        return 1
    print("Current split complete: every assigned side has the required artifacts.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
