"""Create Ds004469/Ds005602 subsets from the current ALL status tables.

The status tables are the source of truth for the cohort and clinical label.
Each row names a primary ``*_SPHARM.coef`` file.  All artifacts sharing that
Subject stem are copied from the current ALL_Left/ALL_Right folders into four
flat, legacy-compatible folders:

    Ds004469_Left, Ds004469_Right, Ds005602_Left, Ds005602_Right

No train/test assignment is made here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "SPHARM" / "split_data"
DATASETS = ("Ds004469", "Ds005602")
REQUIRED_COLUMNS = ("FileName", "Status", "Dataset")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_status(path: Path, expected_side: str) -> List[MutableMapping[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"status table not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(column not in reader.fieldnames for column in REQUIRED_COLUMNS):
            raise ValueError(f"{path} must contain columns: {', '.join(REQUIRED_COLUMNS)}")
        rows = []
        seen = set()
        for row in reader:
            filename = (row.get("FileName") or "").strip()
            status = (row.get("Status") or "").strip()
            dataset = (row.get("Dataset") or "").strip()
            if not filename:
                continue
            if not filename.endswith("_SPHARM.coef"):
                raise ValueError(f"unsupported FileName in {path}: {filename}")
            if status not in {"Healthy", "TLE"}:
                raise ValueError(f"unsupported Status in {path}: {status}")
            if dataset not in DATASETS:
                raise ValueError(f"unsupported Dataset in {path}: {dataset}")
            if filename in seen:
                raise ValueError(f"duplicate FileName in {path}: {filename}")
            if not filename.startswith(expected_side + "_"):
                raise ValueError(f"{filename} does not belong to {expected_side}")
            seen.add(filename)
            rows.append({"FileName": filename, "Status": status, "Dataset": dataset})
        return rows


def copy_one(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and sha256(destination) == sha256(source):
            return "existing_identical"
        raise FileExistsError(f"destination conflict: {destination}")
    shutil.copy2(source, destination)
    return "copied"


def write_status_table(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["FileName"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    args = parser.parse_args(argv)
    split_root = args.split_root.resolve()
    status_paths = {
        "left": split_root / "ALL_Left_file_status.csv",
        "right": split_root / "ALL_Right_file_status.csv",
    }
    all_dirs = {
        "left": split_root / "ALL_Left",
        "right": split_root / "ALL_Right",
    }
    all_files: Dict[str, Dict[str, List[Path]]] = {}
    for side, directory in all_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"ALL source directory not found: {directory}")
        by_name: Dict[str, List[Path]] = {}
        for entry in directory.rglob("*"):
            if entry.is_file():
                by_name.setdefault(entry.name, []).append(entry)
        all_files[side] = by_name

    rows_by_side = {
        side: load_status(path, side) for side, path in status_paths.items()
    }
    manifest: MutableMapping[str, object] = {
        "schema": "spharm_dataset_subsets_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_status_tables": {side: str(path) for side, path in status_paths.items()},
        "source_all_directories": {side: str(path) for side, path in all_dirs.items()},
        "train_test_split": "not_created",
        "datasets": {},
    }

    for dataset in DATASETS:
        dataset_info: MutableMapping[str, object] = {"left": {}, "right": {}}
        for side, rows in rows_by_side.items():
            selected = [row for row in rows if row["Dataset"] == dataset]
            destination_dir = split_root / f"{dataset}_{side.capitalize()}"
            destination_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            existing = 0
            missing_subjects = []
            copied_artifacts = 0
            for row in selected:
                coef_name = row["FileName"]
                stem = coef_name[: -len("_SPHARM.coef")]
                matching = [
                    path for name, paths in all_files[side].items()
                    if name.startswith(stem + "_")
                    for path in paths
                ]
                if not matching:
                    missing_subjects.append(coef_name)
                    continue
                copied_artifacts += len(matching)
                for source in matching:
                    result = copy_one(source, destination_dir / source.name)
                    if result == "copied":
                        copied += 1
                    else:
                        existing += 1
            status_path = destination_dir / "status.csv"
            write_status_table(status_path, selected)
            dataset_info[side] = {
                "subjects": len(selected),
                "status_counts": {
                    "Healthy": sum(row["Status"] == "Healthy" for row in selected),
                    "TLE": sum(row["Status"] == "TLE" for row in selected),
                },
                "copied_artifacts": copied,
                "existing_identical_artifacts": existing,
                "expected_artifacts": copied_artifacts,
                "missing_subject_files": missing_subjects,
                "destination": str(destination_dir),
            }
        manifest["datasets"][dataset] = dataset_info

    manifest_path = split_root / "dataset_subsets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")
    missing = sum(
        len(manifest["datasets"][dataset][side]["missing_subject_files"])
        for dataset in DATASETS
        for side in ("left", "right")
    )
    if missing:
        print(f"ERROR: {missing} Subject files were not found in ALL directories.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
