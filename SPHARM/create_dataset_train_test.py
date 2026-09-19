"""Create dataset-specific train/test folders from the current ALL status tables.

The existing patient-level assignment in ``current_split_manifest.json`` is
reused.  This keeps both hemispheres of a Subject in the same split and avoids
re-sampling.  Dataset membership and Healthy/TLE status come from the two ALL
status tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "SPHARM" / "split_data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "SPHARM" / "split_data_train_test_current"
DATASETS = ("Ds004469", "Ds005602")
REQUIRED_COLUMNS = ("FileName", "Status", "Dataset")
SUBJECT_RE = re.compile(r"_(sub-[^_]+)_hippocampus_")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_status(path: Path, side: str) -> List[MutableMapping[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(column not in reader.fieldnames for column in REQUIRED_COLUMNS):
            raise ValueError(f"{path} must contain {', '.join(REQUIRED_COLUMNS)}")
        rows: List[MutableMapping[str, str]] = []
        seen = set()
        for row in reader:
            filename = (row.get("FileName") or "").strip()
            status = (row.get("Status") or "").strip()
            dataset = (row.get("Dataset") or "").strip()
            match = SUBJECT_RE.search(filename)
            if not filename or not match:
                raise ValueError(f"invalid FileName in {path}: {filename}")
            if not filename.startswith(side + "_") or not filename.endswith("_SPHARM.coef"):
                raise ValueError(f"filename does not match side {side}: {filename}")
            if status not in {"Healthy", "TLE"} or dataset not in DATASETS:
                raise ValueError(f"invalid status/dataset in {path}: {status}/{dataset}")
            if filename in seen:
                raise ValueError(f"duplicate FileName in {path}: {filename}")
            seen.add(filename)
            rows.append({
                "FileName": filename,
                "Status": status,
                "Dataset": dataset,
                "Subject": match.group(1),
            })
        return rows


def write_status(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(
            ({key: row[key] for key in REQUIRED_COLUMNS} for row in rows),
            key=lambda row: row["FileName"],
        ))


def copy_artifact(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and sha256(destination) == sha256(source):
            return "existing_identical"
        raise FileExistsError(f"destination conflict: {destination}")
    shutil.copy2(source, destination)
    return "copied"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    split_root = args.split_root.resolve()
    output_root = args.output_root.resolve()

    manifest_path = split_root / "current_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = {item["subject_id"]: item["split"] for item in manifest["subjects"]}
    if set(assignments.values()) != {"train", "test"}:
        raise ValueError("current split manifest must contain train and test assignments")

    status_paths = {
        "left": split_root / "ALL_Left_file_status.csv",
        "right": split_root / "ALL_Right_file_status.csv",
    }
    all_dirs = {"left": split_root / "ALL_Left", "right": split_root / "ALL_Right"}
    all_files: Dict[str, Dict[str, List[Path]]] = {}
    for side, directory in all_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"ALL source directory not found: {directory}")
        by_name: Dict[str, List[Path]] = {}
        for entry in directory.rglob("*"):
            if entry.is_file():
                by_name.setdefault(entry.name, []).append(entry)
        all_files[side] = by_name

    rows_by_side = {side: read_status(path, side) for side, path in status_paths.items()}
    dataset_by_subject: Dict[str, str] = {}
    for rows in rows_by_side.values():
        for row in rows:
            previous = dataset_by_subject.setdefault(row["Subject"], row["Dataset"])
            if previous != row["Dataset"]:
                raise ValueError(f"Subject assigned to different datasets: {row['Subject']}")
            if row["Subject"] not in assignments:
                raise ValueError(f"Subject missing from current split manifest: {row['Subject']}")

    output_root.mkdir(parents=True, exist_ok=True)
    result: MutableMapping[str, object] = {
        "schema": "spharm_dataset_train_test_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_policy": {
            "manifest": str(manifest_path),
            "seed": manifest.get("seed"),
            "test_fraction": manifest.get("test_fraction"),
            "patient_level_assignment": True,
        },
        "source_status_tables": {side: str(path) for side, path in status_paths.items()},
        "train_test_split": "created",
        "datasets": {},
    }

    for dataset in DATASETS:
        dataset_result: MutableMapping[str, object] = {}
        for side, rows in rows_by_side.items():
            selected = [row for row in rows if row["Dataset"] == dataset]
            side_root = output_root / f"{dataset}_{side.capitalize()}"
            side_root.mkdir(parents=True, exist_ok=True)
            side_summary: MutableMapping[str, object] = {}
            all_selected_rows: List[Mapping[str, str]] = []
            for split in ("train", "test"):
                split_rows = [row for row in selected if assignments[row["Subject"]] == split]
                destination = side_root / split
                destination.mkdir(parents=True, exist_ok=True)
                copied = 0
                existing = 0
                expected = 0
                missing: List[str] = []
                for row in split_rows:
                    stem = row["FileName"][: -len("_SPHARM.coef")]
                    matching = [
                        path
                        for name, paths in all_files[side].items()
                        if name.startswith(stem + "_")
                        for path in paths
                    ]
                    if not matching:
                        missing.append(row["FileName"])
                        continue
                    expected += len(matching)
                    for source in matching:
                        copied_or_existing = copy_artifact(source, destination / source.name)
                        if copied_or_existing == "copied":
                            copied += 1
                        else:
                            existing += 1
                write_status(destination / "status.csv", split_rows)
                all_selected_rows.extend(split_rows)
                side_summary[split] = {
                    "subjects": len(split_rows),
                    "status_counts": {
                        "Healthy": sum(row["Status"] == "Healthy" for row in split_rows),
                        "TLE": sum(row["Status"] == "TLE" for row in split_rows),
                    },
                    "expected_artifacts": expected,
                    "copied_artifacts": copied,
                    "existing_identical_artifacts": existing,
                    "missing_subject_files": missing,
                    "destination": str(destination),
                }
            write_status(side_root / "status.csv", all_selected_rows)
            dataset_result[side] = side_summary
        result["datasets"][dataset] = dataset_result

    result_path = output_root / "dataset_train_test_manifest.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Manifest: {result_path}")
    missing = sum(
        len(result["datasets"][dataset][side][split]["missing_subject_files"])
        for dataset in DATASETS
        for side in ("left", "right")
        for split in ("train", "test")
    )
    if missing:
        print(f"ERROR: {missing} Subject files were not found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
