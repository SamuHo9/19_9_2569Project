"""Create a reproducible train/test split from the current SPHARM outputs.

Unlike the legacy splitter, this script does not read historical feature CSVs.
It discovers complete subjects directly from the current ``spharm_results``
folders, assigns each patient/Subject ID to one split, and then copies the
available left and right artifacts to that same split.  This prevents a
patient with both hemispheres from appearing in train on one side and test on
the other side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEFT_SOURCE = PROJECT_ROOT / "ICP" / "output_left_hippocampus" / "spharm_results"
DEFAULT_RIGHT_SOURCE = PROJECT_ROOT / "ICP" / "output_right_hippocampus" / "spharm_results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "SPHARM" / "split_data_current"
REQUIRED_SUFFIXES = (
    "_SPHARM.coef",
    "_SPHARM.vtk",
    "_SPHARM_grid.vtk",
    "_SPHARM_ellalign.coef",
)
COEF_RE = re.compile(
    r"^(?P<side>left|right)_(?P<label>Healthy|TLE)_(?P<subject>sub-[^_]+)"
    r"_hippocampus_(?P<hemi>lh|rh)_aligned_SPHARM\.coef$"
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_side(source_dir: Path, expected_side: str) -> Tuple[Dict[str, MutableMapping[str, object]], List[dict]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"SPHARM source directory not found: {source_dir}")
    subjects: Dict[str, MutableMapping[str, object]] = {}
    incomplete: List[dict] = []
    for coef in sorted(source_dir.glob("*_SPHARM.coef"), key=lambda p: p.name.lower()):
        match = COEF_RE.match(coef.name)
        if not match or match.group("side") != expected_side:
            raise ValueError(f"unexpected coefficient filename in {source_dir}: {coef.name}")
        subject_id = match.group("subject")
        if subject_id in subjects:
            raise ValueError(f"duplicate {expected_side} Subject ID: {subject_id}")
        stem = coef.name[: -len("_SPHARM.coef")]
        artifacts = sorted(
            (entry for entry in source_dir.iterdir() if entry.is_file() and entry.name.startswith(stem + "_")),
            key=lambda p: p.name.lower(),
        )
        missing = [suffix for suffix in REQUIRED_SUFFIXES if not (source_dir / f"{stem}{suffix}").is_file()]
        if missing:
            incomplete.append({"subject_id": subject_id, "stem": stem, "missing": missing})
            continue
        subjects[subject_id] = {
            "subject_id": subject_id,
            "label": match.group("label"),
            "stem": stem,
            "source": str(source_dir),
            "artifacts": [str(path) for path in artifacts],
        }
    return subjects, incomplete


def pattern_for(item: Mapping[str, MutableMapping[str, object]]) -> str:
    left = str(item["left"]["label"]) if "left" in item else "missing"
    right = str(item["right"]["label"]) if "right" in item else "missing"
    return f"{left}/{right}"


def assign_splits(
    subjects: Mapping[str, MutableMapping[str, object]],
    test_fraction: float,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    grouped: Dict[str, List[str]] = {}
    for subject_id, item in subjects.items():
        grouped.setdefault(pattern_for(item), []).append(subject_id)

    assignments: Dict[str, str] = {}
    pattern_test_counts: Dict[str, int] = {}
    for pattern in sorted(grouped):
        ids = sorted(grouped[pattern])
        random.Random(f"{seed}:{pattern}").shuffle(ids)
        n_test = int(math.floor(len(ids) * test_fraction + 0.5))
        if len(ids) >= 2 and test_fraction > 0:
            n_test = max(1, min(len(ids) - 1, n_test))
        elif len(ids) == 1:
            n_test = 0
        for subject_id in ids[:n_test]:
            assignments[subject_id] = "test"
        for subject_id in ids[n_test:]:
            assignments[subject_id] = "train"
        pattern_test_counts[pattern] = n_test
    return assignments, pattern_test_counts


def copy_side(
    subjects: Mapping[str, MutableMapping[str, object]],
    assignments: Mapping[str, str],
    side: str,
    output_root: Path,
) -> dict:
    counts = {"train": 0, "test": 0}
    file_counts = {"train": 0, "test": 0}
    byte_counts = {"train": 0, "test": 0}
    for split in ("train", "test"):
        destination_dir = output_root / f"ALL_{side.capitalize()}" / split
        if destination_dir.exists() and any(destination_dir.iterdir()):
            raise FileExistsError(
                f"destination is not empty: {destination_dir}; use a new output root or --overwrite"
            )
        destination_dir.mkdir(parents=True, exist_ok=True)

    for subject_id, item in sorted(subjects.items()):
        split = assignments[subject_id]
        destination_dir = output_root / f"ALL_{side.capitalize()}" / split
        counts[split] += 1
        for source_string in item["artifacts"]:
            source = Path(source_string)
            destination = destination_dir / source.name
            if destination.exists():
                if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
                    raise FileExistsError(f"destination conflict: {destination}")
                continue
            shutil.copy2(source, destination)
            file_counts[split] += 1
            byte_counts[split] += source.stat().st_size
    return {
        "subjects": counts,
        "copied_files": file_counts,
        "copied_bytes": byte_counts,
        "output_root": str(output_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-source", type=Path, default=DEFAULT_LEFT_SOURCE)
    parser.add_argument("--right-source", type=Path, default=DEFAULT_RIGHT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1")
    left, left_incomplete = discover_side(args.left_source.resolve(), "left")
    right, right_incomplete = discover_side(args.right_source.resolve(), "right")
    if left_incomplete or right_incomplete:
        raise ValueError(
            "current SPHARM source contains incomplete subjects: "
            f"left={len(left_incomplete)}, right={len(right_incomplete)}"
        )

    subjects: Dict[str, MutableMapping[str, object]] = {}
    for subject_id, item in left.items():
        subjects.setdefault(subject_id, {})["left"] = item
    for subject_id, item in right.items():
        subjects.setdefault(subject_id, {})["right"] = item

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset in ("ALL_Left", "ALL_Right"):
        dataset_root = output_root / dataset
        if args.overwrite and dataset_root.exists():
            raise ValueError("--overwrite is intentionally disabled for directory cleanup; choose a new output root")

    assignments, pattern_test_counts = assign_splits(subjects, args.test_fraction, args.seed)
    left_result = copy_side(left, assignments, "left", output_root)
    right_result = copy_side(right, assignments, "right", output_root)
    manifest = {
        "schema": "spharm_current_subject_split_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "patient_subject_level_stratified_split",
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "required_suffixes": list(REQUIRED_SUFFIXES),
        "source": {"left": str(args.left_source.resolve()), "right": str(args.right_source.resolve())},
        "subjects_total": len(subjects),
        "paired_subjects": len(set(left) & set(right)),
        "left_only_subjects": len(set(left) - set(right)),
        "right_only_subjects": len(set(right) - set(left)),
        "split_subject_counts": {
            "train": sum(value == "train" for value in assignments.values()),
            "test": sum(value == "test" for value in assignments.values()),
        },
        "pattern_counts": {
            pattern: {
                "total": sum(pattern_for(subjects[sid]) == pattern for sid in subjects),
                "test": pattern_test_counts[pattern],
                "train": sum(pattern_for(subjects[sid]) == pattern for sid in subjects) - pattern_test_counts[pattern],
            }
            for pattern in sorted(pattern_test_counts)
        },
        "side_results": {"left": left_result, "right": right_result},
        "subjects": [
            {
                "subject_id": subject_id,
                "split": assignments[subject_id],
                "pattern": pattern_for(subjects[subject_id]),
                "left_label": subjects[subject_id].get("left", {}).get("label"),
                "right_label": subjects[subject_id].get("right", {}).get("label"),
            }
            for subject_id in sorted(subjects)
        ],
    }
    manifest_path = output_root / "current_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "subjects"}, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
