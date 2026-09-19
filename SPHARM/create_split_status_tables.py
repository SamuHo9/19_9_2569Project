"""Create two compact CSV tables for current left/right SPHARM labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "SPHARM" / "split_data"


def create_table(split_root: Path, side: str, output_path: Path) -> Dict[str, int]:
    manifest_path = split_root / "current_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    counts: Dict[str, int] = {}
    hemisphere = "lh" if side == "left" else "rh"
    summary_path = split_root / "spharm_processing_summary.csv"
    dataset_map: Dict[str, str] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8-sig") as sf:
            reader = csv.DictReader(sf)
            for srow in reader:
                sub_id = srow.get("Subject_ID")
                ds = srow.get("Dataset")
                if sub_id and ds:
                    dataset_map[sub_id] = ds

    for item in manifest["subjects"]:
        label = item.get(f"{side}_label")
        if not label:
            continue
        split = item["split"]
        stem = f"{side}_{label}_{item['subject_id']}_hippocampus_{hemisphere}_aligned"
        filename = f"{stem}_SPHARM.coef"
        status = label
        dataset = dataset_map.get(item["subject_id"], "")
        row_dict = {"FileName": filename, "Status": status, "Dataset": dataset}
        rows.append(row_dict)
        counts[status] = counts.get(status, 0) + 1

    rows.sort(key=lambda row: (row["Status"], row["FileName"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["FileName", "Status", "Dataset"])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    args = parser.parse_args(argv)
    split_root = args.split_root.resolve()
    outputs = {
        "left": split_root / "ALL_Left_file_status.csv",
        "right": split_root / "ALL_Right_file_status.csv",
    }
    summary = {}
    for side, path in outputs.items():
        summary[side] = {
            "path": str(path),
            "rows_by_status": create_table(split_root, side, path),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
