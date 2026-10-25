#!/usr/bin/env python3
"""Inspect VYNDRA datasets and cache a single file inventory.

This script does not open image pixels, train models, or modify raw data.
Directory-derived class counts are provisional until the layout is verified.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Khana": PROJECT_ROOT / "data/raw/khana/khana",
    "Food-101 HF cache": PROJECT_ROOT / "data/raw/food101/hf_cache",
    "Indian Food Images": PROJECT_ROOT
    / "data/raw/indian_food_images/Indian Food Images",
}

INDEX_PATH = PROJECT_ROOT / "data/processed/file_index.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".avif",
}

ANNOTATION_WORDS = (
    "annotation", "annotations", "mask", "masks",
    "segmentation", "label", "labels",
)


def is_annotation_or_mask(path: Path) -> bool:
    """Flag likely annotation/mask paths by name only; verify manually."""
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    return any(
        word in part
        for part in parts
        for word in ANNOTATION_WORDS
    ) or any(word in name for word in ANNOTATION_WORDS)


def scan_dataset(dataset_name: str, root: Path) -> list[dict]:
    """Walk one dataset once and record file metadata without reading contents."""
    rows = []
    if not root.exists():
        print(f"\nWARNING: {dataset_name} path does not exist: {root}")
        return rows

    print(f"\nScanning {dataset_name}: {root}")
    print("This records paths and file names; it does not read image pixels.")

    for current_root, dirs, files in os.walk(root):
        # Ignore hidden/system folders to avoid irrelevant metadata.
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".") and d != "__MACOSX"
        )

        current = Path(current_root)
        for filename in files:
            if filename.startswith("."):
                continue

            path = current / filename
            ext = path.suffix.lower()
            try:
                rel_path = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                rel_path = str(path)

            rows.append({
                "dataset": dataset_name,
                "path": str(path),
                "relative_path": rel_path,
                "extension": ext,
                "parent_folder": path.parent.name,
                "is_image": str(ext in IMAGE_EXTENSIONS).lower(),
                "is_annotation_or_mask": str(
                    is_annotation_or_mask(path)
                ).lower(),
            })

    print(f"Recorded {len(rows):,} file paths.")
    return rows


def write_index(rows: list[dict]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset", "path", "relative_path", "extension",
        "parent_folder", "is_image", "is_annotation_or_mask",
    ]
    with INDEX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved file index: {INDEX_PATH}")


def read_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    with INDEX_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def print_tree(root: Path, max_depth: int = 2) -> None:
    """Print folder names only; do not list every file."""
    if not root.exists():
        print("  [MISSING]")
        return

    print(f"  {root}")
    root_depth = len(root.parts)
    shown = 0
    for current_root, dirs, _files in os.walk(root):
        current = Path(current_root)
        depth = len(current.parts) - root_depth
        if depth >= max_depth:
            dirs[:] = []
        for folder in sorted(dirs):
            print(f"  {'  ' * (depth + 1)}{folder}/")
            shown += 1
            if shown >= 60:
                print("  ... folder display capped at 60 entries")
                return


def print_text_preview(path: Path, lines: int) -> None:
    print(f"\n--- {path} (first {lines} lines) ---")
    if not path.exists():
        print("File not found.")
        return
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for number, line in enumerate(handle):
                if number >= lines:
                    break
                print(line.rstrip())
    except OSError as exc:
        print(f"Could not read file: {exc}")


def print_csv_preview(path: Path, rows_to_show: int = 10) -> None:
    print(f"\n--- {path} (first {rows_to_show} rows) ---")
    if not path.exists():
        print("File not found.")
        return
    try:
        with path.open(
            "r", newline="", encoding="utf-8-sig", errors="replace"
        ) as handle:
            reader = csv.reader(handle)
            for number, row in enumerate(reader):
                if number >= rows_to_show:
                    break
                print(" | ".join(row))
    except (OSError, csv.Error) as exc:
        print(f"Could not read CSV: {exc}")


def report_dataset(name: str, root: Path, rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"DATASET: {name}")
    print(f"Expected root: {root}")
    print(f"Exists: {root.exists()}")
    print("\nTop-level folder structure:")
    print_tree(root)

    dataset_rows = [r for r in rows if r["dataset"] == name]
    if not dataset_rows:
        print("\nNo indexed files for this dataset.")
        return

    image_rows = [r for r in dataset_rows if r["is_image"] == "true"]
    annotation_rows = [
        r for r in dataset_rows
        if r["is_annotation_or_mask"] == "true"
    ]

    print(f"\nTotal indexed files: {len(dataset_rows):,}")
    print(f"Files with recognised image extensions: {len(image_rows):,}")
    print(f"Paths flagged by annotation/mask naming: {len(annotation_rows):,}")

    print("\nFile extension counts (all files):")
    for ext, count in Counter(
        r["extension"] or "[no extension]" for r in dataset_rows
    ).most_common(20):
        print(f"  {ext}: {count:,}")

    print("\nSample image paths:")
    for row in image_rows[:8]:
        print(f"  {row['path']}")

    print("\nSample paths flagged as possible annotations/masks:")
    if annotation_rows:
        for row in annotation_rows[:8]:
            print(f"  {row['path']}")
    else:
        print("  None flagged by filename/path heuristics.")

    print("\nDirectory-based image counts (PROVISIONAL, not verified classes):")
    counts = Counter(r["parent_folder"] for r in image_rows)
    if not counts:
        print("  No image files found by extension.")
    else:
        print("  Highest 10 parent-folder counts:")
        for folder, count in counts.most_common(10):
            print(f"    {folder}: {count:,}")
        print("  Lowest 10 non-empty parent-folder counts:")
        for folder, count in sorted(
            counts.items(), key=lambda item: (item[1], item[0])
        )[:10]:
            print(f"    {folder}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scan the datasets and replace the cached file index.",
    )
    args = parser.parse_args()

    if INDEX_PATH.exists() and not args.refresh:
        print(f"Using existing file index: {INDEX_PATH}")
        rows = read_index()
    else:
        rows = []
        for name, root in DATASETS.items():
            rows.extend(scan_dataset(name, root))
        write_index(rows)

    for name, root in DATASETS.items():
        report_dataset(name, root, rows)

    khana_root = DATASETS["Khana"]
    print_text_preview(khana_root / "labels.txt", 20)
    print_csv_preview(khana_root / "taxonomy.csv", 10)

    print("\n" + "=" * 72)
    print("INSPECTION COMPLETE")
    print(f"Cached inventory: {INDEX_PATH}")
    print("Directory counts are provisional until the layouts are reviewed.")
    print("Do not use these counts as evaluation metrics.")


if __name__ == "__main__":
    main()