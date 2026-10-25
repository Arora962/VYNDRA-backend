#!/usr/bin/env python3
"""Smoke-test the trained classifier on individual images and save predictions to CSV.

This script checks that inference runs. It does NOT calculate accuracy because it does
not know the correct labels for these images. Use manually labelled test data for metrics.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vyndra.vision import predict_image, DEFAULT_MODEL

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_WORDS = ("collage", "chatgpt", "screenshot", "twotoone", "photo-collage", "4_pic")


def collect_images(source: Path, include_suspect: bool = False) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in EXTENSIONS else []
    if not source.is_dir():
        raise FileNotFoundError(f"Image source does not exist: {source}")
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not include_suspect:
        paths = [p for p in paths if not any(word in p.name.lower() for word in SKIP_WORDS)]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="YOLO images", help="An image file or folder; default: YOLO images")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to classifier weights")
    parser.add_argument("--output", default="results/tables/vision_smoke_test.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=20, help="Maximum images to test (default: 20)")
    parser.add_argument("--include-suspect", action="store_true", help="Include filenames suggesting collage/screenshots/AI images")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    model = Path(args.model)
    if not model.is_absolute():
        model = ROOT / model
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    images = collect_images(source, args.include_suspect)[:args.limit]
    if not images:
        print(f"No supported individual images found in: {source}")
        print("Supported extensions: jpg, jpeg, png, webp, bmp. Try --source uploads or pass one image path.")
        return 2
    if not model.is_file():
        print(f"Model file not found: {model}")
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    print(f"Model: {model}")
    print(f"Images selected: {len(images)}")
    for number, image in enumerate(images, start=1):
        try:
            result = predict_image(image, top_k=3, model_path=model)
            top = result["top_predictions"]
            row = {
                "image_path": str(image.relative_to(ROOT)) if image.is_relative_to(ROOT) else str(image),
                "top1_class": top[0]["class_name"],
                "top1_confidence": top[0]["confidence"],
                "top2_class": top[1]["class_name"] if len(top) > 1 else "",
                "top2_confidence": top[1]["confidence"] if len(top) > 1 else "",
                "top3_class": top[2]["class_name"] if len(top) > 2 else "",
                "top3_confidence": top[2]["confidence"] if len(top) > 2 else "",
                "status": "ok",
                "error": "",
            }
            print(f"[{number}/{len(images)}] {image.name} -> {row['top1_class']} ({row['top1_confidence']:.3f})")
        except Exception as exc:
            row = {"image_path": str(image), "top1_class": "", "top1_confidence": "", "top2_class": "", "top2_confidence": "", "top3_class": "", "top3_confidence": "", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            errors.append((image, exc))
            print(f"[{number}/{len(images)}] ERROR {image.name}: {type(exc).__name__}: {exc}")
        rows.append(row)

    fields = ["image_path", "top1_class", "top1_confidence", "top2_class", "top2_confidence", "top3_class", "top3_confidence", "status", "error"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(row["status"] == "ok" for row in rows)
    print("\nSmoke-test summary")
    print(f"Images attempted: {len(rows)}")
    print(f"Inference succeeded: {successes}")
    print(f"Inference errors: {len(errors)}")
    print(f"CSV saved: {output}")
    print("NOTE: This is an inference smoke test, not an accuracy evaluation. No ground-truth labels were supplied.")
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
