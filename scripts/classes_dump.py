#!/usr/bin/env python3
"""Export class names from the existing Ultralytics classification model."""

from pathlib import Path
import argparse
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "best.pt"),
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "classes.txt"),
        help="Path for the exported class names.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Verify the path before running this script."
        )

    model = YOLO(str(model_path))
    names = model.names

    if isinstance(names, dict):
        class_names = [str(names[key]) for key in sorted(names)]
    else:
        class_names = [str(name) for name in names]

    if not class_names:
        raise RuntimeError("The model returned no class names.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{name}\n" for name in class_names),
        encoding="utf-8",
    )

    print(f"Model: {model_path}")
    print(f"Number of class names: {len(class_names)}")
    print(f"Saved to: {output_path}")
    print("\nFirst 20 class names:")
    for index, name in enumerate(class_names[:20]):
        print(f"{index}: {name}")


if __name__ == "__main__":
    main()