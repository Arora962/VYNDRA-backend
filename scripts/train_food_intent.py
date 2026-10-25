#!/usr/bin/env python3
"""Train and save the chat food-intent model once for API startup."""
from pathlib import Path
import json
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vyndra.chat_intelligence import train_intent_model
MODEL = ROOT / "models" / "food_intent.joblib"
REPORT = ROOT / "results" / "tables" / "food_intent_training_metrics.json"

def main():
    model, metrics = train_intent_model()
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    metrics.pop("test_rows", None)
    REPORT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL.relative_to(ROOT)}")
    print(f"Saved metrics: {REPORT.relative_to(ROOT)}")
    print("The API will load this model once per process.")

if __name__ == "__main__":
    main()
