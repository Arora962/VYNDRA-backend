#!/usr/bin/env python3
"""Compare 100 g default portions with Anuvaad serving estimates on a filled template."""
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vyndra.portions import serving_info
TEMPLATE = ROOT / "data" / "processed" / "reference_meals_template.csv"
OUTPUT = ROOT / "results" / "tables" / "portion_evaluation.csv"

def main():
    frame = pd.read_csv(TEMPLATE)
    required = {"meal_id", "food_name", "reference_grams"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"Template must contain columns: {sorted(required)}")
    frame["reference_grams"] = pd.to_numeric(frame["reference_grams"], errors="coerce")
    frame = frame[frame["reference_grams"].between(1, 5000)].copy()
    frame = frame[~frame["meal_id"].astype(str).str.startswith("example_replace_me")]
    if frame.empty:
        raise SystemExit("No reference rows yet. Replace the example row with real weighed meal data, then rerun.")
    output = []
    for row in frame.to_dict("records"):
        info = serving_info(str(row["food_name"]))
        reference = float(row["reference_grams"])
        output.append({"meal_id": row["meal_id"], "food_name": row["food_name"], "reference_grams": reference,
                       "default_100g_error": abs(100.0 - reference),
                       "serving_estimate_grams": info["grams_per_serving"],
                       "serving_estimate_error": abs(float(info["grams_per_serving"]) - reference),
                       "default_portion": info["default_portion"], "servings_unit": info["servings_unit"]})
    result = pd.DataFrame(output)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    reference = result["reference_grams"].to_numpy()
    print(f"Reference meals evaluated: {len(result)}")
    for label, predicted in [("100 g default", [100.0] * len(result)), ("Anuvaad serving estimate", result["serving_estimate_grams"].tolist())]:
        mae = mean_absolute_error(reference, predicted)
        mape = (abs((reference - predicted) / reference).mean() * 100) if (reference != 0).all() else float("nan")
        print(f"{label}: MAE={mae:.2f} g, MAPE={mape:.2f}%")
    print(f"Saved row-level results: {OUTPUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
