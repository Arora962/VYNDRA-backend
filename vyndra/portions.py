"""Serving-size estimates derived from the supplied Anuvaad nutrition workbook."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pandas as pd
from .nutrition import normalize_food_name

ROOT = Path(__file__).resolve().parents[1]
SERVING_FILE = ROOT / "data" / "raw" / "Anuvaad_INDB_2024.11.xlsx"
DEFAULT_PORTION_G = 100.0

@lru_cache(maxsize=1)
def _serving_map() -> dict[str, dict]:
    if not SERVING_FILE.exists():
        return {}
    frame = pd.read_excel(SERVING_FILE)
    needed = {"food_name", "energy_kcal", "unit_serving_energy_kcal", "servings_unit"}
    if not needed.issubset(frame.columns):
        return {}
    frame["energy_kcal"] = pd.to_numeric(frame["energy_kcal"], errors="coerce")
    frame["unit_serving_energy_kcal"] = pd.to_numeric(frame["unit_serving_energy_kcal"], errors="coerce")
    frame["servings_unit"] = frame["servings_unit"].fillna("").astype(str).str.strip()
    frame["_key"] = frame["food_name"].map(normalize_food_name)
    frame["_grams"] = frame["unit_serving_energy_kcal"] / frame["energy_kcal"] * 100.0
    valid = frame[
        frame["servings_unit"].ne("") & frame["_grams"].between(5, 1000, inclusive="both")
        & frame["_grams"].notna() & frame["energy_kcal"].gt(0)
    ]
    result = {}
    for _, row in valid.drop_duplicates("_key").iterrows():
        result[str(row["_key"])] = {"grams_per_serving": round(float(row["_grams"]), 2), "servings_unit": row["servings_unit"], "default_portion": False}
    return result


def serving_info(food_name: str) -> dict:
    """Return a cleaned serving estimate, or a documented 100 g default."""
    found = _serving_map().get(normalize_food_name(food_name))
    if found:
        return dict(found)
    return {"grams_per_serving": DEFAULT_PORTION_G, "servings_unit": "100 g default", "default_portion": True}


def portion_grams(food_name: str, grams=None, servings=None) -> dict:
    """Resolve user-specified grams or a serving multiplier into portion grams."""
    info = serving_info(food_name)
    if grams is not None:
        value = float(grams)
        if not 1 <= value <= 5000:
            raise ValueError("grams must be between 1 and 5000")
        info.update({"quantity_grams": round(value, 2), "portion_source": "user_grams"})
        return info
    if servings is not None:
        value = float(servings)
        if not 0 < value <= 20:
            raise ValueError("servings must be greater than 0 and at most 20")
        info.update({"quantity_grams": round(info["grams_per_serving"] * value, 2), "portion_source": "serving_estimate"})
        return info
    info.update({"quantity_grams": info["grams_per_serving"], "portion_source": "serving_estimate"})
    return info
