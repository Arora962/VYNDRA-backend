"""Nutrition lookup for VYNDRA. Nutrition values are estimates based on the supplied food table."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NUTRITION_CSV = RAW / "Db.csv"
DIET_CSV = RAW / "Selective_Diet_Mapping.csv"
SERVING_XLSX = RAW / "Anuvaad_INDB_2024.11.xlsx"
MACROS = ["energy_kcal", "carb_g", "protein_g", "fat_g", "fibre_g"]


def normalize_food_name(value: object) -> str:
    """Normalize a food name for lookup; this is not a semantic equivalence test."""
    text = str(value or "").lower().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_table() -> pd.DataFrame:
    if not NUTRITION_CSV.exists():
        raise FileNotFoundError(f"Nutrition file not found: {NUTRITION_CSV}")
    df = pd.read_csv(NUTRITION_CSV)
    required = {"food_name", *MACROS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Db.csv is missing required columns: {sorted(missing)}")
    df["food_name"] = df["food_name"].fillna("").astype(str).str.strip()
    for col in MACROS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["food_name"].ne("")].dropna(subset=MACROS).copy()
    df["_normalized_name"] = df["food_name"].map(normalize_food_name)
    if DIET_CSV.exists():
        diet = pd.read_csv(DIET_CSV)
        if "food_name" in diet.columns:
            keep = [c for c in ["food_name", "diet_tags", "mood_tags", "allergens", "best_time_to_eat", "description"] if c in diet.columns]
            diet = diet[keep].copy()
            diet["_normalized_name"] = diet["food_name"].map(normalize_food_name)
            diet = diet.drop(columns=["food_name"]).drop_duplicates("_normalized_name")
            df = df.merge(diet, on="_normalized_name", how="left")
    return df.reset_index(drop=True)


_FOODS = None

def get_food_table() -> pd.DataFrame:
    """Load the nutrition table once per Python process."""
    global _FOODS
    if _FOODS is None:
        _FOODS = _load_table()
    return _FOODS.copy()


def find_food(food_name: str) -> dict:
    """Exact normalized name lookup only; returns a clear error for unknown foods."""
    query = normalize_food_name(food_name)
    if not query:
        raise ValueError("food_name must not be empty")
    df = get_food_table()
    hits = df[df["_normalized_name"].eq(query)]
    if hits.empty:
        return {"found": False, "food_name_requested": food_name, "message": "No exact nutrition-table match. Review the food name; no nutrition values were guessed."}
    row = hits.iloc[0]
    result = {"found": True, "food_name": row["food_name"], "basis": "per 100 g", "match_type": "exact_normalized_name"}
    for col in MACROS:
        result[col] = round(float(row[col]), 2)
    for col in ["diet_tags", "mood_tags", "allergens", "best_time_to_eat", "description"]:
        if col in df.columns and pd.notna(row.get(col)):
            result[col] = str(row[col])
    return result


def calculate_nutrition(food_name: str, grams: float) -> dict:
    """Scale the table's per-100g nutrition values to a user-specified gram amount."""
    try:
        grams = float(grams)
    except (TypeError, ValueError):
        raise ValueError("grams must be a number")
    if not 0 < grams <= 5000:
        raise ValueError("grams must be greater than 0 and no more than 5000")
    food = find_food(food_name)
    if not food["found"]:
        return food
    factor = grams / 100.0
    result = dict(food)
    result["portion_g"] = round(grams, 2)
    result["basis"] = "scaled from supplied table's per-100g values"
    for col in MACROS:
        result[col] = round(food[col] * factor, 2)
    result["nutrition_note"] = "Estimated values; actual nutrition varies by recipe and portion. Food-name match is exact after text normalization."
    return result
