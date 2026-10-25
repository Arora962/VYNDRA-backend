"""Complete data/processed/class_to_food.csv so that every one of the 220 model classes has a row.

Classes that are not listed get match_type=none (no equivalent in the nutrition table).
Spacing and capital letters are ignored when matching, and the class names are rewritten
exactly as they appear in classes.txt.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "data/processed/class_to_food.csv"
classes = [l.strip() for l in (ROOT / "data/processed/classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]


def key(text):
    return " ".join(str(text).split()).lower()


df = pd.read_csv(csv_path).fillna("")
rows = {key(r.class_name): (str(r.food_name).strip(), str(r.match_type).strip().lower()) for r in df.itertuples()}
unknown = [k for k in rows if k not in {key(c) for c in classes}]
if unknown:
    print("WARNING - these rows do not match any model class (typo?):", unknown)

out = []
for c in classes:
    food, mtype = rows.get(key(c), ("", "none"))
    out.append({"class_name": c, "food_name": food if mtype != "none" else "", "match_type": mtype})
out = pd.DataFrame(out)
out.to_csv(csv_path, index=False)
print(f"classes: {len(out)} | with nutrition: {(out.match_type != 'none').sum()} | none: {(out.match_type == 'none').sum()}")