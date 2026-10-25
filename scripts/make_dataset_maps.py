"""Write reviewed class maps for Khana and Food-101 -> best.pt classes.

Class names are taken from data/processed/classes.txt, so spacing is always exact.
Only classes that are approved (exact/family) in class_to_food.csv are used.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
key = lambda s: " ".join(str(s).split()).lower()
classes = [l.strip("\n") for l in (ROOT / "data/processed/classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
canon = {key(c): c for c in classes}
approved = pd.read_csv(ROOT / "data/processed/class_to_food.csv").fillna("")
ok = {r.class_name for r in approved.itertuples() if r.match_type in ("exact", "family")}

KHANA = [
    ("aloo gobi", "allo gobi|aloo gobi masala", "family"),
    ("biryani", "biryani", "exact"),
    ("chana masala", "chole chickpea curry", "family"),
    ("chapati", "chappati|roti|phulka", "exact"),
    ("fish curry", "fish curry", "exact"),
    ("garlic naan", "garlice naan|cheesy garlic naan", "exact"),
    ("gulab jamun", "gulab jamoon", "exact"),
    ("idli", "idly|masala idly", "exact"),
    ("masala dosa", "masala dosa", "exact"),
    ("medu vada", "medu vada", "exact"),
    ("palak paneer", "palak panner|saag panner", "exact"),
    ("seekh kebab", "mutton seekh kebab", "family"),
    ("uttapam", "uthappam", "exact"),
    ("kulfi", "kulfi", "exact"),
    ("paneer masala", "panner butter masala", "family"),
]
FOOD101 = [
    ("chicken_curry", "chicken curry", "exact"),
    ("ice_cream", "ice icream", "exact"),
    ("lasagna", "lasagna", "exact"),
    ("spaghetti_bolognese", "spaghetti", "exact"),
    ("macaroni_and_cheese", "mac and cheese", "exact"),
    ("ramen", "ramen noodles", "family"),
    ("fried_rice", "vegetable fried rice|chicken fried rice|egg fried rice", "family"),
    ("chocolate_cake", "cake", "family"),
    ("carrot_cake", "cake", "family"),
    ("red_velvet_cake", "cake", "family"),
]


def build(dataset, rows):
    out = []
    for folder, models, match_type in rows:
        accepted = [canon[key(m)] for m in models.split("|")]
        assert all(a in ok for a in accepted), (folder, accepted)
        out.append({"dataset": dataset, "dataset_class": folder, "best_pt_class": accepted[0],
                    "accepted_model_classes": "|".join(accepted), "match_type": match_type, "status": "ok"})
    return pd.DataFrame(out)


build("khana", KHANA).to_csv(ROOT / "data/processed/khana_class_map.csv", index=False)
build("food101", FOOD101).to_csv(ROOT / "data/processed/food101_class_map.csv", index=False)
print("Wrote khana_class_map.csv (15 classes) and food101_class_map.csv (10 classes)")