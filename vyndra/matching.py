"""Food-name matching for VYNDRA; suggestions require review before production use."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .nutrition import get_food_table, normalize_food_name

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REVIEW_CSV = PROCESSED / "class_to_food_review.csv"
APPROVED_CSV = PROCESSED / "class_to_food.csv"
ALIASES = {
    "idly": "idli", "idlis": "idli", "dosa": "dosa", "dosai": "dosa",
    "chappati": "chapati", "chapathi": "chapati", "roti": "chapati",
    "panner": "paneer", "cocunut": "coconut", "biriyani": "biryani",
    "biriani": "biryani", "curd rice": "curd rice", "aloo gobi": "aloo gobi",
}


def normalize_for_matching(value: object) -> str:
    name = normalize_food_name(value)
    return ALIASES.get(name, name)


def load_class_names() -> list[str]:
    path = PROCESSED / "classes.txt"
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Fallback to model weights only if the class list has not been exported yet.
    model_path = ROOT / "models" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError("No classes.txt or models/best.pt found. Run scripts/classes_dump.py first.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Could not import ultralytics. Activate the environment with ultralytics installed, then run scripts/classes_dump.py.") from exc
    names = YOLO(str(model_path)).names
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    return [str(x) for x in names]


def build_review_file() -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    foods = get_food_table()
    food_names = foods["food_name"].astype(str).tolist()
    normalized_foods = [normalize_for_matching(x) for x in food_names]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True, sublinear_tf=True)
    food_matrix = vectorizer.fit_transform(normalized_foods)
    rows = []
    for class_name in load_class_names():
        query = normalize_for_matching(class_name)
        exact_indices = [i for i, x in enumerate(normalized_foods) if x == query]
        if exact_indices:
            idx = exact_indices[0]
            rows.append({"class_name": class_name, "suggested_food": food_names[idx], "similarity": 1.0, "status": "exact_text_review_required"})
            continue
        scores = cosine_similarity(vectorizer.transform([query]), food_matrix).ravel()
        idx = int(scores.argmax())
        score = float(scores[idx])
        status = "needs_review" if score >= 0.25 else "no_match"
        rows.append({"class_name": class_name, "suggested_food": food_names[idx] if status != "no_match" else "", "similarity": round(score, 4), "status": status})
    out = pd.DataFrame(rows, columns=["class_name", "suggested_food", "similarity", "status"])
    out.to_csv(REVIEW_CSV, index=False)
    return out


def load_approved_mapping() -> dict[str, dict]:
    """Read only the manually reviewed class_to_food.csv; suggestions are never auto-approved."""
    if not APPROVED_CSV.exists():
        return {}
    df = pd.read_csv(APPROVED_CSV).fillna("")
    required = {"class_name", "food_name", "match_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"{APPROVED_CSV} must contain columns: class_name, food_name, match_type")
    allowed = {"exact", "family", "none"}
    mapping = {}
    for _, row in df.iterrows():
        cls = str(row["class_name"]).strip()
        food = str(row["food_name"]).strip()
        match_type = str(row["match_type"]).strip().lower()
        if not cls:
            continue
        if match_type not in allowed:
            raise ValueError(f"Invalid match_type {match_type!r} for class {cls!r}; use exact, family, or none")
        mapping[cls] = {"food_name": food if match_type != "none" else "", "match_type": match_type, "reviewed": True}
    return mapping


def match_class_to_food(class_name: str) -> dict:
    approved = load_approved_mapping().get(class_name)
    if approved:
        if approved["match_type"] == "none":
            return {"class_name": class_name, "found": False, "match_type": "none", "reviewed": True, "message": "This class has no approved nutrition-table equivalent."}
        from .nutrition import find_food
        result = find_food(approved["food_name"])
        result.update({"class_name": class_name, "match_type": approved["match_type"], "reviewed": True})
        if not result.get("found"):
            result["message"] = "Approved mapping points to a food name not found in the nutrition table; please correct class_to_food.csv."
        return result
    if REVIEW_CSV.exists():
        df = pd.read_csv(REVIEW_CSV).fillna("")
        hit = df[df["class_name"].astype(str).eq(class_name)]
        if not hit.empty:
            row = hit.iloc[0]
            return {"class_name": class_name, "found": False, "reviewed": False, "suggested_food": str(row["suggested_food"]), "similarity": float(row["similarity"]), "status": str(row["status"]), "message": "This is an unreviewed suggestion; nutrition values are not returned."}
    return {"class_name": class_name, "found": False, "reviewed": False, "message": "No approved food mapping or generated suggestion found."}
