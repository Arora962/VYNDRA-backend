from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vyndra.matching import build_review_file

if __name__ == "__main__":
    df = build_review_file()
    print(f"Saved review suggestions: {ROOT / 'data/processed/class_to_food_review.csv'}")
    print(f"Model classes processed: {len(df)}")
    print("Status counts:")
    print(df["status"].value_counts(dropna=False).to_string())
    print("\nFirst 15 suggestions (review these; they are not approved mappings):")
    print(df.head(15).to_string(index=False))
