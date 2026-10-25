#!/usr/bin/env python3
"""Conservatively auto-approve safe image-folder mappings.

Approves normalized exact-name matches and a small explicit list of common
spelling/label variants. It skips fuzzy-only guesses. This is a convenience
step, not a guarantee that every image is correctly labelled.
"""
from pathlib import Path
import csv, re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "data/processed/image_class_map_review.csv"
OUT = ROOT / "data/processed/image_class_map.csv"


def norm(s):
    s = re.sub(r"[_-]+", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# Explicitly curated variants where the wording/spelling is a known variant.
# Keep this list conservative; add a pair only when the labels are equivalent
# enough for the project's evaluation.
SAFE_ALIASES = {
    ("aloo gobi", "aloo gobi masala"),
    ("chapati", "chappati"),
    ("ras malai", "rasmalai"),
    ("gulab jamun", "gulab jamoon"),
    ("biryani", "biriyani"),
    ("ras malai", "ras malai"),
    ("butter chicken", "butter chicken"),
    ("palak paneer", "palak paneer"),
    ("dal makhani", "dal makhani"),
    ("dum aloo", "dum aloo"),
    ("naan", "naan"),
    ("poha", "poha"),
    ("lassi", "lassi"),
}


def main():
    if not REVIEW.is_file():
        raise SystemExit(f"Missing {REVIEW}. Run: python scripts/map_dataset_classes.py")
    with REVIEW.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {REVIEW}")
    needed = {"dataset", "dataset_class", "best_pt_class", "similarity"}
    if not needed.issubset(rows[0]):
        raise SystemExit(f"Review CSV needs columns: {sorted(needed)}")

    out_rows = []
    reasons = Counter()
    for r in rows:
        source = r["dataset_class"].strip()
        target = r["best_pt_class"].strip()
        ns, nt = norm(source), norm(target)
        compact_equal = ns.replace(" ", "") == nt.replace(" ", "")
        alias_pair = (ns, nt) in SAFE_ALIASES
        reverse_alias_pair = (nt, ns) in SAFE_ALIASES
        if ns == nt or compact_equal:
            status, basis = "ok", "normalized_exact"
        elif alias_pair or reverse_alias_pair:
            status, basis = "ok", "curated_alias"
        else:
            status, basis = "skip", "fuzzy_suggestion_not_auto_approved"
        reasons[basis] += 1
        out_rows.append({
            "dataset": r.get("dataset", "indian_food_images"),
            "dataset_class": source,
            "best_pt_class": target,
            "status": status,
            "mapping_basis": basis,
            "similarity": r.get("similarity", ""),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "dataset_class", "best_pt_class", "status", "mapping_basis", "similarity"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    approved = [r for r in out_rows if r["status"] == "ok"]
    print(f"Wrote: {OUT}")
    print(f"Total dataset classes: {len(out_rows)}")
    print(f"Automatically approved: {len(approved)}")
    print(f"Skipped (uncertain fuzzy matches): {len(out_rows) - len(approved)}")
    print(f"Approval reasons: {dict(reasons)}")
    print("\nApproved mappings:")
    for r in approved:
        print(f"  {r['dataset_class']} -> {r['best_pt_class']} [{r['mapping_basis']}]")
    print("\nNote: auto-approved names are not proof that every image is correctly labelled or independent of training data.")
    if not approved:
        raise SystemExit("No safe mappings found; do not force fuzzy matches. Check the review CSV and class labels.")

if __name__ == "__main__":
    main()
