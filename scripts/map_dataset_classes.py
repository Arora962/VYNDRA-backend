#!/usr/bin/env python3
"""Suggest mappings from Kaggle Indian Food Images folders to VYNDRA model classes.
Suggestions are NOT ground truth. Review every row before approving it.
"""
from pathlib import Path
import csv, re, sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET = ROOT / 'data/raw/indian_food_images/Indian Food Images'
CLASSES = ROOT / 'data/processed/classes.txt'
OUT = ROOT / 'data/processed/image_class_map_review.csv'

def norm(s):
    s = re.sub(r'[_-]+', ' ', str(s).lower())
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def main():
    if not CLASSES.is_file(): raise SystemExit(f'Missing {CLASSES}; run scripts/classes_dump.py first.')
    if not DATASET.is_dir(): raise SystemExit(f'Dataset folder not found: {DATASET}\nEdit DATASET in this script to match your actual folder.')
    model_classes = [x.strip() for x in CLASSES.read_text(encoding='utf-8').splitlines() if x.strip()]
    source_classes = sorted(p.name for p in DATASET.iterdir() if p.is_dir())
    if not source_classes: raise SystemExit(f'No class subfolders found under {DATASET}')
    all_names = [norm(x) for x in model_classes + source_classes]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), lowercase=False)
    matrix = vec.fit_transform(all_names)
    m = matrix[:len(model_classes)]
    d = matrix[len(model_classes):]
    sims = cosine_similarity(d, m)
    rows=[]
    for i, src in enumerate(source_classes):
        j = int(sims[i].argmax()); score = float(sims[i,j]); target = model_classes[j]
        status = 'exact_text_review_required' if norm(src) == norm(target) else ('needs_review' if score >= 0.22 else 'no_match')
        rows.append({'dataset':'indian_food_images','dataset_class':src,'best_pt_class':target,'similarity':round(score,4),'status':status})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['dataset','dataset_class','best_pt_class','similarity','status']); w.writeheader(); w.writerows(rows)
    print(f'Dataset classes found: {len(source_classes)}')
    print(f'Model classes available: {len(model_classes)}')
    print(f'Suggestions saved: {OUT}')
    from collections import Counter
    print('Status counts:', dict(Counter(r['status'] for r in rows)))
    print('\nReview all suggestions. Copy approved rows into data/processed/image_class_map.csv with columns dataset,dataset_class,best_pt_class,status; set status=ok ONLY when semantically correct. Use status=skip for uncertain/incorrect rows.')
    print('First 20 suggestions:')
    for r in rows[:20]: print(f"{r['dataset_class']} -> {r['best_pt_class']} ({r['similarity']:.3f}) [{r['status']}]")
if __name__ == '__main__': main()
