#!/usr/bin/env python3
"""Copy a bounded, reproducible sample from reviewed Kaggle class mappings."""
from pathlib import Path
import csv, random, shutil, sys
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/raw/indian_food_images/Indian Food Images'
MAPPING=ROOT/'data/processed/image_class_map.csv'
OUT=ROOT/'data/test_images'
MANIFEST=ROOT/'data/processed/test_manifest.csv'
EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}
SEED=42

def main():
    if not MAPPING.is_file(): raise SystemExit(f'Missing reviewed mapping: {MAPPING}')
    if not SOURCE.is_dir(): raise SystemExit(f'Dataset folder not found: {SOURCE}')
    with MAPPING.open(newline='',encoding='utf-8-sig') as f: maps=list(csv.DictReader(f))
    required={'dataset_class','best_pt_class','status'}
    if not maps or not required.issubset(maps[0]): raise SystemExit(f'Mapping must include columns: {sorted(required)}')
    approved={r['dataset_class'].strip():r['best_pt_class'].strip() for r in maps if r.get('status','').strip().lower()=='ok' and r.get('dataset_class','').strip() and r.get('best_pt_class','').strip()}
    if not approved: raise SystemExit('No approved rows. Edit image_class_map.csv and set status=ok only for manually verified mappings.')
    rng=random.Random(SEED); rows=[]; max_per_class=30
    for source_class, target_class in sorted(approved.items()):
        folder=SOURCE/source_class
        if not folder.is_dir(): print(f'SKIP missing source folder: {folder}'); continue
        images=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
        rng.shuffle(images); chosen=images[:max_per_class]
        if not chosen: print(f'SKIP no supported images: {source_class}'); continue
        safe_target=target_class.replace('/','_').replace('\\','_')
        dest_dir=OUT/safe_target; dest_dir.mkdir(parents=True,exist_ok=True)
        for k,src in enumerate(chosen):
            # Prefix names to avoid collisions where several source classes map to one model class.
            dest=dest_dir/f'{source_class}__{k:03d}__{src.name}'
            shutil.copy2(src,dest)
            rows.append({'dataset':'indian_food_images','dataset_class':source_class,'true_class':target_class,'source_path':str(src),'image_path':str(dest.relative_to(ROOT))})
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    with MANIFEST.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','dataset_class','true_class','source_path','image_path']);w.writeheader();w.writerows(rows)
    print(f'Approved source classes: {len(approved)}')
    print(f'Images copied: {len(rows)} (maximum 30 per approved source class; seed={SEED})')
    print(f'Test manifest: {MANIFEST}')
    print('Caution: independent-test status is unknown because overlap with the classifier training data has not been verified.')
if __name__=='__main__': main()
