#!/usr/bin/env python3
"""Prepare a reproducible image manifest from local dataset roots; never includes hidden macOS files."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import pandas as pd
IMAGE_EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--kaggle-root',type=Path); p.add_argument('--khana-root',type=Path); p.add_argument('--food101-root',type=Path); p.add_argument('--own-photos-root',type=Path,default=ROOT/'data/test_images_own'); p.add_argument('--seed',type=int,default=42); p.add_argument('--max-per-class-dataset',type=int,default=30); a=p.parse_args()
 if a.max_per_class_dataset<1: p.error('--max-per-class-dataset must be >=1')
 map_path=ROOT/'data/processed/image_class_map.csv'
 if not map_path.exists(): raise SystemExit('Missing data/processed/image_class_map.csv')
 maps=pd.read_csv(map_path).fillna(''); maps=maps[maps.status.astype(str).str.lower().eq('ok')].copy()
 # Correct the known external dataset label to the actual classifier class when present.
 maps.loc[maps.dataset_class.astype(str).eq('aloo_gobi'),'best_pt_class']='allo gobi'
 roots={'indian_food_images':a.kaggle_root,'khana':a.khana_root,'food101':a.food101_root}; rows=[]; rng=random.Random(a.seed)
 for dataset,root in roots.items():
  if root is None: continue
  root=root.expanduser().resolve()
  if not root.exists(): print(f'[WARN] {dataset} root does not exist: {root}'); continue
  dmap=maps[maps.dataset.astype(str).eq(dataset)]
  if dataset!='indian_food_images':
   csv=ROOT/'data/processed'/f'{dataset}_class_map.csv'
   if not csv.exists():
    template=root.parent/f'{dataset}_class_map_TEMPLATE.csv'
    classes=sorted(x.name for x in root.iterdir() if x.is_dir() and not x.name.startswith('._'))
    pd.DataFrame([{'dataset':dataset,'dataset_class':c,'best_pt_class':'','accepted_model_classes':'','match_type':'','status':'needs_review'} for c in classes]).to_csv(template,index=False)
    print(f'[WARN] No reviewed mapping for {dataset}. Created template: {template}. Fill best_pt_class and match_type exact/family, then rerun.')
    continue
   dmap=pd.read_csv(csv).fillna(''); dmap=dmap[dmap.status.astype(str).str.lower().eq('ok')]
  for m in dmap.itertuples():
   cls=str(m.dataset_class); model_class=str(m.best_pt_class)
   accepted=str(getattr(m,'accepted_model_classes','') or model_class).strip()
   if not accepted: accepted=model_class
   match_type=str(getattr(m,'match_type','exact')).lower()
   if dataset=='indian_food_images':
    # image_class_map is a reviewed dataset-label map; nutrition mapping approval is checked separately.
    model_mapping=pd.read_csv(ROOT/'data/processed/class_to_food.csv').fillna('')
    approved=set(model_mapping.loc[model_mapping.match_type.isin(['exact','family']),'class_name'].astype(str))
    if model_class not in approved: continue
   elif match_type not in {'exact','family'}: continue
   class_dir=root/cls
   if not class_dir.is_dir():
    # Kaggle directories commonly use underscores; do not guess unrelated directories.
    alt=root/cls.replace(' ','_')
    if alt.is_dir(): class_dir=alt
    else: continue
   files=[x for x in class_dir.rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXTS and not x.name.startswith('._') and not any(p.name.startswith('._') for p in x.parents)]
   files=sorted(files); rng.shuffle(files); files=files[:a.max_per_class_dataset]
   for image in files:
    # Store paths relative to the dataset root; the evaluator receives the root separately.
    rel=image.relative_to(root)
    rows.append({'dataset':dataset,'dataset_class':cls,'model_class':model_class,'accepted_model_classes':accepted,'match_type':match_type,'relative_path':rel.as_posix(),'seed':a.seed})
 # Own photos are evaluated separately and only when the folder name is an approved model class.
 own=a.own_photos_root.expanduser().resolve() if a.own_photos_root else None
 if own and own.is_dir():
  approved_map=pd.read_csv(ROOT/'data/processed/class_to_food.csv').fillna('')
  eligible=set(approved_map.loc[approved_map.match_type.isin(['exact','family']),'class_name'].astype(str))
  for class_dir in sorted(x for x in own.iterdir() if x.is_dir() and not x.name.startswith('._')):
   if class_dir.name not in eligible: continue
   files=[x for x in class_dir.rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXTS and not x.name.startswith('._') and not any(p.name.startswith('._') for p in x.parents)]
   rng.shuffle(files)
   for image in files:
    rows.append({'dataset':'own_photos','dataset_class':class_dir.name,'model_class':class_dir.name,'accepted_model_classes':class_dir.name,'match_type':'exact_or_family_approved','relative_path':image.relative_to(own).as_posix(),'seed':a.seed})
 out=ROOT/'data/processed/vision_test_manifest.csv'; out.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows,columns=['dataset','dataset_class','model_class','accepted_model_classes','match_type','relative_path','seed']).to_csv(out,index=False)
 print(f'Wrote {len(rows)} manifest rows to {out}')
 if not rows: print('No eligible images found. Configure dataset roots and reviewed mapping files before vision evaluation.')
if __name__=='__main__': main()
