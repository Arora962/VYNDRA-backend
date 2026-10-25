#!/usr/bin/env python3
"""Compare approved exact/family mappings with character-TF-IDF fuzzy suggestions."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/tables/part4'; PROC=ROOT/'data/processed'
def norm(x):
 import re
 return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9]+',' ',str(x).lower())).strip()
def main():
 approved_path=PROC/'class_to_food.csv'; review_path=PROC/'class_to_food_review.csv'
 if not approved_path.exists() or not review_path.exists(): raise SystemExit('Missing approved or review mapping CSV.')
 approved=pd.read_csv(approved_path).fillna(''); review=pd.read_csv(review_path).fillna('')
 foods=pd.read_csv(ROOT/'data/raw/Db.csv').fillna(''); names=foods.food_name.astype(str).tolist(); norm_names=[norm(x) for x in names]
 vec=TfidfVectorizer(analyzer='char',ngram_range=(2,4),sublinear_tf=True); matrix=vec.fit_transform(norm_names)
 rows=[]
 eligible=[r for r in approved.itertuples() if str(r.match_type) in {'exact','family'} and str(r.food_name).strip()]
 queries=vec.transform([norm(r.class_name) for r in eligible])
 all_scores=cosine_similarity(queries,matrix)
 for row_index,r in enumerate(eligible):
  cls=str(r.class_name); truth=str(r.food_name); kind=str(r.match_type)
  exact=next((names[i] for i,x in enumerate(norm_names) if x==norm(truth)), '')
  scores=all_scores[row_index]; fuzzy=names[int(scores.argmax())]
  rows.append({'class_name':cls,'reviewed_food_name':truth,'match_type':kind,'exact_normalized_candidate':exact,'fuzzy_candidate':fuzzy,'fuzzy_similarity':float(scores.max()),'exact_agrees_with_review':norm(exact)==norm(truth),'fuzzy_agrees_with_review':norm(fuzzy)==norm(truth)})
 out=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); out.to_csv(OUT/'matching_method_comparison.csv',index=False)
 summary={'reviewed_mapped_classes_evaluated':len(out),'exact_normalized_agreement':float(out.exact_agrees_with_review.mean()) if len(out) else None,'fuzzy_top1_agreement':float(out.fuzzy_agrees_with_review.mean()) if len(out) else None,'warning':'Agreement with the reviewed mapping is not independent ground truth and does not establish semantic correctness.'}
 import json; (OUT/'matching_method_comparison.json').write_text(json.dumps(summary,indent=2)); print(f'Wrote comparison for {len(out)} reviewed mapped classes; this is agreement with the reviewed mapping, not independent accuracy.')
if __name__=='__main__': main()
