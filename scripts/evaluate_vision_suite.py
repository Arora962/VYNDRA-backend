#!/usr/bin/env python3
"""Evaluate YOLO top-1/top-3 and classical HSV histogram baselines on a reviewed manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
OUT=ROOT/'results/tables/part7'; FIG=ROOT/'results/figures/part7'

def resolve_image(dataset, value, roots):
 p=Path(str(value))
 root=roots.get(str(dataset))
 if root is not None:
  candidate=root/p
  return candidate if candidate.is_file() else None
 candidate=ROOT/p
 return candidate if candidate.is_file() else None
def hsv_feature(path):
 image=cv2.imread(str(path))
 if image is None: return None
 hsv=cv2.cvtColor(cv2.resize(image,(128,128)),cv2.COLOR_BGR2HSV)
 hist=cv2.calcHist([hsv],[0,1,2],None,[8,4,4],[0,180,0,256,0,256]); cv2.normalize(hist,hist)
 return hist.flatten()
def metrics_row(name,y_true,y_pred):
 p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
 return {'method':name,'top1_accuracy':accuracy_score(y_true,y_pred),'macro_precision_supported':p,'macro_recall_supported':r,'macro_f1_supported':f}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,default=ROOT/'data/processed/vision_test_manifest.csv'); ap.add_argument('--kaggle-root',type=Path); ap.add_argument('--khana-root',type=Path); ap.add_argument('--food101-root',type=Path); ap.add_argument('--own-photos-root',type=Path,default=ROOT/'data/test_images_own'); ap.add_argument('--baseline-root',type=Path); ap.add_argument('--baseline-map',type=Path,default=ROOT/'data/processed/image_class_map.csv'); ap.add_argument('--baseline-classes',type=int,default=20); ap.add_argument('--baseline-images-per-class',type=int,default=100); args=ap.parse_args()
 if not args.manifest.exists(): raise SystemExit('Missing manifest. Run scripts/prepare_vision_manifest.py with local dataset roots.')
 df=pd.read_csv(args.manifest).fillna('')
 roots={'indian_food_images':args.kaggle_root.expanduser().resolve() if args.kaggle_root else None,'khana':args.khana_root.expanduser().resolve() if args.khana_root else None,'food101':args.food101_root.expanduser().resolve() if args.food101_root else None,'own_photos':args.own_photos_root.expanduser().resolve() if args.own_photos_root else None}
 df['resolved_path']=[resolve_image(row.dataset,row.relative_path,roots) for row in df.itertuples()]; df=df[df.resolved_path.notna()].copy()
 if df.empty: raise SystemExit('No manifest image paths resolve. Keep images under the repo or edit the manifest to use relative paths from the project root.')
 OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
 # Classical baseline prefers the full baseline root (up to 100 images per reviewed class).
 # If that root is absent, it falls back to eligible manifest rows and reports the smaller sample honestly.
 if args.baseline_root:
  b_root=args.baseline_root.expanduser().resolve()
  if not args.baseline_map.exists(): raise SystemExit(f'Missing baseline map: {args.baseline_map}')
  bmap=pd.read_csv(args.baseline_map).fillna(''); bmap=bmap[(bmap.status.astype(str).str.lower()=='ok')]
  approved_map=pd.read_csv(ROOT/'data/processed/class_to_food.csv').fillna(''); approved=set(approved_map.loc[approved_map.match_type.isin(['exact','family']),'class_name'].astype(str))
  base_rows=[]
  for item in bmap.itertuples():
   if str(item.best_pt_class) not in approved: continue
   class_dir=b_root/str(item.dataset_class)
   if not class_dir.is_dir(): class_dir=b_root/str(item.dataset_class).replace(' ','_')
   if not class_dir.is_dir(): continue
   images=sorted(x for x in class_dir.rglob('*') if x.is_file() and (x.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'} or x.suffix=='') and not x.name.startswith('._') and not any(q.name.startswith('._') for q in x.parents))[:args.baseline_images_per_class]
   for image in images: base_rows.append({'dataset_class':str(item.dataset_class),'model_class':str(item.best_pt_class),'dataset':'indian_food_images','resolved_path':image,'relative_path':image.relative_to(b_root).as_posix()})
  base=pd.DataFrame(base_rows)
 else:
  base=df.sort_values('relative_path').groupby('dataset_class',group_keys=False).head(args.baseline_images_per_class)
 feature_rows=[]
 for row in base.itertuples():
  feat=hsv_feature(row.resolved_path)
  if feat is not None: feature_rows.append({'dataset_class':row.dataset_class,'model_class':row.model_class,'dataset':row.dataset,'path':row.relative_path,'feature':feat})
 class_counts=pd.Series([r['dataset_class'] for r in feature_rows]).value_counts() if feature_rows else pd.Series(dtype=int)
 if len(class_counts)<args.baseline_classes or (len(class_counts) and int(class_counts.min())<args.baseline_images_per_class):
  print(f'[WARNING] Classical baseline has {len(class_counts)} classes and {len(feature_rows)} usable images; target is {args.baseline_classes} classes x {args.baseline_images_per_class} images. Do not claim the target was met.')
 baseline_results=[]
 if feature_rows:
  X=np.vstack([r['feature'] for r in feature_rows]); y=np.array([r['model_class'] for r in feature_rows]); indices=np.arange(len(y))
  if len(set(y))>=2 and len(y)>=6:
   train_idx,test_idx=train_test_split(indices,test_size=.25,random_state=42,stratify=y if min(pd.Series(y).value_counts())>=2 else None)
   for name,model in [('HSV_GaussianNB',GaussianNB()),('HSV_DecisionTree',DecisionTreeClassifier(random_state=42,max_depth=30)),('HSV_kNN',KNeighborsClassifier(n_neighbors=min(5,len(train_idx))))]:
    model.fit(X[train_idx],y[train_idx]); pred=model.predict(X[test_idx]); baseline_results.append(metrics_row(name,y[test_idx],pred))
    if name=='HSV_kNN':
     labels=sorted(set(y)); cm=confusion_matrix(y[test_idx],pred,labels=labels); fig,ax=plt.subplots(figsize=(9,7)); im=ax.imshow(cm); ax.set_title('HSV k-NN baseline confusion matrix'); ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=6); ax.set_yticks(range(len(labels)),labels,fontsize=6); fig.colorbar(im,ax=ax,label='Image count'); fig.tight_layout(); fig.savefig(FIG/'vision_baseline_confusion.png',dpi=160); plt.close(fig)
 # YOLO predictions on all eligible manifest rows. Model import/load is deferred until here.
 try: from vyndra.vision import predict_image
 except Exception as e: raise SystemExit(f'Could not import VYNDRA vision module: {e}')
 preds=[]
 for row in df.itertuples():
  try:
   result=predict_image(row.resolved_path,top_k=3); top=result['top_predictions']; expected=str(row.model_class); accepted=str(getattr(row,'accepted_model_classes',expected) or expected).split('|'); accepted=[x.strip() for x in accepted if x.strip()] or [expected]; names=[x['class_name'] for x in top]; conf=float(top[0]['confidence']) if top else 0.0
   preds.append({'dataset':row.dataset,'dataset_class':row.dataset_class,'expected_model_class':expected,'accepted_model_classes':'|'.join(accepted),'predicted_top1':names[0] if names else '', 'top3':'|'.join(names),'top1_confidence':conf,'top1_correct':bool(names and names[0] in accepted),'top1_primary_correct':bool(names and names[0]==expected),'top3_correct':any(x in names for x in accepted),'relative_path':row.relative_path})
  except Exception as e: preds.append({'dataset':row.dataset,'dataset_class':row.dataset_class,'expected_model_class':str(row.model_class),'predicted_top1':'','top3':'','top1_confidence':0,'top1_correct':False,'top1_primary_correct':False,'top3_correct':False,'accepted_model_classes':str(getattr(row,'accepted_model_classes',row.model_class)),'relative_path':row.relative_path,'error_type':type(e).__name__})
 pred_df=pd.DataFrame(preds); pred_df.to_csv(OUT/'vision_predictions_all_datasets.csv',index=False)
 valid=pred_df[pred_df.predicted_top1.ne('')]; y=valid.expected_model_class; yhat=valid.predicted_top1
 supported=sorted(y.unique()); p,r,f,_=precision_recall_fscore_support(y,yhat,labels=supported,average='macro',zero_division=0)
 summary={'manifest_rows':len(df),'successfully_predicted_rows':len(valid),'classical_baseline_classes':int(base.dataset_class.nunique()) if len(base) else 0,'classical_baseline_images':int(len(feature_rows)),'classical_baseline_target_classes':args.baseline_classes,'classical_baseline_target_images_per_class':args.baseline_images_per_class,'top1_accuracy_accepted_classes':float(valid.top1_correct.mean()) if len(valid) else None,'top1_accuracy_primary_class':float(valid.top1_primary_correct.mean()) if len(valid) else None,'top3_accuracy_accepted_classes':float(valid.top3_correct.mean()) if len(valid) else None,'mean_confidence_when_correct':float(valid.loc[valid.top1_correct,'top1_confidence'].mean()) if valid.top1_correct.any() else None,'mean_confidence_when_wrong':float(valid.loc[~valid.top1_correct,'top1_confidence'].mean()) if (~valid.top1_correct).any() else None,'macro_precision_supported_classes':float(p),'macro_recall_supported_classes':float(r),'macro_f1_supported_classes':float(f),'class_support':valid.expected_model_class.value_counts().to_dict(),'baseline':baseline_results,'low_confidence_threshold':0.59,'threshold_note':'Provisional threshold selected from the existing 270-image evaluation; recalibrate on the expanded held-out set before product use.'}
 (OUT/'vision_evaluation_summary.json').write_text(json.dumps(summary,indent=2))
 if len(valid):
  report=classification_report(y,yhat,labels=supported,output_dict=True,zero_division=0); pd.DataFrame(report).T.to_csv(OUT/'vision_per_class_metrics_supported.csv')
  cm=confusion_matrix(y,yhat,labels=supported); fig,ax=plt.subplots(figsize=(max(8,len(supported)*.45),max(6,len(supported)*.4))); im=ax.imshow(cm); ax.set_title('YOLO classifier confusion matrix — supported classes only'); ax.set_xlabel('Predicted model class'); ax.set_ylabel('Expected model class'); ax.set_xticks(range(len(supported)),supported,rotation=90,fontsize=6); ax.set_yticks(range(len(supported)),supported,fontsize=6); fig.colorbar(im,ax=ax,label='Image count'); fig.tight_layout(); fig.savefig(FIG/'vision_confusion_supported.png',dpi=160); plt.close(fig)
  pred_df[~pred_df.top1_correct].sort_values('top1_confidence',ascending=False).head(20).to_csv(OUT/'vision_worst_errors_top20.csv',index=False)
  pred_df.groupby('dataset').agg(images=('relative_path','count'),top1_accuracy_accepted=('top1_correct','mean'),top3_accuracy_accepted=('top3_correct','mean'),top1_accuracy_primary=('top1_primary_correct','mean')).reset_index().to_csv(OUT/'vision_metrics_by_dataset.csv',index=False)
 comparison=[{'method':'YOLO_best.pt','top1_accuracy':summary['top1_accuracy_accepted_classes'],'macro_precision_supported':summary['macro_precision_supported_classes'],'macro_recall_supported':summary['macro_recall_supported_classes'],'macro_f1_supported':summary['macro_f1_supported_classes']},*baseline_results]
 pd.DataFrame(comparison).to_csv(OUT/'vision_model_comparison.csv',index=False)
 print(f"Vision evaluation complete: {len(valid)}/{len(df)} predictions; accepted top-1={summary['top1_accuracy_accepted_classes']}; accepted top-3={summary['top3_accuracy_accepted_classes']}. Results: {OUT}")
if __name__=='__main__': main()
