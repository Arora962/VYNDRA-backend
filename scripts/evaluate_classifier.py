#!/usr/bin/env python3
"""Evaluate best.pt against a reviewed test_manifest.csv and save standard metrics."""
from pathlib import Path
import csv, sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from vyndra.vision import predict_image, DEFAULT_MODEL
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    manifest=ROOT/'data/processed/test_manifest.csv'
    if not manifest.is_file(): raise SystemExit(f'Missing {manifest}; run scripts/prepare_test_set.py first.')
    df=pd.read_csv(manifest)
    if df.empty: raise SystemExit('Manifest has no rows.')
    required={'true_class','image_path'}
    if not required.issubset(df.columns): raise SystemExit(f'Manifest requires columns: {sorted(required)}')
    y_true=[]; y_pred=[]; top3=[]; records=[]
    for i,row in df.iterrows():
        image=ROOT/str(row['image_path']); truth=str(row['true_class'])
        try:
            result=predict_image(image,top_k=3)
            preds=result['top_predictions']; top_names=[p['class_name'] for p in preds]
            pred=top_names[0] if top_names else ''
            y_true.append(truth); y_pred.append(pred); top3.append(truth in top_names)
            records.append({'image_path':str(row['image_path']),'dataset_class':row.get('dataset_class',''),'true_class':truth,'top1_class':pred,'top1_confidence':preds[0]['confidence'] if preds else '', 'top2_class':top_names[1] if len(top_names)>1 else '', 'top3_class':top_names[2] if len(top_names)>2 else '', 'top3_correct':truth in top_names,'status':'ok','error':''})
        except Exception as e:
            records.append({'image_path':str(row['image_path']),'dataset_class':row.get('dataset_class',''),'true_class':truth,'top1_class':'','top1_confidence':'','top2_class':'','top3_class':'','top3_correct':False,'status':'error','error':f'{type(e).__name__}: {e}'})
        if (i+1)%25==0: print(f'Processed {i+1}/{len(df)} images')
    out=ROOT/'results/tables'; figs=ROOT/'results/figures'; out.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/'vision_predictions.csv',index=False)
    good=[r for r in records if r['status']=='ok']
    if not good: raise SystemExit('No successful predictions. Check the model/runtime and image paths.')
    truth=[r['true_class'] for r in good]; pred=[r['top1_class'] for r in good]
    labels=sorted(set(truth)|set(pred))
    p,r,f,s=precision_recall_fscore_support(truth,pred,labels=labels,zero_division=0)
    per=pd.DataFrame({'class_name':labels,'precision':p,'recall':r,'f1':f,'support':s}).sort_values(['f1','support'])
    per.to_csv(out/'vision_per_class_metrics.csv',index=False)
    pd.DataFrame(classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0)).transpose().to_csv(out/'vision_classification_report.csv')
    cm=confusion_matrix(truth,pred,labels=labels)
    # Keep a readable chart even for many classes by plotting only the 25 most frequent true classes.
    support=pd.Series(truth).value_counts(); shown=support.head(25).index.tolist()
    idx=[labels.index(x) for x in shown]
    cm_small=cm[idx,:][:,idx]
    fig,ax=plt.subplots(figsize=(max(10,len(shown)*0.45),max(8,len(shown)*0.42)))
    im=ax.imshow(cm_small,interpolation='nearest',aspect='auto')
    ax.set_title('VYNDRA Food Classifier — Confusion Matrix (25 Most Frequent Classes)')
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    ax.set_xticks(range(len(shown)));ax.set_xticklabels(shown,rotation=90,fontsize=7)
    ax.set_yticks(range(len(shown)));ax.set_yticklabels(shown,fontsize=7)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figs/'vision_confusion_matrix.png',dpi=180);plt.close(fig)
    wrong=[x for x in good if x['true_class']!=x['top1_class']]
    pd.DataFrame(wrong).head(20).to_csv(out/'vision_error_analysis_top20.csv',index=False)
    summary={'images_in_manifest':len(df),'successful_predictions':len(good),'inference_errors':len(df)-len(good),'top1_accuracy':accuracy_score(truth,pred),'top3_accuracy':sum(x['top3_correct'] for x in good)/len(good),'macro_f1':float(per['f1'].mean()),'weighted_f1':float((per['f1']*per['support']).sum()/max(1,per['support'].sum())),'independent_test_verified':'no — training data overlap/license provenance must be checked'}
    with (out/'vision_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary));w.writeheader();w.writerow(summary)
    print('\nEvaluation summary')
    for k,v in summary.items(): print(f'{k}: {v}')
    print(f'\nSaved: {out}/vision_summary.csv, vision_predictions.csv, vision_per_class_metrics.csv, vision_classification_report.csv, vision_error_analysis_top20.csv')
    print(f'Saved figure: {figs}/vision_confusion_matrix.png')
    print('IMPORTANT: These metrics are only as trustworthy as the manually reviewed class mappings. Dataset overlap with training is not verified.')
if __name__=='__main__':main()
