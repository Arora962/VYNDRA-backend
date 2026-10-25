#!/usr/bin/env python3
"""Compare text classifiers, evaluate fuzzy mention extraction, VADER and LDA topics."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/chat_intelligence'
OUT=ROOT/'results/tables/part4'; FIG=ROOT/'results/figures/part4'

def ensure_data():
    path=DATA/'chat_messages_synthetic.csv'
    if not path.exists():
        raise SystemExit('Missing synthetic chat dataset. Run: python scripts/make_chat_dataset.py')
    df=pd.read_csv(path).fillna('')
    if len(df)<3000 or not {'text','label','split'}.issubset(df): raise SystemExit('Dataset must have >=3000 rows and text/label/split columns.')
    if df.text.duplicated().any(): raise SystemExit('Duplicate message text found; regenerate dataset.')
    return df

def compare_models(df):
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    tr=df[df.split=='train']; va=df[df.split=='validation']; te=df[df.split=='test']
    results=[]; fitted=[]
    vectorizers={
      'bow_word':CountVectorizer(ngram_range=(1,2), min_df=2, max_features=20000),
      'tfidf_word':TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=True),
      'tfidf_char':TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5), min_df=2, max_features=25000, sublinear_tf=True),
    }
    models={'MultinomialNB':MultinomialNB(),'LogisticRegression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),'LinearSVC':LinearSVC(class_weight='balanced',random_state=42),'DecisionTree':DecisionTreeClassifier(max_depth=30,class_weight='balanced',random_state=42)}
    for vname,vec in vectorizers.items():
      for mname,est in models.items():
        Xtr=vec.fit_transform(tr.text); Xva=vec.transform(va.text); Xte=vec.transform(te.text)
        model=est.fit(Xtr,tr.label)
        val_pred=model.predict(Xva); test_pred=model.predict(Xte)
        p,r,f,_=precision_recall_fscore_support(te.label,test_pred,average='macro',zero_division=0)
        row={'vectorizer':vname,'model':mname,'validation_accuracy':accuracy_score(va.label,val_pred),'accuracy':accuracy_score(te.label,test_pred),'macro_precision':p,'macro_recall':r,'macro_f1':f}
        results.append(row); fitted.append((row,vec,model,Xte,test_pred))
    res=pd.DataFrame(results).sort_values(['validation_accuracy'],ascending=False)
    res.to_csv(OUT/'text_model_comparison.csv',index=False)
    bestrow=res.iloc[0].to_dict(); best=next(x for x in fitted if x[0]['model']==bestrow['model'] and x[0]['vectorizer']==bestrow['vectorizer'])
    _,vec,model,Xte,pred=best
    report=classification_report(te.label,pred,output_dict=True,zero_division=0)
    (OUT/'text_best_classification_report.json').write_text(json.dumps(report,indent=2))
    bad=te.copy(); bad['predicted_label']=pred; bad=bad[bad.label.ne(bad.predicted_label)]; bad.to_csv(OUT/'text_misclassified_messages.csv',index=False)
    labels=sorted(df.label.unique()); cm=confusion_matrix(te.label,pred,labels=labels)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm); ax.set_title(f"Text classifier confusion matrix — {bestrow['model']} / {bestrow['vectorizer']}"); ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right'); ax.set_yticks(range(len(labels)),labels); fig.colorbar(im,ax=ax,label='Message count'); fig.tight_layout(); fig.savefig(FIG/'text_confusion_matrix.png',dpi=160); plt.close(fig)
    # Refit a fresh vectorizer/model pair so no estimator accidentally retains a vocabulary
    # from another configuration in the comparison loop.
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    vectorizer_factories={
      'bow_word':lambda:CountVectorizer(ngram_range=(1,2), min_df=2, max_features=20000),
      'tfidf_word':lambda:TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=True),
      'tfidf_char':lambda:TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5), min_df=2, max_features=25000, sublinear_tf=True),
    }
    estimator_factories={
      'MultinomialNB':lambda:MultinomialNB(),
      'LogisticRegression':lambda:LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),
      'LinearSVC':lambda:LinearSVC(class_weight='balanced',random_state=42),
      'DecisionTree':lambda:DecisionTreeClassifier(max_depth=30,class_weight='balanced',random_state=42),
    }
    selected_vec=vectorizer_factories[str(bestrow['vectorizer'])]()
    selected_Xtr=selected_vec.fit_transform(tr.text)
    selected_model=estimator_factories[str(bestrow['model'])]()
    if bestrow['model']=='LinearSVC':
      selected_estimator=CalibratedClassifierCV(selected_model,cv=3)
      selected_estimator.fit(selected_Xtr,tr.label)
    else:
      selected_estimator=selected_model.fit(selected_Xtr,tr.label)
    selected_pipeline=make_pipeline(selected_vec,selected_estimator)
    models_dir=ROOT/'models'; models_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(selected_pipeline,models_dir/'food_intent_comparison_best.joblib')
    joblib.dump(selected_pipeline,models_dir/'food_intent.joblib')
    # Calibrate the nudge confidence threshold on validation data only, using the same
    # rule as the API: predicted food-related class AND maximum class probability >= threshold.
    val_scores=selected_pipeline.predict_proba(va.text)
    val_labels=selected_pipeline.predict(va.text)
    confidence=val_scores.max(axis=1)
    truth_food=va.label.ne('not_food').to_numpy()
    predicted_food=np.isin(val_labels,['food_intent','food_discussion'])
    thresholds=sorted(set([0.0,1.0,*[float(x) for x in confidence]]))
    threshold_rows=[]
    for threshold in thresholds:
      positive=predicted_food & (confidence>=threshold)
      tp=int(np.sum(positive & truth_food)); fp=int(np.sum(positive & ~truth_food)); fn=int(np.sum(~positive & truth_food))
      precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
      f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
      threshold_rows.append({'threshold':threshold,'precision':precision,'recall':recall,'f1':f1})
    threshold_df=pd.DataFrame(threshold_rows).sort_values(['f1','precision','threshold'],ascending=[False,False,False])
    threshold_df.to_csv(OUT/'nudge_threshold_validation_curve.csv',index=False)
    uncapped_threshold=float(threshold_df.iloc[0].threshold)
    # Synthetic templates can produce near-1.0 confidence saturation. Cap at 0.90 so the
    # threshold remains usable for the frontend and the result explicitly acknowledges this limitation.
    selected_threshold=min(uncapped_threshold,0.90)
    positive=predicted_food & (confidence>=selected_threshold)
    tp=int(np.sum(positive & truth_food)); fp=int(np.sum(positive & ~truth_food)); fn=int(np.sum(~positive & truth_food))
    selected_precision=tp/(tp+fp) if tp+fp else 0.0; selected_recall=tp/(tp+fn) if tp+fn else 0.0
    selected_f1=2*selected_precision*selected_recall/(selected_precision+selected_recall) if selected_precision+selected_recall else 0.0
    (OUT/'nudge_threshold.json').write_text(json.dumps({'threshold':selected_threshold,'uncapped_validation_optimum':uncapped_threshold,'validation_f1_at_threshold':selected_f1,'validation_precision_at_threshold':selected_precision,'validation_recall_at_threshold':selected_recall,'criterion':'maximise binary food-related nudge F1 on validation split only, then cap the threshold at 0.90 because synthetic templates saturate confidence; test split is not used to select threshold','model':bestrow['model'],'vectorizer':bestrow['vectorizer']},indent=2))
    (OUT/'text_comparison_summary.json').write_text(json.dumps({'best_model':bestrow,'models_compared':len(results),'selection_criterion':'highest validation accuracy only; test macro-F1 is reported independently and not used for model selection','justification':'The chosen model is selected by validation accuracy, with test macro precision/recall/F1 reported only after selection. TF-IDF/BoW representations are compared across four model families. The calibrated pipeline provides class probabilities for threshold-based nudges. Synthetic template performance is optimistic; inspect misclassified rows and the hand-authored review set before production use.'},indent=2))
    return res

def evaluate_mentions():
    source=DATA/'mention_review_template.csv'
    if not source.exists(): print('Mention template absent; skipping mention evaluation.'); return
    df=pd.read_csv(source).fillna('')
    if len(df)<150: print(f'Mention template has {len(df)} rows; needs at least 150.'); return
    try:
      from rapidfuzz import fuzz
    except ImportError as e: raise SystemExit('Install rapidfuzz from requirements.txt to evaluate mention matching.') from e
    lex_path=DATA/'food_lexicon.txt'; lex=[x.strip() for x in lex_path.read_text().splitlines() if x.strip()] if lex_path.exists() else []
    foods=sorted(set(lex+[str(x) for x in df.food if str(x).strip()]))
    rows=[]
    for _,r in df.iterrows():
      text=str(r.text).lower(); expected=str(r.food).strip().lower(); best=''; score=0
      for food in foods:
        s=max(fuzz.partial_ratio(food.lower(),text),fuzz.ratio(food.lower(),text))
        if s>score: best,score=food,s
      detected=best if score>=78 else ''
      rows.append({'text':r.text,'expected_food':expected,'detected_food':detected,'similarity':score,'correct':bool(expected and expected==detected) if expected else not bool(detected)})
    pred=pd.DataFrame(rows); pred.to_csv(OUT/'mention_predictions_review.csv',index=False); pred[~pred.correct].to_csv(OUT/'mention_failures_review.csv',index=False)
    tp=sum((pred.expected_food!='')&(pred.detected_food!='')&(pred.expected_food==pred.detected_food)); fp=sum((pred.detected_food!='')&((pred.expected_food=='')|(pred.expected_food!=pred.detected_food))); fn=sum((pred.expected_food!='')&(pred.expected_food!=pred.detected_food))
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
    (OUT/'mention_metrics_review.json').write_text(json.dumps({'rows':len(pred),'precision':precision,'recall':recall,'f1':f1,'threshold':78,'note':'Review expected_food labels before interpreting metrics.'},indent=2))

def sentiment_and_topics(df):
    OUT.mkdir(parents=True,exist_ok=True)
    labelled=df[df.sentiment_label.ne('')].head(500).copy()
    try:
      from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
      if len(labelled)<300: print('Fewer than 300 labelled messages; sentiment step skipped.')
      else:
        analyser=SentimentIntensityAnalyzer(); labelled['vader_compound']=labelled.text.map(lambda x:analyser.polarity_scores(x)['compound']); labelled['vader_label']=labelled.vader_compound.map(lambda x:'positive' if x>=.05 else 'negative' if x<=-.05 else 'neutral'); labelled.to_csv(OUT/'sentiment_vader_predictions.csv',index=False)
        p,r,f,_=precision_recall_fscore_support(labelled.sentiment_label,labelled.vader_label,average='macro',zero_division=0)
        (OUT/'sentiment_vader_metrics.json').write_text(json.dumps({'rows':len(labelled),'macro_precision':p,'macro_recall':r,'macro_f1':f,'note':'Sentiment labels in synthetic data are heuristic and must be reviewed; not independent ground truth.'},indent=2))
    except ImportError:
      print('VADER unavailable: install vaderSentiment from requirements.txt; sentiment step skipped.')
    food=df[df.label.isin(['food_intent','food_discussion'])].text.astype(str).tolist()
    if len(food)<1000: raise SystemExit('Need at least 1,000 food-related messages for LDA.')
    vec=CountVectorizer(stop_words='english',min_df=3,max_df=.95,ngram_range=(1,2)); X=vec.fit_transform(food)
    scores=[]; models=[]
    for k in (4,6,8,10):
      lda=LatentDirichletAllocation(n_components=k,random_state=42,learning_method='batch',max_iter=10); lda.fit(X); score=float(lda.perplexity(X)); scores.append({'n_topics':k,'perplexity':score}); models.append((score,k,lda))
    pd.DataFrame(scores).to_csv(OUT/'topic_count_selection.csv',index=False)
    # Lowest perplexity is a transparent criterion, not a claim of semantic truth.
    _,k,lda=min(models,key=lambda x:x[0]); terms=np.array(vec.get_feature_names_out()); topics=[]
    for i,weights in enumerate(lda.components_): topics.append({'topic':i,'top_terms':terms[np.argsort(weights)[-12:][::-1]].tolist()})
    (OUT/'lda_topics.json').write_text(json.dumps({'selected_n_topics':k,'criterion':'minimum in-sample perplexity among 4, 6, 8, 10; interpret cautiously','topics':topics},indent=2))
    assignments=lda.transform(X).argmax(axis=1); pd.DataFrame({'text':food,'topic':assignments}).to_csv(OUT/'lda_topic_assignments.csv',index=False)

def main():
    df=ensure_data(); res=compare_models(df); evaluate_mentions(); sentiment_and_topics(df)
    print(f"Wrote text model comparison ({len(res)} configurations) to {OUT}")
if __name__=='__main__': main()
