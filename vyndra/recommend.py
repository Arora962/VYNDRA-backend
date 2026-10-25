"""Nutrition-based clustering and recommendation helpers for VYNDRA."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from .nutrition import get_food_table, normalize_food_name
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tables/part6'
MACROS=['energy_kcal','protein_g','carb_g','fat_g','fibre_g']

def _tags(value):
    if pd.isna(value) or not str(value).strip(): return []
    return [x.strip() for x in re.split(r'[,|;]',str(value)) if x.strip() and x.strip().casefold() not in {'none', 'nan', 'null'}]

@lru_cache(maxsize=1)
def _clustered():
    df=get_food_table().copy()
    missing=set(MACROS)-set(df.columns)
    if missing: raise ValueError(f'Nutrition table missing macro columns: {sorted(missing)}')
    numeric=df[MACROS].apply(pd.to_numeric,errors='coerce'); good=numeric.notna().all(axis=1)
    df=df.loc[good].reset_index(drop=True); numeric=numeric.loc[good].reset_index(drop=True)
    X=StandardScaler().fit_transform(numeric)
    # Choose k by maximum silhouette among 2..10; save all requested diagnostics.
    metrics=[]; candidates=[]
    max_k=min(10,len(df)-1)
    for k in range(2,max_k+1):
        model=KMeans(n_clusters=k,random_state=42,n_init=20); labels=model.fit_predict(X)
        sil=float(silhouette_score(X,labels)); db=float(davies_bouldin_score(X,labels)); metrics.append({'k':k,'silhouette':sil,'davies_bouldin':db}); candidates.append((sil,k,labels,model))
    _,k,labels,model=max(candidates,key=lambda x:x[0]); df['cluster_id']=labels
    desc={}
    for cid,g in df.groupby('cluster_id'):
        means=g[MACROS].mean(); kcal=float(means.energy_kcal); protein=float(means.protein_g); carb=float(means.carb_g); fat=float(means.fat_g)
        if kcal<100: text='Lower-energy foods on average'
        elif protein>15: text='Higher-protein foods on average'
        elif carb>25: text='Carbohydrate-rich foods on average'
        elif fat>15: text='Higher-fat foods on average'
        else: text='Mixed-macronutrient foods on average'
        desc[int(cid)]={'description':text,'food_count':int(len(g)),'mean_energy_kcal_per_100g':round(kcal,2),'mean_protein_g_per_100g':round(protein,2),'mean_carbs_g_per_100g':round(carb,2),'mean_fat_g_per_100g':round(fat,2)}
    return df,X,metrics,desc,k

def run_clustering():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage,dendrogram
    from sklearn.decomposition import PCA
    df,X,metrics,desc,k=_clustered(); OUT.mkdir(parents=True,exist_ok=True); figdir=ROOT/'results/figures/part6'; figdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(OUT/'food_kmeans_metrics.csv',index=False)
    out=df[['food_name',*MACROS,'cluster_id']].copy(); out.to_csv(OUT/'food_clusters.csv',index=False)
    import json; (OUT/'food_cluster_descriptions.json').write_text(json.dumps({'selected_k':k,'selection_criterion':'maximum silhouette over k=2..10','clusters':desc},indent=2))
    pca=PCA(n_components=2,random_state=42); coords=pca.fit_transform(X)
    fig,ax=plt.subplots(figsize=(9,6)); scatter=ax.scatter(coords[:,0],coords[:,1],c=df.cluster_id,s=16,alpha=.75,cmap=plt.cm.tab10); ax.set_title('Nutrition-based food clusters (PCA projection)'); ax.set_xlabel(f'Principal component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)'); ax.set_ylabel(f'Principal component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)'); fig.colorbar(scatter,ax=ax,label='K-Means cluster'); fig.tight_layout(); fig.savefig(figdir/'food_clusters_pca.png',dpi=180); plt.close(fig)
    # Ward dendrogram over a deterministic sample to keep figure size manageable; metrics use all foods.
    sample=min(80,len(X)); Z=linkage(X[:sample],method='ward')
    fig,ax=plt.subplots(figsize=(14,6)); dendrogram(Z,labels=df.food_name.iloc[:sample].tolist(),leaf_rotation=90,leaf_font_size=6,ax=ax); ax.set_title(f'Ward hierarchical dendrogram (first {sample} nutrition rows)'); ax.set_xlabel('Food name'); ax.set_ylabel('Ward linkage distance'); fig.tight_layout(); fig.savefig(figdir/'food_ward_dendrogram.png',dpi=180); plt.close(fig)
    ward_rows=[]
    for wk in range(2, min(10, len(X)-1)+1):
        wlabels=AgglomerativeClustering(n_clusters=wk, linkage='ward').fit_predict(X)
        ward_rows.append({'k':wk,'silhouette':float(silhouette_score(X,wlabels))})
    pd.DataFrame(ward_rows).to_csv(OUT/'food_hierarchical_silhouette.csv',index=False)
    # TF-IDF top terms for each rule-generated diet/mood tag group.
    raw=get_food_table().copy()
    for group_col in ('diet_tags','mood_tags'):
        if group_col not in raw.columns: continue
        grouped=[]
        for tag in sorted({tag for value in raw[group_col].fillna('') for tag in _tags(value)}):
            names=raw.loc[raw[group_col].fillna('').map(lambda v: tag in _tags(v)), 'food_name'].astype(str).tolist()
            if not names: continue
            vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
            matrix=vectorizer.fit_transform(names)
            terms=np.array(vectorizer.get_feature_names_out())
            weights=np.asarray(matrix.mean(axis=0)).ravel()
            top=terms[np.argsort(weights)[-10:][::-1]].tolist()
            grouped.append({'group_type':group_col,'group':tag,'food_count':len(names),'top_terms':'|'.join(top)})
        pd.DataFrame(grouped).to_csv(OUT/f'food_tfidf_{group_col}.csv',index=False)
    print(f'Clustered {len(df)} foods; selected k={k}; wrote K-Means and Ward silhouette results for k=2..{max(x["k"] for x in metrics)}')

def food_insights(food_name:str, diet:str|None=None)->dict:
    df,X,metrics,desc,k=_clustered(); query=normalize_food_name(food_name); norm=df.food_name.map(normalize_food_name); hits=df[norm.eq(query)]
    if hits.empty: return {'found':False,'food':food_name,'message':'Food not found in the nutrition table.'}
    row=hits.iloc[0]; cid=int(row.cluster_id)
    diet_map=ROOT/'data/raw/Selective_Diet_Mapping.csv'; tags={}
    if diet_map.exists():
        m=pd.read_csv(diet_map).fillna(''); m['_norm']=m.food_name.map(normalize_food_name); match=m[m._norm.eq(query)]
        if not match.empty: tags=match.iloc[0].to_dict()
    same=df[df.cluster_id.eq(cid)&df.food_name.ne(row.food_name)].copy(); kcal=float(row.energy_kcal)
    same['kcal_saving_pct']=(kcal-same.energy_kcal)/kcal*100 if kcal>0 else -999
    same=same[same.kcal_saving_pct.ge(15)].sort_values(['kcal_saving_pct','food_name'],ascending=[False,True]).head(5)
    swaps=[{'food':str(r.food_name),'kcal_saving_pct':round(float(r.kcal_saving_pct),2)} for r in same.itertuples()]
    diet_tags=_tags(tags.get('diet_tags',''))
    allowed_diets=['Vegetarian','Vegan','Keto','Pescatarian','Mediterranean','Paleo']
    if diet:
        fit=100 if diet in diet_tags else 0
        diet_fit={'diet':diet,'score':fit,'label':'tagged as suitable' if fit else 'not tagged as suitable','notes':['Fit is based only on rule-generated Selective_Diet_Mapping tags; it is not a clinical assessment.']}
    else:
        diet_fit={'scores':{name:(100 if name in diet_tags else 0) for name in allowed_diets},'notes':['Scores indicate only whether the rule-generated tag is present; 0 means not tagged, not proof of unsuitability.']}
    return {'found':True,'food':str(row.food_name),'cluster':desc[cid],'allergens':_tags(tags.get('allergens','')),'diet_tags':diet_tags,'diet_fit':diet_fit,'best_time':str(tags.get('best_time_to_eat','Not specified') or 'Not specified'),'swaps':swaps,'tags_source':'Selective_Diet_Mapping.csv; tags are rule-generated and require caution'}

def main(): run_clustering()
if __name__=='__main__': main()
