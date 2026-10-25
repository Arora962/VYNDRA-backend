#!/usr/bin/env python3
"""Compute centrality, Louvain communities, planted-community agreement and engagement."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'data/processed/social_scale'; OUT=ROOT/'results/tables/part5'; FIG=ROOT/'results/figures/part5'

def main():
 events_path=SRC/'social_events.csv'; users_path=SRC/'social_users.csv'
 if not events_path.exists() or not users_path.exists(): raise SystemExit('Run python scripts/make_social_data.py first.')
 ev=pd.read_csv(events_path); users=pd.read_csv(users_path)
 if len(users)!=60 or len(ev)<1500: raise SystemExit(f'Requires 60 users and >=1500 events; found {len(users)} and {len(ev)}.')
 G=nx.DiGraph(); G.add_nodes_from(users.user_id.astype(str));
 for r in ev.itertuples():
  if G.has_edge(r.source_user,r.target_user): G[r.source_user][r.target_user]['weight']+=1
  else: G.add_edge(r.source_user,r.target_user,weight=1)
 # Community detection uses the undirected weighted graph; directed graph retained for in/out degree.
 U=G.to_undirected(); communities=list(nx.community.louvain_communities(U,weight='weight',seed=42)) if U.number_of_edges() else [{n} for n in U]
 modularity=nx.community.modularity(U,communities,weight='weight') if U.number_of_edges() else 0.0
 communities=sorted([sorted(x) for x in communities],key=lambda x:(-len(x),x)); predicted={u:i for i,c in enumerate(communities) for u in c}
 planted=dict(zip(users.user_id.astype(str),users.planted_community.astype(int)))
 ari=adjusted_rand_score([planted[u] for u in G.nodes],[predicted[u] for u in G.nodes]); nmi=normalized_mutual_info_score([planted[u] for u in G.nodes],[predicted[u] for u in G.nodes])
 pagerank=nx.pagerank(G,weight='weight'); bet=nx.betweenness_centrality(G,normalized=True,weight=None); close=nx.closeness_centrality(G); indeg=dict(G.in_degree(weight='weight')); outdeg=dict(G.out_degree(weight='weight'))
 try: eig=nx.eigenvector_centrality(U,max_iter=2000,weight='weight')
 except nx.NetworkXException: eig={u:0.0 for u in U}
 rows=[]
 for u in G.nodes:
  rows.append({'user_id':u,'in_degree':indeg[u],'out_degree':outdeg[u],'betweenness':bet[u],'closeness':close[u],'eigenvector':eig[u],'pagerank':pagerank[u],'community':predicted[u],'planted_community':planted[u]})
 central=pd.DataFrame(rows).sort_values('pagerank',ascending=False); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); central.to_csv(OUT/'social_centrality.csv',index=False)
 comm_rows=[{'id':i,'members':'|'.join(c),'size':len(c)} for i,c in enumerate(communities)]; pd.DataFrame(comm_rows).to_csv(OUT/'social_communities.csv',index=False)
 ev['timestamp']=pd.to_datetime(ev.timestamp,utc=True); meals=ev[ev.event_type.eq('meal_share')]; react=ev[ev.event_type.eq('reaction')]
 meals_user=meals.groupby('source_user').size().to_dict(); reactions_meal=react.groupby('meal_id').size() if not react.empty else pd.Series(dtype=float)
 # For reaction-to-meal ratio, use meal-linked reactions where meal_id is present; report missing linkage explicitly.
 linked=react[react.meal_id.fillna('').astype(str).ne('')]; per_meal=linked.groupby('meal_id').size(); # Response latency: each reply is matched to the latest prior message from the same sender/recipient pair.
 message_times={}; response_minutes=[]
 for event in ev.sort_values('timestamp').itertuples():
  pair=(str(event.source_user),str(event.target_user))
  if event.event_type=='message': message_times[pair]=event.timestamp
  elif event.event_type=='reply' and pair in message_times:
   response_minutes.append(max(0.0,(event.timestamp-message_times[pair]).total_seconds()/60))
 median_response_minutes=float(np.median(response_minutes)) if response_minutes else None
 engagement={'meal_shares_by_user':{u:int(meals_user.get(u,0)) for u in G.nodes},'reactions_per_meal_with_linked_reaction':{str(k):int(v) for k,v in per_meal.items()},'median_response_time_minutes':median_response_minutes,'note':'The generated event schema links reactions to meals only when meal_id is present; unlinked reactions are not silently attributed.'}
 (OUT/'social_metrics.json').write_text(json.dumps({'users':len(users),'events':len(ev),'modularity':modularity,'adjusted_rand_index_vs_planted':ari,'normalized_mutual_info_vs_planted':nmi,'communities':comm_rows,'engagement':engagement},indent=2))
 # Network plot, node size by PageRank and colour by detected community.
 pos=nx.spring_layout(U,seed=42,weight='weight'); colors=[predicted[n] for n in U.nodes]; sizes=[3000* pagerank[n]+100 for n in U.nodes]
 fig,ax=plt.subplots(figsize=(12,9)); nx.draw_networkx_edges(U,pos,alpha=.18,ax=ax); nx.draw_networkx_nodes(U,pos,node_color=colors,node_size=sizes,cmap=plt.cm.tab10,ax=ax); nx.draw_networkx_labels(U,pos,font_size=6,ax=ax); ax.set_title('VYNDRA synthetic social network: Louvain community and PageRank size'); ax.set_xlabel('Spring-layout x coordinate'); ax.set_ylabel('Spring-layout y coordinate'); ax.axis('off'); fig.tight_layout(); fig.savefig(FIG/'social_network_communities.png',dpi=180); plt.close(fig)
 print(f'Wrote social analytics for {len(users)} users / {len(ev)} events; modularity={modularity:.4f}; ARI={ari:.4f}; NMI={nmi:.4f}')
if __name__=='__main__': main()
