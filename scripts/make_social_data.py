#!/usr/bin/env python3
"""Generate a deterministic synthetic 60-user social network and >=1,500 events."""
from __future__ import annotations
import argparse, random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); p.add_argument('--events',type=int,default=1800); a=p.parse_args()
 if a.events<1500: p.error('--events must be >=1500')
 rng=random.Random(a.seed); out=ROOT/'data/processed/social_scale'; out.mkdir(parents=True,exist_ok=True)
 users=[f'u{i:03d}' for i in range(1,61)]; community={u:(i//15) for i,u in enumerate(users)}
 events=[]; start=datetime(2025,1,1,tzinfo=timezone.utc); last_time={}
 types=['message','reply','reaction','meal_share']
 for idx in range(a.events):
  src=rng.choice(users); same=[u for u in users if community[u]==community[src] and u!=src]; cross=[u for u in users if community[u]!=community[src]]
  tgt=rng.choice(same if rng.random()<0.82 else cross)
  # Timestamped directed events. Add small increments and occasional longer delays.
  ts=start+timedelta(minutes=idx*4+rng.randint(0,3))
  typ=rng.choices(types,weights=[.42,.24,.22,.12])[0]
  meal_id=f'meal_{idx:05d}' if typ=='meal_share' else ''
  events.append({'event_id':f'e{idx+1:05d}','timestamp':ts.isoformat(),'source_user':src,'target_user':tgt,'event_type':typ,'meal_id':meal_id,'planted_source_community':community[src],'planted_target_community':community[tgt]})
  if typ=='message': last_time[(src,tgt)]=ts
 # Attach reactions to previously shared meals so reactions-per-meal is measurable.
 meal_ids=[e['meal_id'] for e in events if e['event_type']=='meal_share' and e['meal_id']]
 for event in events:
  if event['event_type']=='reaction' and meal_ids:
   event['meal_id']=rng.choice(meal_ids)
 df=pd.DataFrame(events); df.to_csv(out/'social_events.csv',index=False)
 pd.DataFrame([{'user_id':u,'name':f'User {i:02d}','planted_community':community[u]} for i,u in enumerate(users,1)]).to_csv(out/'social_users.csv',index=False)
 # compatibility files consumed by existing social module
 edges=df[['source_user','target_user']].rename(columns={'source_user':'source','target_user':'target'}).drop_duplicates()
 legacy=ROOT/'data/processed/social_recommendations'; legacy.mkdir(parents=True,exist_ok=True); edges.to_csv(legacy/'social_edges.csv',index=False)
 profile_rows=[]
 foods=['dosa','idli','biryani','salad','paneer','dal','pizza','poha','sushi','noodles','rajma','chole']
 for u in users:
  own=[f for f in foods if rng.random()<(.7 if foods.index(f)%4==community[u] else .18)]
  profile_rows.append({'user_id':u,'food_interests':'|'.join(own or [rng.choice(foods)])})
 pd.DataFrame(profile_rows).to_csv(legacy/'user_food_profiles.csv',index=False)
 print(f'Wrote {len(users)} users and {len(df)} timestamped directed events to {out}')
if __name__=='__main__': main()
