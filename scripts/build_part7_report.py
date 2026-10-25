"""Aggregate existing evaluation artifacts without inventing missing metrics."""
from pathlib import Path
import json, csv
ROOT=Path(__file__).resolve().parents[1]; TABLES=ROOT/'results'/'tables'; OUT=TABLES/'part7_summary.json'
summary={'generated_from_existing_files':True,'artifacts':{},'interpretation':[
 'Only metrics present in existing run outputs are reported.',
 'Vision scores are meaningful only if the test manifest and labels are independent and correct.',
 'Chat scores on small bundled examples are development baselines, not production performance.' ]}
for p in sorted(TABLES.rglob('*')):
 if p.is_file() and p.name not in {'part7_summary.json','part6_7_verification.json'}:
  rec={'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size}
  if p.suffix.lower()=='.json':
   try: rec['content']=json.loads(p.read_text(encoding='utf-8'))
   except Exception: rec['content_status']='unreadable JSON'
  elif p.suffix.lower()=='.csv':
   try:
    with p.open(newline='',encoding='utf-8-sig') as f:
     reader=csv.reader(f); header=next(reader,[]); rows=sum(1 for _ in reader)
    rec.update({'columns':header,'data_rows':rows})
   except Exception: rec['content_status']='unreadable CSV'
  summary['artifacts'][str(p.relative_to(ROOT))]=rec
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(f'Wrote {OUT.relative_to(ROOT)} with {len(summary["artifacts"])} existing artifacts.')
