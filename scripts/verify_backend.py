"""Run structural checks and write a machine-readable verification report."""
from pathlib import Path
import ast, json, sys, importlib.util
ROOT=Path(__file__).resolve().parents[1]
files=['app.py','vyndra/database.py','vyndra/backend_api.py','vyndra/nutrition.py','vyndra/matching.py','vyndra/chat_intelligence.py','vyndra/social_recommendations.py']
report={'python_syntax':{},'required_files':{},'notes':['Structural verification only; does not prove model quality or validate on another computer.']}
for rel in files:
 p=ROOT/rel
 report['required_files'][rel]=p.exists()
 if p.exists():
  try: ast.parse(p.read_text(encoding='utf-8')); report['python_syntax'][rel]='PASS'
  except SyntaxError as e: report['python_syntax'][rel]=f'FAIL: {e}'
 else: report['python_syntax'][rel]='NOT CHECKED: file missing'
out=ROOT/'results'/'tables'/'part6_7_verification.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not all(report['required_files'].values()) or any(v!='PASS' for v in report['python_syntax'].values()): sys.exit(1)
