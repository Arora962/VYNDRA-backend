"""Score the saved intent model on the 150 hand-written review messages (the realistic test)."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables/part4"
model = joblib.load(ROOT / "models/food_intent.joblib")
df = pd.read_csv(ROOT / "data/processed/chat_intelligence/chat_messages_handwritten_review.csv")
pred = model.predict(df.text)
labels = ["food_intent", "food_discussion", "not_food"]
report = classification_report(df.label, pred, labels=labels, output_dict=True, zero_division=0)
cm = pd.DataFrame(confusion_matrix(df.label, pred, labels=labels), index=labels, columns=labels)
summary = {"rows": len(df), "accuracy": accuracy_score(df.label, pred), "macro_f1": report["macro avg"]["f1-score"], "per_class": {k: report[k] for k in labels}}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "review_set_intent_metrics.json").write_text(json.dumps(summary, indent=2))
cm.to_csv(OUT / "review_set_intent_confusion.csv")
df.assign(predicted=pred)[df.label != pred].to_csv(OUT / "review_set_intent_errors.csv", index=False)
print(f"Review-set accuracy {summary['accuracy']:.3f}, macro-F1 {summary['macro_f1']:.3f} on {len(df)} messages")
print(cm)