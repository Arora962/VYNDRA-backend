#!/usr/bin/env python3
"""Run deterministic Part 4 evaluations and save reports under results/tables/part4."""
import json
from pathlib import Path

import pandas as pd

from vyndra.chat_intelligence import (
    DEFAULT_CHAT_DATA, DEFAULT_REPORTS, _read_csv, analyze_sentiment,
    extract_food_mentions, load_food_lexicon, predict_food_intent,
    run_mention_evaluation, run_sentiment_evaluation, run_topic_model, save_json,
    train_intent_model,
)


def main():
    DEFAULT_REPORTS.mkdir(parents=True, exist_ok=True)

    model, intent = train_intent_model()
    intent["test_rows"].to_csv(DEFAULT_REPORTS / "intent_predictions.csv", index=False)
    save_json(DEFAULT_REPORTS / "intent_metrics.json", {
        "test_examples": intent["test_examples"], "accuracy": intent["accuracy"],
        "macro_f1": intent["macro_f1"], "classification_report": intent["classification_report"],
        "random_state": 42,
        "caveat": "Small hand-labelled demo dataset; results are a baseline, not production validation."
    })

    mention = run_mention_evaluation()
    mention["rows"].to_csv(DEFAULT_REPORTS / "mention_predictions.csv", index=False)
    save_json(DEFAULT_REPORTS / "mention_metrics.json", {
        k: v for k, v in mention.items() if k != "rows"
    })

    sentiment = run_sentiment_evaluation()
    sentiment["rows"].to_csv(DEFAULT_REPORTS / "sentiment_predictions.csv", index=False)
    save_json(DEFAULT_REPORTS / "sentiment_metrics.json", {
        "examples": sentiment["examples"], "accuracy": sentiment["accuracy"],
        "macro_f1": sentiment["macro_f1"],
        "classification_report": sentiment["classification_report"],
        "caveat": "Small hand-labelled demo dataset; VADER is a lexicon baseline."
    })

    chats = _read_csv(DEFAULT_CHAT_DATA)
    topics = run_topic_model(chats["text"].astype(str).tolist(), n_topics=4, random_state=42)
    save_json(DEFAULT_REPORTS / "topic_model.json", topics)
    pd.DataFrame(topics["assignments"]).to_csv(DEFAULT_REPORTS / "topic_assignments.csv", index=False)

    examples = [
        "I'm craving dosa and chutney for dinner!",
        "That biryani was amazing.",
        "The weather is nice today.",
        "I feel tired and don't want to eat."
    ]
    pd.DataFrame([{
        "text": text,
        "food_intent": predict_food_intent(text, model)["label"],
        "food_mentions": "|".join(extract_food_mentions(text)),
        "sentiment": analyze_sentiment(text)["label"],
    } for text in examples]).to_csv(DEFAULT_REPORTS / "demo_predictions.csv", index=False)

    summary = {
        "intent_accuracy": intent["accuracy"],
        "intent_macro_f1": intent["macro_f1"],
        "mention_precision": mention["precision"],
        "mention_recall": mention["recall"],
        "mention_f1": mention["f1"],
        "sentiment_accuracy": sentiment["accuracy"],
        "sentiment_macro_f1": sentiment["macro_f1"],
        "topic_count": topics["n_topics"],
        "topic_documents": topics["n_documents"],
        "note": "Use these results as baseline results only. The included examples are small and hand-labelled."
    }
    save_json(DEFAULT_REPORTS / "part4_summary.json", summary)
    print("Part 4 evaluations completed.")
    print(json.dumps(summary, indent=2))
    print(f"Reports saved to: {DEFAULT_REPORTS}")


if __name__ == "__main__":
    main()
