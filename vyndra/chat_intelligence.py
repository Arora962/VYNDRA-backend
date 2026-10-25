"""VYNDRA Part 4: small, reproducible chat-intelligence baselines.

This module is deliberately additive: it does not modify app.py or Part 2/3 files.
It provides:
- food-intent classification (TF-IDF + LogisticRegression)
- food-mention extraction (phrase matching against an explicit food lexicon)
- sentiment analysis (VADER)
- topic modelling (TF-IDF + LDA)

Models are trained from the included small, hand-labelled demonstration dataset.
Treat results as a baseline, not as evidence of production performance.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # Keep the API importable before dependencies are installed.
    SentimentIntensityAnalyzer = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART4_DATA = PROJECT_ROOT / "data" / "processed" / "chat_intelligence"
DEFAULT_INTENT_DATA = PART4_DATA / "food_intent_examples.csv"
DEFAULT_MENTION_DATA = PART4_DATA / "food_mention_examples.csv"
DEFAULT_SENTIMENT_DATA = PART4_DATA / "sentiment_examples.csv"
DEFAULT_CHAT_DATA = PART4_DATA / "topic_documents.csv"
DEFAULT_REPORTS = PROJECT_ROOT / "results" / "tables" / "part4"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Part 4 data file not found: {path}\n"
            "Extract the Part 4 ZIP into the backend project root."
        )
    return pd.read_csv(path).fillna("")


def load_food_lexicon(extra_foods: Optional[Iterable[str]] = None) -> list[str]:
    """Load food phrases from bundled lexicon, nutrition table and model classes."""
    foods: set[str] = set()
    lexicon_path = PART4_DATA / "food_lexicon.txt"
    if lexicon_path.exists():
        foods.update(
            line.strip().lower() for line in lexicon_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    # Use the project's nutrition database if one of the known files exists.
    raw_dir = PROJECT_ROOT / "data" / "raw"
    candidate_files = [
        raw_dir / "Db.csv",
        raw_dir / "Selective_Diet_Mapping.csv",
        raw_dir / "selective_diet_mapping.csv",
    ]
    for csv_path in candidate_files:
        if not csv_path.exists():
            continue
        try:
            frame = pd.read_csv(csv_path, nrows=10000)
            for col in frame.columns:
                if any(word in str(col).lower() for word in ("food", "dish", "name", "item")):
                    foods.update(
                        str(x).strip().lower() for x in frame[col].dropna().tolist()
                        if str(x).strip() and len(str(x).strip()) < 90
                    )
        except Exception:
            # A source nutrition file may have a different encoding/schema; bundled
            # lexicon still makes this module usable.
            pass

    classes_path = PROJECT_ROOT / "data" / "processed" / "classes.txt"
    if classes_path.exists():
        foods.update(
            line.strip().lower() for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    if extra_foods:
        foods.update(str(food).strip().lower() for food in extra_foods if str(food).strip())

    # Longer phrases first, then escape all terms for safe literal matching.
    return sorted({f for f in foods if f}, key=lambda x: (-len(x), x))


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def extract_food_mentions(text: str, foods: Optional[Iterable[str]] = None) -> list[str]:
    """Extract known food phrases by case-insensitive whole-phrase matching.

    This is a transparent dictionary baseline, not a general-purpose NER model.
    It can miss misspellings and food names not present in the lexicon.
    """
    normal = _normalise_text(text)
    lexicon = list(foods) if foods is not None else load_food_lexicon()
    matches: list[tuple[int, int, str]] = []
    for food in sorted({str(f).strip().lower() for f in lexicon if str(f).strip()},
                       key=lambda x: (-len(x), x)):
        # Phrase boundaries allow spaces/hyphens but avoid matching inside words.
        pattern = r"(?<![a-z0-9])" + re.escape(food).replace(r"\ ", r"[\s-]+") + r"(?![a-z0-9])"
        for match in re.finditer(pattern, normal):
            matches.append((match.start(), match.end(), food))
    # Keep longest non-overlapping spans; this avoids returning "chicken" inside
    # "chicken tikka masala" as an additional separate mention.
    chosen: list[tuple[int, int, str]] = []
    for start, end, food in sorted(matches, key=lambda x: (x[0], -(x[1] - x[0]), x[2])):
        if any(not (end <= s or start >= e) for s, e, _ in chosen):
            continue
        chosen.append((start, end, food))
    return [food for _, _, food in sorted(chosen, key=lambda x: x[0])]


def train_intent_model(data_path: Path = DEFAULT_INTENT_DATA, random_state: int = 42):
    """Return fitted model, test rows, and evaluation metrics using a held-out split."""
    frame = _read_csv(data_path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{data_path} must contain columns: {sorted(required)}")
    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(str)
    if frame["label"].nunique() < 2 or frame["label"].value_counts().min() < 2:
        raise ValueError("Need at least two examples per intent class for a stratified split.")
    train_x, test_x, train_y, test_y = train_test_split(
        frame["text"], frame["label"], test_size=0.25, random_state=random_state,
        stratify=frame["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=random_state)),
    ])
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    metrics = {
        "test_examples": int(len(test_y)),
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            test_y, pred, output_dict=True, zero_division=0
        ),
        "test_rows": pd.DataFrame({"text": test_x.values, "true_label": test_y.values,
                                   "predicted_label": pred}),
    }
    return model, metrics


def predict_food_intent(text: str, model=None) -> dict:
    """Predict whether a chat message expresses food-related intent."""
    if model is None:
        model, _ = train_intent_model()
    label = str(model.predict([str(text)])[0])
    scores = model.predict_proba([str(text)])[0]
    classes = list(model.classes_)
    return {
        "label": label,
        "is_food_related": label != "not_food",
        "confidence": float(max(scores)),
        "class_probabilities": {str(c): float(s) for c, s in zip(classes, scores)},
    }


def analyze_sentiment(text: str) -> dict:
    """Return VADER compound score and a coarse positive/neutral/negative label."""
    if SentimentIntensityAnalyzer is not None:
        score = SentimentIntensityAnalyzer().polarity_scores(str(text))
    else:
        # Minimal development fallback only; install requirements.txt to use VADER.
        lowered = str(text).lower()
        positive = sum(lowered.count(word) for word in ("good", "great", "love", "delicious", "happy", "wonderful", "tasty"))
        negative = sum(lowered.count(word) for word in ("bad", "hate", "awful", "sad", "disgusting", "terrible"))
        raw = positive - negative
        compound = max(-1.0, min(1.0, raw / max(1, positive + negative)))
        score = {"neg": float(negative > positive), "neu": float(positive == negative),
                 "pos": float(positive > negative), "compound": compound}
    compound = float(score["compound"])
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return {"label": label, "compound": compound, "scores": score}


def run_topic_model(documents: Iterable[str], n_topics: int = 4, random_state: int = 42,
                    top_words: int = 8) -> dict:
    """Fit LDA to a corpus and return topic terms plus each document's dominant topic."""
    docs = [str(d).strip() for d in documents if str(d).strip()]
    if len(docs) < 2:
        raise ValueError("Topic modelling requires at least two non-empty chat documents.")
    n_topics = max(2, min(int(n_topics), len(docs)))
    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=1.0, min_df=1, ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )
    matrix = vectorizer.fit_transform(docs)
    if matrix.shape[1] == 0:
        raise ValueError("No usable words found in the chat corpus.")
    model = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_state, learning_method="batch",
        max_iter=20
    )
    topic_distribution = model.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for index, component in enumerate(model.components_):
        top_indices = component.argsort()[-top_words:][::-1]
        topics.append({"topic_id": int(index), "top_terms": [str(terms[i]) for i in top_indices]})
    assignments = [
        {"document": doc, "dominant_topic": int(topic_distribution[i].argmax()),
         "topic_probabilities": [float(x) for x in topic_distribution[i]]}
        for i, doc in enumerate(docs)
    ]
    return {"topics": topics, "assignments": assignments,
            "n_documents": len(docs), "n_topics": n_topics,
            "note": "Topic modelling is unsupervised; there is no accuracy score without labelled topics."}


def run_mention_evaluation(data_path: Path = DEFAULT_MENTION_DATA) -> dict:
    """Evaluate dictionary mention extraction against manually labelled gold mentions."""
    frame = _read_csv(data_path)
    required = {"text", "gold_mentions"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{data_path} must contain columns: {sorted(required)}")
    tp = fp = fn = 0
    rows = []
    foods = load_food_lexicon()
    for _, row in frame.iterrows():
        gold = {x.strip().lower() for x in str(row["gold_mentions"]).split("|") if x.strip()}
        predicted = set(extract_food_mentions(str(row["text"]), foods))
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        rows.append({"text": row["text"], "gold_mentions": "|".join(sorted(gold)),
                     "predicted_mentions": "|".join(sorted(predicted)),
                     "true_positive": len(gold & predicted), "false_positive": len(predicted - gold),
                     "false_negative": len(gold - predicted)})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"examples": len(frame), "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "precision": precision, "recall": recall, "f1": f1,
            "rows": pd.DataFrame(rows)}


def run_sentiment_evaluation(data_path: Path = DEFAULT_SENTIMENT_DATA) -> dict:
    """Evaluate VADER labels on a small hand-labelled demo set."""
    frame = _read_csv(data_path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{data_path} must contain columns: {sorted(required)}")
    predictions = [analyze_sentiment(text)["label"] for text in frame["text"]]
    return {
        "examples": int(len(frame)),
        "accuracy": float(accuracy_score(frame["label"], predictions)),
        "macro_f1": float(f1_score(frame["label"], predictions, average="macro", zero_division=0)),
        "rows": pd.DataFrame({"text": frame["text"], "true_label": frame["label"],
                              "predicted_label": predictions}),
        "classification_report": classification_report(
            frame["label"], predictions, output_dict=True, zero_division=0
        ),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
