# VYNDRA — Methodology

How each component works, why it was chosen, and what its limits are. Metrics are in [`RESULTS.md`](RESULTS.md).

## 1. End-to-end pipelines

```
Problem → Data collection → Preprocessing → Feature representation → Analytics/ML → Evaluation → Insights
```

| Pipeline | Stages |
|---|---|
| Chat | message → normalise → intent classifier → food-mention lookup → nudge decision → store message |
| Photo | upload validation → YOLO top-3 → confidence filter → reviewed class→food mapping → portion → per-100 g scaling → meal saved |
| Social | events → weighted directed graph → centralities + Louvain → recommendations |
| Clustering | nutrition table → scale macros → K-Means / Ward → silhouette → interpretation |

## 2. Nutrition lookup and portions

- **Normalisation:** lower-case, strip punctuation, collapse spaces, a small alias list (e.g. `idly→idli`, `panner→paneer`).
- **Exact matching only** at runtime: the API never fuzzy-matches a food name to produce nutrition. Fuzzy methods are used only to *suggest* mappings for human review.
- **Scaling:** `nutrient = value_per_100g × grams / 100`.
- **Portion resolution order:** explicit grams (1–5,000) › servings × estimated grams per serving (0 < servings ≤ 20) › serving estimate › flagged 100 g default. Serving grams = `unit_serving_energy_kcal / energy_kcal × 100`, accepted if 5–1,000 g.

*Why:* a wrong confident nutrition value is worse than "unknown", so unknown stays unknown.

## 3. Dish recognition (media analytics)

- **Model:** Ultralytics YOLO11n-cls classifier, 220 Indian-food classes (`models/best.pt`), loaded once per process.
- **Decision rules:** top-3 predictions; confidence < 0.10 dropped; `low_confidence` if < 0.59.
- **Mapping gate:** only reviewed `exact`/`family` rows of `class_to_food.csv` yield nutrition. The user can correct a wrong dish by picking from five suggestions.
- **0.59 threshold:** chosen by maximising F1 for "top-1 correct vs incorrect" on the original 270-image result file. It is **provisional**: the test set is small and the class mapping imperfect. Recalibrate on a held-out set before product use.
- **Baselines:** HSV colour histograms (8×4×4 bins on 128×128 images) with Gaussian Naïve Bayes, Decision Tree and k-NN, to show how much a colour-only method achieves.
- **Evaluation design:** images from independent external datasets (Kaggle, Khana), mapped through reviewed class maps, 30 per class, seed 42. Metrics: top-1/top-3 (accepted-class and primary-label), precision/recall/F1 on *supported* classes, confusion matrix, worst-confident errors.
- **Training:** `best.pt` was trained outside this repository. **TODO:** document training data, epochs, augmentations and the validation score, because the training-data source decides whether the external test is truly independent.

## 4. Class-to-food mapping

1. Character n-gram (2–4) TF-IDF over normalised food names; cosine similarity between each class name and every food.
2. Status: exact-text match (needs review), similarity ≥ 0.25 (needs review), else no match.
3. **Manual review** sets `exact`, `family` (close relative, e.g. "aloo gobi masala" → "Potato cauliflower") or `none`.
4. `evaluate_matching.py` compares exact-normalised and fuzzy methods against the reviewed mapping. This measures agreement with the reviewer, **not** independent correctness.

## 5. Text analytics

| Task | Method | Notes |
|---|---|---|
| Intent (3 classes) | 12 configurations: {Bag-of-Words, TF-IDF word, TF-IDF char} × {Multinomial NB, Logistic Regression, Linear SVM (calibrated), Decision Tree} | Selected by **validation accuracy only**; test metrics reported after selection. The saved model is Bag-of-Words + Multinomial NB. |
| Nudge | `intent ∈ {food_intent, food_discussion}` **and** probability ≥ threshold **and** a food phrase found | Threshold maximises nudge F1 on the validation split, capped at 0.90 because template data saturates probabilities. |
| Food mentions | Runtime: whole-phrase dictionary match (≈90 phrases, longest non-overlapping span). Offline comparison: rapidfuzz fuzzy matching (threshold 78) | The dictionary baseline cannot find unseen foods or misspellings. |
| Sentiment | VADER compound score (≥ 0.05 positive, ≤ −0.05 negative) | Synthetic labels are heuristic, so scores are indicative only. |
| Topics | Offline: CountVectorizer (1–2-grams) + LDA with k ∈ {4,6,8,10}; API demo: TF-IDF + LDA, k = 4 on 20 documents | k is chosen by in-sample perplexity, which favours the largest k; interpret topics qualitatively. |

*Why synthetic data:* real chats were not collected for privacy reasons. Synthetic templates are reproducible but unrealistic, so a separate hand-written review set is used as the realistic check.

## 6. Social network analytics

- **Graph:** directed, weighted (edge weight = number of events); community detection on the undirected weighted graph.
- **Centralities:** PageRank (influence), betweenness (brokerage), closeness (reach), eigenvector (connection to well-connected users), degree.
- **Communities:** Louvain (seed 42), modularity, and agreement with planted groups via Adjusted Rand Index and Normalised Mutual Information.
- **Recommendations:** friends by shared food interests and mutual friends (excluding self and existing friends); foods by neighbours' interests (excluding the user's own); curated dietary guidance and swaps.
- **Limit:** planted communities make recovery easy; centrality here is structural, **not** real-world influence.

## 7. Food clustering

K-Means (k = 2…10) on standardised energy, protein, carbohydrate and fat per 100 g; k chosen by maximum silhouette (Davies–Bouldin also reported); Ward hierarchical clustering with a dendrogram; PCA for visualisation; TF-IDF summaries of diet/mood tags per group. A high silhouette is a selection heuristic, **not** proof of natural food categories.

## 8. Data model (SQLite)

`users` (diet, allergens) · `meals` (totals, source, metadata) · `meal_items` (per-food grams and nutrients, `0 < grams ≤ 5000`) · `messages` (text, intent, probability, nudge, optional meal) · `reactions` · `chat_events` (derived analysis metadata only). Parameterised SQL is used throughout; foreign keys link meals to items and messages to meals.

## 9. Privacy, security and ethics

- Synthetic chat and social data; no real user data or private chats.
- `/api/chat/send` stores text (needed to show conversations); `/api/chat/analyze` stores only derived labels, length and a SHA-256 digest.
- Uploaded photos are deleted unless `keep=true`; extension, size and content are validated.
- No authentication, authorisation or rate limiting: **development only**.
- Diet/allergen guidance is tag-based and not medical advice.

## 10. Known limitations

1. Dish recognition gives a likely label, not portion size.
2. External accuracy (≈30% top-1) is far below training validation; domain shift and unknown training-data provenance are the main suspects.
3. Template-generated chat data inflates text metrics; the realistic review set still needs human label review.
4. Diet, mood and allergen tags are rule-generated.
5. Only 115 of 220 dish classes have an approved nutrition mapping.
6. Food-101 is mapped but not yet evaluated; portion estimation is not yet evaluated against weighed meals.
7. Results can differ in the last decimals across library versions; pin versions with `requirements.txt`.