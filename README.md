# VYNDRA — Backend

> *Intelligence behind every bite.*

VYNDRA is a food-aware social analytics backend. It combines **text analytics** (detecting food intent and food mentions in chat messages), **media analytics** (recognising Indian dishes in meal photos) and **social analytics** (a friend graph with food-based recommendations). A nutrition table turns recognised foods into calories and macronutrients.

Course project for **CBS3009 — Advanced Social, Text and Media Analytics** (final-year B.Tech.).

## 1. Idea

A Snapchat-like chat app notices when a message talks about food and nudges the user to send a photo of their meal. The photo is analysed and the result is attached to the message. This repository contains the backend only; the React Native app is built separately.

## 2. Project status

| Component | Status |
|---|---|
| Nutrition lookup and gram scaling (`Db.csv`, 1,014 foods) | Implemented |
| Meal logging in SQLite with input validation | Implemented |
| Food-intent classifier (TF-IDF + Logistic Regression), mention extraction, VADER sentiment, LDA topics | Implemented on a small hand-labelled set |
| Dish classifier (`models/best.pt`, 220 classes) and evaluation scripts | Implemented; not yet exposed as an API route |
| Class-to-nutrition mapping with review workflow | Review file generated; approved mapping `class_to_food.csv` required |
| Social graph, food-interest clusters, recommendations | Implemented on a 20-user synthetic network |
| Photo upload endpoint, users/messages tables, food nudge | **Not yet implemented** |
| Portion sizes from serving data | **Not yet implemented** (default 100 g or user-supplied grams) |

## 3. How it works

```
chat text ──► food intent + food mentions + sentiment ──► nudge (planned)
meal photo ─► YOLO11n-cls (best.pt) ─► top-3 dish labels ─► reviewed class_to_food.csv ─► nutrition (per 100 g, scaled to grams)
friend edges + food interests ─► graph metrics, communities, clusters, friend and food recommendations
```

Unmatched dishes never receive guessed nutrition. Only mappings that you reviewed (`class_to_food.csv`) return values.

## 4. Repository layout

```
app.py                 Flask entry point
vyndra/                backend package (api, database, nutrition, matching, vision, chat, social)
scripts/               data preparation and evaluation scripts
tests/                 pytest tests
data/raw/              small nutrition tables (large image datasets are NOT in git)
data/processed/        class lists, mappings, small demo datasets, test manifest
models/best.pt         our trained dish classifier
results/               evaluation tables and figures produced by the scripts
docs/                  API reference, privacy checklist
```

## 5. Setup (macOS, Python 3.12)

```bash
git clone <repo-url> && cd VYNDRA-backend
python3.12 -m venv ~/.venvs/vyndra-backend      # keep the environment outside the repo
source ~/.venvs/vyndra-backend/bin/activate
python -m pip install -r requirements.txt
```

`models/best.pt` is included. The large image datasets (Khana, Food-101, Kaggle Indian Food Images) are **not** in the repository; see Section 10 for sources.

## 6. Run the server

```bash
python app.py        # http://127.0.0.1:5001
```

Environment variables (all optional): `VYNDRA_HOST`, `VYNDRA_PORT` (default 5001), `VYNDRA_DB_PATH` (default `data/vyndra.sqlite3`), `VYNDRA_CORS_ORIGINS`. Debug mode is off.

## 7. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | health check |
| GET | `/api/privacy` | privacy and retention summary |
| GET | `/api/foods/search?q=idli` | search the nutrition table |
| GET | `/api/nutrition/<food_name>` | per-100 g nutrition (exact normalised name) |
| POST | `/api/nutrition/calculate` | `{"food_name","grams"}` → scaled nutrition |
| POST | `/api/meals` | save a meal: `{"items":[{"food_name","quantity_g"}],"notes"}` |
| GET | `/api/meals`, `/api/meals/<id>` | list and read saved meals |
| POST | `/api/chat/analyze` | `{"text"}` → food intent, food mentions, sentiment |
| GET | `/api/chat/topics` | LDA topics on the bundled chat sample |
| GET | `/api/social/graph` | graph metrics and communities (synthetic network) |
| GET | `/api/social/clusters?n_clusters=3` | food-interest clusters |
| GET | `/api/social/recommendations/<user_id>` | friend and food suggestions |
| GET | `/api/social/dietary-guidance/<food>` | curated guidance |
| GET | `/api/social/healthier-swaps/<food>` | curated swap ideas |

```bash
curl -s http://127.0.0.1:5001/api/health
curl -s -X POST http://127.0.0.1:5001/api/nutrition/calculate \
  -H 'Content-Type: application/json' -d '{"food_name":"idli","grams":150}'
curl -s -X POST http://127.0.0.1:5001/api/chat/analyze \
  -H 'Content-Type: application/json' -d '{"text":"I am craving masala dosa for dinner"}'
```

Full details: `docs/API_REFERENCE.md`.

## 8. Tests

```bash
python -m pytest -q        # 15 tests
```

## 9. Reproducing the evaluations

Some scripts import the package; run them from the repo root with `PYTHONPATH=.`.

```bash
python scripts/classes_dump.py                         # class names of best.pt -> data/processed/classes.txt
python scripts/build_food_mapping.py                   # suggestions -> class_to_food_review.csv (review by hand)
python scripts/map_dataset_classes.py                  # Kaggle folders -> model classes (review)
python scripts/auto_review_image_mapping.py            # approve exact/alias matches only
python scripts/prepare_test_set.py                     # sample <=30 images per approved class
python scripts/evaluate_classifier.py                  # top-1/top-3, per-class metrics, confusion matrix
python scripts/evaluate_vision_smoke.py                # inference check on a few own photos (no accuracy)
PYTHONPATH=. python scripts/evaluate_chat_intelligence.py
PYTHONPATH=. python scripts/evaluate_social_recommendations.py
```

## 10. Data and sources

| Dataset | Use | Notes |
|---|---|---|
| Anuvaad INDB 2024.11 (nutrition table, 1,014 foods; `Db.csv` derived from it) | nutrition values | citation and licence: TODO verify |
| `Selective_Diet_Mapping.csv` | diet, allergen and meal-time tags | tags are rule-generated from about 40 sentence templates; guidance, not clinical truth |
| Kaggle "Indian Food Images" (80 classes, 4,000 images) | external test of the classifier | author and licence: TODO verify |
| Khana (arXiv 2509.06006; about 131K images, 80 classes) | downloaded; not yet used in evaluation | dataset licence: TODO verify |
| Food-101 (Bossard et al., ECCV 2014) | downloaded; not yet used in evaluation | licence listed as "unknown"; scientific fair use |
| Chat, social and guidance files in `data/processed/` | demos and baselines | **synthetic or hand-written** |
| Ultralytics YOLO11 | classifier architecture | AGPL-3.0 |

## 11. Results so far (honest summary)

- **Dish classifier (`best.pt`, 220 classes).** Validation during training: top-1 86.2%, top-5 98.3% (97 epochs). This is not an independent test, and the training-data source is undocumented.
- **External check** on 270 Kaggle images (9 mapped classes, 30 images each): top-1 **31.9%**, top-3 **54.1%**. Best classes: gulab jamun, dal makhani, naan. The weak aloo-gobi score is partly a label-mapping issue. The large gap to validation accuracy is a finding about generalisation. See `results/tables/vision_*.csv`.
- **Chat baselines** on very small sets: intent accuracy 70% (30 test messages, 3 classes), sentiment 96.7% (30 messages), mention extraction 12 examples (not meaningful, the dictionary already contains the answers).
- **Social demo:** 20 synthetic users, 24 edges, 2 communities.

## 12. Limitations and ethics

- Chat and social data are simulated. No real user data and no private chats were scraped.
- Nutrition values are table averages per 100 g, not measurements of a specific dish; portion size is not estimated from the photo.
- No authentication or rate limiting: this is a development server. Do not expose it publicly.
- Raw chat text is not stored by `/api/chat/analyze`; only derived labels, text length and a SHA-256 digest are kept.
- Uploaded images are not retained.

## 13. Team and contribution

TODO: names, roles, and a short statement of what was built by the team versus reused (YOLO11, scikit-learn, NetworkX, VADER, public datasets).