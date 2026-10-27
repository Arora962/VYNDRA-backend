# VYNDRA — Backend

> *Intelligence behind every bite.*

VYNDRA is a food-aware social analytics backend for a chat app. It notices when a message talks about food, nudges the user to share a photo of the meal, recognises the dish in the photo, converts it to calories and macronutrients, and analyses the friend network for influencers, communities and food recommendations.

Course project for **CBS3009 — Advanced Social, Text and Media Analytics** (final-year B.Tech.). The React Native mobile client is a separate repository.

| Pillar | What is implemented | Main code |
|---|---|---|
| **Text analytics** | Food-intent classification (3 classes), food-mention extraction, VADER sentiment, LDA topics | `vyndra/chat_intelligence.py`, `scripts/run_text_analytics.py` |
| **Media analytics** | 220-class dish recognition (YOLO11n-cls), classical HSV-histogram baselines, external-dataset evaluation | `vyndra/vision.py`, `scripts/evaluate_vision_suite.py` |
| **Nutrition** | 1,014 Indian foods, per-100 g scaling, serving-size estimates, reviewed class-to-food mapping | `vyndra/nutrition.py`, `matching.py`, `portions.py` |
| **Social analytics** | PageRank / betweenness / closeness / eigenvector centrality, Louvain communities, friend and food recommendations | `vyndra/social_*.py`, `scripts/run_social_analytics.py` |
| **Food clustering** | K-Means and Ward clustering on macronutrients, TF-IDF diet/mood tag summaries | `vyndra/recommend.py`, `scripts/cluster_foods.py` |

## Contents
1. [Architecture](#1-architecture)
2. [Quick start (run the API)](#2-quick-start-run-the-api)
3. [Configuration](#3-configuration)
4. [Tests](#4-tests)
5. [Reproducing everything from scratch](#5-reproducing-everything-from-scratch)
6. [Where every result comes from](#6-where-every-result-comes-from)
7. [Notebooks](#7-notebooks)
8. [Documentation index](#8-documentation-index)
9. [Troubleshooting](#9-troubleshooting)
10. [Limitations, ethics, licences, team](#10-limitations-ethics-licences-team)

---

## 1. Architecture

```
 Mobile app (React Native)
        │  HTTP / JSON  (multipart for photos)
        ▼
 Flask app (app.py, blueprints)            SQLite (data/vyndra.sqlite3)
 ├─ backend_api.py  users, meals, nutrition, insights ───► users, meals, meal_items,
 ├─ chat_api.py     chat + food-intent nudge               messages, reactions, chat_events
 ├─ photo_api.py    photo → dish → nutrition
 └─ social_api.py   influencers, communities, recommendations
        │
        ├─ models/best.pt             YOLO11n-cls dish classifier (220 classes)
        ├─ models/food_intent.joblib  chat food-intent classifier
        ├─ data/raw/Db.csv            nutrition table (per 100 g)
        └─ data/processed/class_to_food.csv   reviewed class → food mapping
```

```
chat text ─► food intent + mentions + sentiment ─► nudge "share a photo?"
meal photo ─► YOLO top-3 ─► reviewed mapping ─► nutrition (per 100 g × portion) ─► saved meal
friend events + food interests ─► graph metrics, communities, clusters, recommendations
```

Unmapped dishes **never receive guessed nutrition**: only rows you reviewed in `class_to_food.csv` return values.

### Repository layout

```
app.py, config.py        Flask entry point (application factory) and central settings
vyndra/                  backend package: APIs, database, nutrition, matching, vision, chat, social
scripts/                 data preparation, training and evaluation scripts (Section 5)
tests/                   pytest suite
data/raw/                nutrition tables, Khana label files (image datasets are NOT in git)
data/processed/          class lists, reviewed mappings, synthetic datasets, manifests
models/                  best.pt (dish classifier), food_intent.joblib (chat classifier)
results/tables|figures/  all evaluation outputs; part4=text, part5=social, part6=clustering, part7=vision
notebooks/               analytics notebooks (cached-results driven)
docs/                    API, datasets, methodology, results
uploads/                 optional retained photos (empty; only used with keep=true)
```

---

## 2. Quick start (run the API)

Everything the server needs is **already in the repository** (models, nutrition table, mappings). No dataset download is needed to run the API.

**Requirements:** Python 3.12, git, about 3 GB free disk (PyTorch).

```bash
git clone <your-repo-url> VYNDRA-backend && cd VYNDRA-backend
python3.12 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py                             # http://127.0.0.1:5001
```

Smoke test (new terminal):

```bash
curl -s http://127.0.0.1:5001/api/health
curl -s -X POST http://127.0.0.1:5001/api/nutrition/calculate \
  -H 'Content-Type: application/json' -d '{"food_name":"idli","grams":150}'
curl -s -X POST http://127.0.0.1:5001/api/chat/analyze \
  -H 'Content-Type: application/json' -d '{"text":"I am craving masala dosa for dinner"}'
```

The SQLite database is created automatically at `data/vyndra.sqlite3` on first start. The first photo request is slow because the YOLO model loads lazily.

**Connect the mobile app:** the server binds to `0.0.0.0:5001`. Use your computer's **LAN IP** (not `localhost`), e.g. `http://192.168.1.5:5001`, with the phone on the same Wi-Fi. Allow incoming connections in your firewall if needed.

---

## 3. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VYNDRA_HOST` | `0.0.0.0` | bind address |
| `VYNDRA_PORT` | `5001` | port |
| `VYNDRA_DB_PATH` | `data/vyndra.sqlite3` | SQLite file |
| `VYNDRA_CORS_ORIGINS` | `http://localhost:8081,http://127.0.0.1:8081` | comma-separated allowed web origins |
| `VYNDRA_UPLOAD_DIR` | `uploads/` | where `keep=true` photos are stored |
| `VYNDRA_LOG_LEVEL` | `INFO` | logging level |

Fixed limits in `config.py` and the API code: images ≤ 8 MB (JPG, PNG, WEBP, BMP), chat text ≤ 4,000 characters, JSON bodies ≤ 64 KB, portions 1–5,000 g, low-confidence threshold 0.59. Debug mode is off.

---

## 4. Tests

```bash
python -m pytest -q                        # 55 tests pass; 1 optional slow test skips without a photo
python scripts/acceptance_check.py --all   # checks that required outputs and docs exist
```

The slow test runs the real YOLO model. To enable it, put a photo in `data/test_images_own/<model class name>/` and run `python -m pytest -m slow -q`.

---

## 5. Reproducing everything from scratch

All results are already committed in `results/`, so you can read them without running anything. Use this section to regenerate them or to rebuild on a new machine.

Run every command **from the repository root with the virtual environment active**. Set this once per terminal session (some scripts import the `vyndra` package):

```bash
export PYTHONPATH=.                # Windows PowerShell: $env:PYTHONPATH="."
```

### Step 0 — Data and models you need

| Item | Needed for | In the repo? |
|---|---|---|
| `data/raw/Anuvaad_INDB_2024.11.xlsx` (nutrition + serving data), `data/raw/Db.csv`, `data/raw/Selective_Diet_Mapping.csv` | everything | **Yes** |
| `models/best.pt` (trained dish classifier) | photo API, vision evaluation | **Yes** |
| Kaggle *Indian Food Images* (80 classes, 4,000 images) | vision stage 5A and 5B | **No — download** |
| Khana dataset (≈131K images, 80 classes) | vision stage 5B | **No — download** |
| Food-101 | optional extra evaluation | **No** (optional) |

Steps 1–4 and 6–7 need **no image data**. Only Step 5 needs the downloads.

### Step 1 — Dish classes and nutrition mapping

| # | Command | What it does | Output |
|---|---|---|---|
| 1.1 | `python scripts/classes_dump.py` | Reads the 220 class names from `best.pt` | `data/processed/classes.txt` *(included)* |
| 1.2 | `python scripts/build_food_mapping.py` | Suggests a nutrition-table food for each class (character n-gram TF-IDF + cosine). **Suggestions only.** | `data/processed/class_to_food_review.csv` *(included)* |
| 1.3 | **Manual review** | You decide each mapping: `exact`, `family` or `none` | `data/processed/class_to_food.csv` *(included: 115 approved rows = 32 exact + 83 family)* |
| 1.4 | `python scripts/fill_none.py` *(optional)* | Adds explicit `none` rows for classes not yet listed so all 220 classes have a row. Edits `class_to_food.csv` in place, so back it up first. | `data/processed/class_to_food.csv` |
| 1.5 | `python scripts/evaluate_matching.py` | Compares exact-normalised matching with fuzzy matching against your reviewed mapping | `results/tables/part4/matching_method_comparison.{csv,json}` |

> `class_to_food.csv` is hand-reviewed. **Never overwrite it** with generated suggestions. Classes missing from it return `needs_manual_selection` and no nutrition values.

### Step 2 — Text and chat intelligence

| # | Command | What it does | Output |
|---|---|---|---|
| 2.1 | `python scripts/make_chat_dataset.py` | Generates 3,600 synthetic messages (seed 42; 1,200 per class; train/validation/test 70/15/15) and 150 hand-written review messages | `data/processed/chat_intelligence/chat_messages_synthetic.csv`, `chat_messages_handwritten_review.csv`, `mention_review_template.csv` |
| 2.2 | `python scripts/run_text_analytics.py` | Compares 12 classifier configurations (Bag-of-Words / TF-IDF word / TF-IDF char × Naïve Bayes / Logistic Regression / Linear SVM / Decision Tree), selects the best on validation accuracy, calibrates the nudge threshold, runs fuzzy-mention, VADER and LDA experiments, **and saves the production model** | `results/tables/part4/text_*`, `nudge_threshold*`, `mention_*review*`, `sentiment_vader_*`, `topic_count_selection.csv`, `lda_*`; `results/figures/part4/text_confusion_matrix.png`; **`models/food_intent.joblib`** |
| 2.3 | `python scripts/evaluate_chat_intelligence.py` | Baselines on the small hand-labelled files (120 intent, 12 mention, 30 sentiment, 20 topic documents) | `results/tables/part4/intent_*`, `mention_metrics.json`, `mention_predictions.csv`, `sentiment_*`, `topic_*`, `demo_predictions.csv`, `part4_summary.json` |
| 2.4 | `python scripts/evaluate_review_set.py` | Scores the saved model on the 150 hand-written messages (the realistic test) | `results/tables/part4/review_set_intent_{metrics.json,confusion.csv,errors.csv}` |
| 2.5 | `python scripts/run_chat_intelligence_demo.py` | Prints a demo of intent + mentions + sentiment | console only |

> ⚠️ **Two scripts write `models/food_intent.joblib`.**
> `run_text_analytics.py` (step 2.2) saves the model used for all reported results and the 0.90 nudge threshold. `train_food_intent.py` (optional) trains a *different* model (TF-IDF + Logistic Regression on only 120 examples) and overwrites it. On the 150 review messages the first model scores **70.7%** accuracy and the second **45.3%**.
> **Rule: run `python scripts/run_text_analytics.py` last** (or skip `train_food_intent.py`). If you ever run `train_food_intent.py`, re-run step 2.2 afterwards.

### Step 3 — Social network analytics

| # | Command | What it does | Output |
|---|---|---|---|
| 3.1 | `python scripts/make_social_data.py --events 1800` | Generates 60 synthetic users and 1,800 timestamped directed events (message, reply, reaction, meal_share) in 4 planted communities of 15 | `data/processed/social_scale/social_users.csv`, `social_events.csv` |
| 3.2 | `python scripts/run_social_analytics.py` | Builds the weighted directed graph; computes centralities, Louvain communities, ARI/NMI vs planted groups, engagement | `results/tables/part5/social_centrality.csv`, `social_communities.csv`, `social_metrics.json`; `results/figures/part5/social_network_communities.png` |
| 3.3 | `python scripts/evaluate_social_recommendations.py` | Friend/food recommendations, user clusters, guidance and swap examples | `results/tables/part5/part5_summary.json`, `social_graph_metrics.json`, `user_clusters.json`, `user_cluster_assignments.csv`, `friend_recommendations.csv`, `food_recommendations.csv`, `dietary_guidance_examples.json`, `healthier_swap_examples.json` *(generated on demand; not committed)* |

> ⚠️ Step 3.1 **also overwrites** `data/processed/social_recommendations/social_edges.csv` and `user_food_profiles.csv`, the two files the live `/api/social/*` endpoints read. The repository ships the original 20-user demo graph (24 edges, 2 communities); after step 3.1 the API serves the 60-user graph.

### Step 4 — Food clustering

| # | Command | Output |
|---|---|---|
| 4.1 | `python scripts/cluster_foods.py` | `results/tables/part6/food_kmeans_metrics.csv` (K-Means k=2..10: silhouette, Davies–Bouldin), `food_clusters.csv`, `food_cluster_descriptions.json`, `food_hierarchical_silhouette.csv`, `food_tfidf_diet_tags.csv`, `food_tfidf_mood_tags.csv`; `results/figures/part6/food_clusters_pca.png`, `food_ward_dendrogram.png` |

This also makes `/api/foods/<name>/insights` cluster-aware.

### Step 5 — Vision evaluation (needs the image datasets)

#### 5.0 Download the datasets

Image datasets are large and are **not** in git (`.gitignore` excludes them). Put them here:

```
data/raw/indian_food_images/Indian Food Images/<80 class folders>/...jpg
data/raw/khana/khana/<80 class folders such as "aloo gobi">/...jpg
data/raw/food101/                      (optional)
```

1. **Kaggle — Indian Food Images (80 classes, 4,000 images).** Search Kaggle for "Indian Food Images Dataset" and confirm owner, URL and licence (record them in `docs/DATASETS.md`). Download in the browser, or with the Kaggle CLI (`pip install kaggle`, put your API token at `~/.kaggle/kaggle.json`, then `kaggle datasets download -d <owner>/<dataset-slug> -p data/raw/indian_food_images --unzip`). The class folders must end up inside `data/raw/indian_food_images/Indian Food Images/`.
2. **Khana (≈131K images, 80 classes, 500×500).** Paper: *Khana: A Comprehensive Indian Cuisine Dataset*, arXiv:2509.06006. The download link is on the dataset page **https://khana.omkar.xyz**. This project stores the archive as `data/raw/khana/dataset.tar.gz`, then:
   ```bash
   tar -xzf data/raw/khana/dataset.tar.gz -C data/raw/khana
   ls data/raw/khana/khana | head          # should list class folders: aloo gobi, biryani, ...
   ls data/raw/khana/khana | wc -l         # expect 80
   ```
   Adjust the extraction folder if the archive's top-level layout differs; the class folders must sit directly under `data/raw/khana/khana/`. `labels.txt` and `taxonomy.csv` in `data/raw/khana/` are already included.
3. **Food-101 (optional).** Bossard et al., ECCV 2014 (ETH Zürich / Hugging Face `ethz/food101`). A class map exists (`food101_class_map.csv`) but Food-101 was **not** part of the committed evaluation. To add it, extract image folders and pass `--food101-root` in step 5B.

Check what you downloaded (writes `data/processed/file_index.csv`, which is git-ignored; reads no pixels):

```bash
python scripts/inspect_datasets.py
```

#### 5A — Stage-1 external check on Kaggle (270 images, 9 classes)

| # | Command | What it does | Output |
|---|---|---|---|
| 5A.1 | `python scripts/map_dataset_classes.py` | Suggests which model class each of the 80 Kaggle folders corresponds to (fuzzy; **not ground truth**) | `data/processed/image_class_map_review.csv` *(included)* |
| 5A.2 | `python scripts/auto_review_image_mapping.py` | Approves only normalised-exact names and a short curated alias list (9 approved, 71 skipped) | `data/processed/image_class_map.csv` *(included)* |
| 5A.3 | `python scripts/prepare_test_set.py` | Samples ≤ 30 images per approved class (seed 42) | `data/test_images/` (git-ignored) and `data/processed/test_manifest.csv` *(generated; not committed)* |
| 5A.4 | `python scripts/evaluate_classifier.py` | Top-1/top-3, per-class metrics, error analysis, confusion matrix | `results/tables/vision_predictions.csv`, `vision_per_class_metrics.csv`, `vision_classification_report.csv`, `vision_error_analysis_top20.csv`, `vision_summary.csv`; **`results/figures/vision_confusion_matrix.png`** |
| 5A.5 | `python scripts/evaluate_vision_smoke.py --source "YOLO images"` *(optional)* | Runs inference on your own photos to prove the pipeline works (**no accuracy**: the true labels are unknown) | `results/tables/vision_smoke_test.csv` |

#### 5B — Stage-2 evaluation on Kaggle + Khana (690 images) with classical baselines

| # | Command | What it does | Output |
|---|---|---|---|
| 5B.1 | `python scripts/make_dataset_maps.py` *(only if you edit the class lists inside it)* | Writes the reviewed Khana (15 classes) and Food-101 (10 classes) → model-class maps; only approved classes are allowed | `data/processed/khana_class_map.csv`, `food101_class_map.csv` *(included)* |
| 5B.2 | see command below | Builds a reproducible manifest: ≤ 30 images per class per dataset, seed 42, paths relative to each dataset root | `data/processed/vision_test_manifest.csv` *(included: 690 rows = 240 Kaggle + 450 Khana)* |
| 5B.3 | see command below | Runs YOLO top-1/top-3 on the manifest and trains three HSV colour-histogram baselines (Gaussian NB, Decision Tree, k-NN) | `results/tables/part7/*`, `results/figures/part7/*` |

```bash
python scripts/prepare_vision_manifest.py \
  --kaggle-root "data/raw/indian_food_images/Indian Food Images" \
  --khana-root  "data/raw/khana/khana"

python scripts/evaluate_vision_suite.py \
  --kaggle-root "data/raw/indian_food_images/Indian Food Images" \
  --khana-root  "data/raw/khana/khana" \
  --baseline-root "data/raw/khana/khana" \
  --baseline-map data/processed/khana_class_map.csv \
  --baseline-classes 15
```

The second command reproduces the committed numbers (HSV baselines use 15 Khana classes × 100 images). It was reconstructed from `vision_evaluation_summary.json`; if your numbers differ, check the dataset roots first. Own photos placed in `data/test_images_own/<model class name>/` are added automatically as a separate `own_photos` dataset.

`evaluate_vision_suite.py` writes: `vision_evaluation_summary.json`, `vision_metrics_by_dataset.csv`, `vision_model_comparison.csv`, `vision_per_class_metrics_supported.csv`, `vision_predictions_all_datasets.csv`, `vision_worst_errors_top20.csv` and the figures `vision_confusion_supported.png`, `vision_baseline_confusion.png`.

### Step 6 — Portion-size evaluation (optional, not done yet)

Replace the example row in `data/processed/reference_meals_template.csv` with real **weighed** meals (`meal_id, food_name, reference_grams, reference_calories, notes`), then run `python scripts/evaluate_portions.py` → `results/tables/portion_evaluation.csv`.

### Step 7 — Summaries and verification

```bash
python -m pytest -q
python scripts/acceptance_check.py --all
python scripts/verify_backend.py          # → results/tables/part6_7_verification.json
python scripts/build_part7_report.py      # → results/tables/part7_summary.json (inventory of every result file)
```

---

## 6. Where every result comes from

| Result / figure | Generated by |
|---|---|
| `results/figures/part4/text_confusion_matrix.png` | `run_text_analytics.py` |
| `results/figures/part5/social_network_communities.png` | `run_social_analytics.py` |
| `results/figures/part6/food_clusters_pca.png`, `food_ward_dendrogram.png` | `cluster_foods.py` |
| `results/figures/vision_confusion_matrix.png` | `evaluate_classifier.py` (stage 1, 270 Kaggle images) |
| `results/figures/part7/vision_confusion_supported.png`, `vision_baseline_confusion.png` | `evaluate_vision_suite.py` (stage 2) |
| `results/figures/nutrition_*.png`, `portion_serving_size_cleaning.png`, `text_model_comparison.png`, `social_centrality_comparison.png`, `food_silhouette_by_k.png` | executing `notebooks/VYNDRA_analytics.ipynb` |
| `results/tables/part4/*` | steps 1.5, 2.2, 2.3, 2.4 |
| `results/tables/part5/*` | steps 3.2, 3.3 |
| `results/tables/part6/*` | step 4.1 |
| `results/tables/part7/*` | step 5B.3 |
| `results/tables/vision_*.csv` | steps 5A.4 and 5A.5 |

Metrics and their interpretation: [`docs/RESULTS.md`](docs/RESULTS.md).

**Results at a glance** (details and caveats in `docs/RESULTS.md`):

| Component | Result | Caveat |
|---|---|---|
| Intent classifier, synthetic test split | macro-F1 1.00 | template data, optimistic |
| Intent classifier, 150 hand-written messages | accuracy 70.7%, macro-F1 0.697 | labels still need human review |
| Dish classifier, Kaggle + Khana (690 images) | top-1 30.4%, top-3 50.1% | external data; far below training validation |
| Colour-histogram baselines | 14.9–26.1% top-1 | different test split, not directly comparable |
| Social graph (60 users, 1,800 events) | modularity 0.502, ARI/NMI 1.0 | planted communities, synthetic |
| Food clustering (1,014 foods) | k = 2, silhouette 0.490 | one large cluster (885 foods) |

---

## 7. Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/VYNDRA_analytics.ipynb` (source: `.py`) | Data-quality checks (missing values, duplicates, outliers, Atwater energy check), serving-size cleaning, text, social, clustering and vision results, each figure with an *observation vs inference* note |
| `notebooks/VYNDRA_Part7_Results_and_Viva.ipynb` (source: `.py`) | Results dashboard from the saved files, plus viva notes and a release checklist |

Both read **saved results** and never retrain or rescan images, so run Steps 1–5 first. Start Jupyter from the repository root or from `notebooks/`. To regenerate the `.ipynb` files from their `.py` sources and save outputs:

```bash
python -m pip install jupytext nbconvert ipykernel     # development tools; not in requirements.txt
jupytext --to ipynb notebooks/VYNDRA_analytics.py -o notebooks/VYNDRA_analytics.ipynb
jupytext --to ipynb notebooks/VYNDRA_Part7_Results_and_Viva.py -o notebooks/VYNDRA_Part7_Results_and_Viva.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/VYNDRA_analytics.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/VYNDRA_Part7_Results_and_Viva.ipynb
```

Do not set `MPLBACKEND`, otherwise figures are not embedded in the saved notebook.

---

## 8. Documentation index

| Document | Contents |
|---|---|
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Every endpoint with request/response shapes, limits and error codes |
| [`docs/DATASETS.md`](docs/DATASETS.md) | All datasets and files, provenance, folder layout, mapping-file formats, citations |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Pipelines, algorithm choices and rationale, thresholds, privacy, limitations |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Metrics, tables and figures with interpretation |

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: vyndra` when running a script | Run from the repo root and `export PYTHONPATH=.` |
| `503 ... model or required data file is missing` | Check `models/best.pt` and `data/raw/Db.csv` exist |
| Phone cannot reach the server | Use the LAN IP, same Wi-Fi, allow the port in your firewall |
| First photo request is slow | The YOLO model loads on first use |
| `Address already in use` | Set `VYNDRA_PORT` to another port |
| Reported chat accuracy changed | `train_food_intent.py` overwrote the model; re-run `run_text_analytics.py` |
| Vision scripts find 0 images | Check the dataset root paths and folder layout in Step 5.0 |
| Hidden `._*` files on external drives | Ignored by the scripts; delete them if they cause trouble |

---

## 10. Limitations, ethics, licences

- Chat and social data are **synthetic**; no real user data was collected.
- Nutrition values are table averages per 100 g; portion size is not estimated from the photo (user grams, servings multiplier, or serving-size estimate).
- The server has **no authentication or rate limiting**. It is a development server; do not expose it publicly.
- `/api/chat/send` stores message text so a conversation can be displayed; `/api/chat/analyze` stores only derived labels, text length and a SHA-256 digest. Uploaded photos are deleted after processing unless `keep=true`.
- Diet, mood and allergen tags are rule-generated; they are guidance, not medical advice.

**Licences and credits:** Ultralytics YOLO11 (AGPL-3.0), scikit-learn, NetworkX, VADER sentiment, Anuvaad INDB 2024.11, Khana (Prabhu, arXiv:2509.06006), Food-101 (Bossard et al., 2014), Kaggle *Indian Food Images*. Full list and licence status: [`docs/DATASETS.md`](docs/DATASETS.md).