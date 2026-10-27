# VYNDRA — Results and Interpretation

Numbers below come from the files committed in `results/`. They can be regenerated with the steps in the README. Each section separates **observation** (what the data shows) from **inference** (a plausible explanation).

## 1. Text analytics (`results/tables/part4`, `results/figures/part4`)

**Classifier comparison** (`text_model_comparison.csv`, synthetic test split, 540 messages)

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| BoW / TF-IDF word with NB, LR, LinearSVC; BoW with Decision Tree; TF-IDF char with NB, LR, LinearSVC | 1.000 | 1.000 |
| TF-IDF word + Decision Tree | 0.994 | 0.994 |
| TF-IDF char + Decision Tree | 0.985 | 0.985 |

Selected: Bag-of-Words + Multinomial NB (validation accuracy; ties resolved by order). Nudge threshold: 0.90 (validation optimum 0.997, capped).

*Observation:* nearly every configuration is perfect. *Inference:* the messages come from a small set of templates, so the task is almost memorisation; this says little about real chats.

**Realistic check:** saved model on 150 hand-written messages (`review_set_intent_*`): accuracy **0.707**, macro-F1 **0.697**.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| food_intent | 0.722 | 0.520 | 0.605 |
| food_discussion | 0.694 | 0.680 | 0.687 |
| not_food | 0.708 | 0.920 | 0.800 |

*Observation:* `food_intent` has the lowest recall. *Inference:* real slang and phrasing are not covered by the templates, so intent messages are pushed to the other classes. Labels in this set are not yet human-reviewed.

> The model file must come from `run_text_analytics.py`. After `train_food_intent.py` the same test gives 0.453 accuracy (see README Step 2).

**Other text results**

| Experiment | Result | Caveat |
|---|---|---|
| Small baselines (`part4_summary.json`) | intent acc. 0.70 (30 messages), sentiment acc. 0.967 (30), mentions F1 1.00 (12), 4 topics on 20 documents | tiny sets; baseline only |
| Fuzzy mention matching on 150 review messages | precision 0.690, recall 0.808, F1 0.744 | `expected_food` labels unreviewed |
| VADER on 500 synthetic messages | macro-F1 0.494 | heuristic labels |
| LDA (`lda_topics.json`) | k = 10 selected (perplexity 297 → 251 for k = 4 → 10) | perplexity is in-sample; always favours larger k |
| Class mapping agreement (`matching_method_comparison.json`, 115 classes) | exact-normalised 1.00, fuzzy top-1 0.583 | agreement with the reviewer, not independent truth |

## 2. Social network analytics (`part5`)

60 users, 1,800 events, 4 communities of 15. **Modularity 0.502; ARI = NMI = 1.0** against the planted groups. Top users by PageRank (`social_centrality.csv`) include u030, u022, u051 (PageRank ≈ 0.021–0.023).

*Observation:* Louvain recovers the planted communities exactly. *Inference:* with 82% intra-community events the structure is easy to detect; it shows the method works, not that real friend groups would be this clear. High centrality here is structural prominence in a generated graph.

## 3. Food clustering (`part6`)

| k | Silhouette | Davies–Bouldin |
|---|---|---|
| **2** | **0.490** | 0.681 |
| 3 | 0.420 | 1.051 |
| 4 | 0.414 | 0.895 |
| 10 | 0.287 | 1.032 |

*Observation:* k = 2 maximises silhouette: one cluster of 885 foods (mean 176 kcal/100 g, 9.1 g fat) and one of 129. *Inference:* the macronutrient space has one dominant mass plus a high-energy/high-fat group; k = 2 is a heuristic choice, not a natural taxonomy.

## 4. Vision (`part7`, `results/tables/vision_*.csv`)

**Stage 1** — 270 Kaggle images, 9 classes × 30 (`vision_summary.csv`): top-1 **31.9%**, top-3 **54.1%**. The macro-F1 of 0.039 in that file is averaged over all 220 classes (most have no test images) and should **not** be quoted; use the supported-class metrics below.

**Stage 2** — 690 images, Kaggle (8 classes) + Khana (15 classes), 30 per class (`vision_evaluation_summary.json`):

| Metric | Value |
|---|---|
| Top-1 (accepted classes) | **30.4%** |
| Top-1 (primary label only) | 28.3% |
| Top-3 (accepted classes) | **50.1%** |
| Mean confidence when correct / wrong | 0.567 / 0.331 |
| Macro precision / recall / F1 (supported classes) | 0.720 / 0.272 / 0.360 |

| Dataset | Images | Top-1 | Top-3 |
|---|---|---|---|
| Kaggle Indian Food Images | 240 | 33.8% | 62.5% |
| Khana | 450 | 28.7% | 43.6% |

**Classical baselines** (HSV histograms, 15 Khana classes × 100 images, 25% test split): Gaussian NB **14.9%**, k-NN **20.0%**, Decision Tree **26.1%** top-1.

*Observation:* confidence is higher when the model is right, and precision (0.72) is much higher than recall (0.27). *Inference:* when the model commits to a supported class it is usually right, but it often predicts a visually similar sibling dish instead; the confidence threshold therefore carries real signal. The training validation score reported for `best.pt` (top-1 86.2%, top-5 98.3%, from training logs not stored in the repo) is far above the external results; the likely causes are domain shift, class-label differences and unknown training-data provenance.

*Comparison caveat:* the HSV baselines are scored on a different test split (25% of 1,500 Khana images) than YOLO (690 manifest images), so the numbers are indicative, not a controlled comparison.

Figures: `figures/vision_confusion_matrix.png` (stage 1), `figures/part7/vision_confusion_supported.png`, `vision_baseline_confusion.png`.

## 5. Data quality and portions (notebook outputs)

- 1,014 foods, no missing values or duplicates; 973 (95.96%) within 10% of the Atwater energy estimate.
- 907 of 1,014 serving estimates valid; 107 fall back to the flagged 100 g default.
- Portion accuracy against weighed meals has **not** been evaluated yet (template has only an example row).

## 6. Software quality

55 automated tests pass (1 optional slow test needs a photo); acceptance checks 1–5 pass.

## 7. What these results do and do not show

- They show a working end-to-end pipeline and an honest evaluation on independent images.
- They do **not** show real-world food-recognition accuracy, real chat performance or real social behaviour.
- Realistic improvements: retrain the dish model with documented, deduplicated training data; review and extend the hand-written chat set; evaluate portions on weighed meals; evaluate Food-101.