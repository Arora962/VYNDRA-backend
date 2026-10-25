# %% [markdown]
# # VYNDRA Analytics Notebook
# This notebook-style Python file is intentionally cache-driven: it reads saved CSV/JSON results
# and does not scan large image datasets. Run the analysis scripts first, then execute this file
# cell-by-cell in VS Code/Jupyter or run `python notebooks/VYNDRA_analytics.py`.
#
# ## Observation vs inference
# Each figure has a title and labelled axes. Printed interpretation distinguishes what the plot
# directly shows (observation) from plausible explanations (inference).

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "data/raw/Db.csv").exists():
    ROOT = Path.cwd().parent
RESULTS = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Nutrition-table quality and preprocessing
# The nutrition table stores values per 100 g. This section calculates missingness, duplicates,
# outliers, kJ/kcal ratios and an Atwater-style energy consistency check from the current data.
# These checks are descriptive and do not establish that source values are medically accurate.

# %%
foods = pd.read_csv(ROOT / "data/raw/Db.csv")
print("Nutrition rows:", len(foods))
print("Columns:", list(foods.columns))
print("Missing values by column:\n", foods.isna().sum().to_string())
print("Exact duplicate rows:", int(foods.duplicated().sum()))
print("Duplicate food names:", int(foods.food_name.duplicated().sum()))
macro_cols = [c for c in ["energy_kcal", "protein_g", "carb_g", "fat_g", "fibre_g"] if c in foods]
print("Macro summary:\n", foods[macro_cols].describe(percentiles=[.01,.25,.5,.75,.99]).round(2).to_string())
for column in macro_cols:
    values = pd.to_numeric(foods[column], errors="coerce").dropna()
    q1, q3 = values.quantile(.25), values.quantile(.75)
    iqr = q3 - q1
    outlier_mask = (pd.to_numeric(foods[column], errors="coerce") < q1 - 1.5*iqr) | (pd.to_numeric(foods[column], errors="coerce") > q3 + 1.5*iqr)
    print(f"IQR outliers for {column}:", int(outlier_mask.sum()))

# %%
if {"energy_kj", "energy_kcal"}.issubset(foods.columns):
    valid = foods[(foods.energy_kcal > 0) & (foods.energy_kj > 0)].copy()
    valid["kj_per_kcal"] = valid.energy_kj / valid.energy_kcal
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid.kj_per_kcal, bins=30)
    ax.set_title("Nutrition table: kJ per kcal ratio")
    ax.set_xlabel("energy_kj / energy_kcal")
    ax.set_ylabel("Food record count")
    fig.tight_layout(); fig.savefig(FIGURES / "nutrition_kj_kcal_ratio.png", dpi=160); plt.show()
    print("Observation: ratio range =", round(valid.kj_per_kcal.min(), 3), "to", round(valid.kj_per_kcal.max(), 3))
    print("Inference: extreme ratios can indicate source or conversion inconsistencies; inspect individual rows before changing values.")

# %%
if {"energy_kcal", "protein_g", "carb_g", "fat_g"}.issubset(foods.columns):
    check = foods.copy()
    check["atwater_kcal"] = 4*check.protein_g + 4*check.carb_g + 9*check.fat_g
    check["relative_difference"] = (check.energy_kcal-check.atwater_kcal).abs()/check.energy_kcal.replace(0,np.nan)
    valid = check.relative_difference.dropna()
    print("Atwater records within 10%:", int((valid <= .10).sum()), "of", len(valid), "(", round((valid <= .10).mean()*100, 2), "%)")
    print("Largest Atwater differences:\n", check.nlargest(10,"relative_difference")[["food_name","energy_kcal","atwater_kcal","relative_difference"]].to_string(index=False))
    fig, ax = plt.subplots(figsize=(8,5)); ax.hist(valid, bins=40)
    ax.set_title("Atwater energy consistency check")
    ax.set_xlabel("Absolute relative difference from 4/4/9 kcal estimate")
    ax.set_ylabel("Food record count"); fig.tight_layout(); fig.savefig(FIGURES / "nutrition_atwater_check.png", dpi=160); plt.show()
    print("Observation: the histogram and calculated proportion show agreement with the simplified 4/4/9 rule.")
    print("Inference: fibre, food-specific energy factors, rounding, and source inconsistencies can explain deviations; the check is not a correction rule.")

# %% [markdown]
# ## 2. Portion-size cleaning
# The cleaned serving estimate is unit_serving_energy_kcal / energy_kcal * 100. Missing or
# out-of-range serving estimates are handled by the documented 100 g fallback in portions.py.

# %%
serving_file = ROOT / "data/raw/Anuvaad_INDB_2024.11.xlsx"
if serving_file.exists():
    serving = pd.read_excel(serving_file)
    needed = {"food_name", "energy_kcal", "unit_serving_energy_kcal", "servings_unit"}
    if needed.issubset(serving.columns):
        serving["energy_kcal"] = pd.to_numeric(serving["energy_kcal"], errors="coerce")
        serving["unit_serving_energy_kcal"] = pd.to_numeric(serving["unit_serving_energy_kcal"], errors="coerce")
        serving["estimated_grams"] = serving["unit_serving_energy_kcal"] / serving["energy_kcal"] * 100
        serving["valid_serving"] = serving["servings_unit"].fillna("").astype(str).str.strip().ne("") & serving["estimated_grams"].between(5, 1000) & serving["estimated_grams"].notna()
        print("Serving records:", len(serving), "valid estimates:", int(serving.valid_serving.sum()), "missing/out-of-range/default candidates:", int((~serving.valid_serving).sum()))
        fig, ax = plt.subplots(figsize=(8,5)); ax.hist(serving.loc[serving.valid_serving, "estimated_grams"], bins=35)
        ax.set_title("Cleaned Anuvaad serving-size estimates")
        ax.set_xlabel("Estimated grams per serving")
        ax.set_ylabel("Food record count"); fig.tight_layout(); fig.savefig(FIGURES / "portion_serving_size_cleaning.png", dpi=160); plt.show()
        print("Observation: valid serving estimates are limited to records with a serving unit and a 5–1,000 g estimate.")
        print("Inference: out-of-range estimates are treated as unreliable and use the documented 100 g fallback in the API.")
    else:
        print("Serving workbook is missing required columns:", sorted(needed - set(serving.columns)))
portion_results = RESULTS / "portion_evaluation.csv"
if portion_results.exists():
    print(pd.read_csv(portion_results).describe(include="all").to_string())
else:
    template = ROOT / "data/processed/reference_meals_template.csv"
    print("Reference-meal evaluation is pending. Fill", template, "then run scripts/evaluate_portions.py.")

# %% [markdown]
# ## 3. Text analytics
# Results are loaded from the saved experiment outputs; the notebook does not retrain models.

# %%
text_path = RESULTS / "part4/text_model_comparison.csv"
if text_path.exists():
    text_results = pd.read_csv(text_path).sort_values("macro_f1", ascending=False)
    print(text_results.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(text_results.model + "\n" + text_results.vectorizer, text_results.macro_f1)
    ax.set_title("Text classifier comparison on held-out synthetic test data")
    ax.set_xlabel("Classifier and vectorizer")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", labelrotation=45); fig.tight_layout(); fig.savefig(FIGURES / "text_model_comparison.png", dpi=160); plt.show()
    print("Observation: bars compare held-out metrics on synthetic messages.")
    print("Inference: synthetic/template data can overstate performance; use the separately reviewed hand-written test set before choosing a production threshold.")
else: print("Text results not found; run scripts/make_chat_dataset.py and scripts/run_text_analytics.py.")

# %% [markdown]
# ## 4. Social network analysis

# %%
social_path = RESULTS / "part5/social_centrality.csv"
if social_path.exists():
    social = pd.read_csv(social_path)
    print(social.head(10).to_string(index=False))
    fig, ax = plt.subplots(figsize=(8,5)); ax.scatter(social.pagerank, social.betweenness, c=social.community, s=40+social.pagerank*3000)
    ax.set_title("Social influence: PageRank vs betweenness centrality")
    ax.set_xlabel("PageRank"); ax.set_ylabel("Betweenness centrality"); fig.tight_layout(); fig.savefig(FIGURES / "social_centrality_comparison.png", dpi=160); plt.show()
    print("Observation: each point is a synthetic user; point size is scaled by PageRank and colour denotes detected community.")
    print("Inference: high centrality indicates structural prominence in this generated graph, not real-world influence.")
else: print("Social results not found; run scripts/make_social_data.py and scripts/run_social_analytics.py.")

# %% [markdown]
# ## 5. Nutrition clustering

# %%
cluster_path = RESULTS / "part6/food_kmeans_metrics.csv"
if cluster_path.exists():
    cluster_metrics = pd.read_csv(cluster_path)
    fig, ax = plt.subplots(figsize=(8,5)); ax.plot(cluster_metrics.k, cluster_metrics.silhouette, marker="o")
    ax.set_title("K-Means silhouette score by number of food clusters")
    ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Silhouette score"); fig.tight_layout(); fig.savefig(FIGURES / "food_silhouette_by_k.png", dpi=160); plt.show()
    print(cluster_metrics.to_string(index=False))
    print("Observation: silhouette measures separation and cohesion under the selected scaled macro features.")
    print("Inference: the maximum score is a useful selection heuristic, not proof that the groups are natural food categories.")
else: print("Food clustering results not found; run scripts/cluster_foods.py.")

# %% [markdown]
# ## 6. Vision evaluation

# %%
vision_path = RESULTS / "part7/vision_evaluation_summary.json"
if vision_path.exists():
    print(json.dumps(json.loads(vision_path.read_text()), indent=2))
else: print("Vision results not found; prepare the manifest and run scripts/evaluate_vision_suite.py.")

# %% [markdown]
# ## Limitations
# - Synthetic chat and social data are useful for reproducible method demonstrations, not evidence of real-user behaviour.
# - Food diet/mood/allergen tags are rule-generated from templates and need independent review.
# - The class-to-food mapping is a human-reviewed judgment; unmatched foods must not receive guessed nutrition.
# - The reported model validation score is not an independent test. External performance may be lower due to domain shift, mapping errors, and data leakage/unknown training provenance.
# - Nutrition values and serving sizes are estimates; the simplified Atwater check is diagnostic only.
# - This notebook reads cached results and intentionally does not rescan the large image datasets.
