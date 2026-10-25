# VYNDRA dataset inventory and provenance

| Dataset | Purpose | Provenance / limitation |
|---|---|---|
| `data/raw/Db.csv` | Nutrition values per 100 g; source table for nutrition analysis and clustering | Supplied project data. Check missing values, duplicates, outliers, energy units and Atwater consistency before interpreting results. |
| `data/raw/Anuvaad_INDB_2024.11.xlsx` | Serving-size estimates | Serving grams are inferred as `unit_serving_energy_kcal / energy_kcal * 100`; missing or out-of-range estimates use the documented 100 g fallback. |
| `data/raw/Selective_Diet_Mapping.csv` | Diet, mood, allergen and meal-time tags | Rule-generated from sentence templates; not independent clinical or dietary ground truth. |
| `data/processed/class_to_food.csv` | Reviewed model-class-to-nutrition mapping | Human-reviewed project mapping. Preserve its contents. `none` means no approved equivalent; never guess nutrition. |
| `data/processed/chat_intelligence/chat_messages_synthetic.csv` | Intent classifier experiments | Deterministic synthetic data generated with seed 42 by `scripts/make_chat_dataset.py`. Template performance is optimistic. |
| `data/processed/chat_intelligence/chat_messages_handwritten_review.csv` | Separate realistic chat review set | Contains individually authored examples and needs human label review before use as an independent test. |
| `data/processed/social_scale/social_events.csv` | Social network analysis | Synthetic timestamped directed events with planted communities; demonstrates methods, not real engagement. |
| `data/processed/vision_test_manifest.csv` | Reproducible image test manifest | Relative paths for eligible mapped image files; external dataset roots are configured locally. Hidden `._*` files are excluded. |
| `data/test_images_own/<class>/` | Separate own-photo vision evaluation | User-supplied photos grouped by approved model class; results must be reported separately from public datasets. |

## Image datasets

The original project instructions say the Khana dataset, Food-101 cache and Kaggle Indian Food Images are stored outside the repository. This patch does not bundle or redistribute those images. Use `scripts/prepare_vision_manifest.py` with the actual local roots. Khana and Food-101 require reviewed class maps before their images can be counted as valid evaluation examples. Only rows mapped to an approved `best.pt` class with `exact` or `family` match type are eligible. Mapping suggestions are not ground truth.
