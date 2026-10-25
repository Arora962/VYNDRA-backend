# VYNDRA dataset inventory

## Nutrition and serving sizes

- `data/raw/Db.csv`: supplied nutrition table; values are treated as per 100 g.
- `data/raw/Anuvaad_INDB_2024.11.xlsx`: serving units and per-serving energy values. Serving grams are estimated as `unit_serving_energy_kcal / energy_kcal * 100`. Rows with missing serving units, non-positive energy, missing calculations, or estimates outside 5–1,000 g are excluded. If no usable serving estimate is available, the API uses a documented 100 g default and marks `default_portion: true`.
- `data/raw/Selective_Diet_Mapping.csv`: diet, mood, allergen, and timing tags are rule-generated from templates; they are not clinical guidance.

## Dish classification and matching

- `models/best.pt`: trained classifier weights supplied with this repository.
- `data/processed/classes.txt`: model class names.
- `data/processed/class_to_food.csv`: reviewed mapping used by the API. Do not automatically promote entries from `class_to_food_review.csv`; those are suggestions only. A `none` mapping must never produce guessed nutrition.
- Existing `results/tables/vision_predictions.csv` is a previous evaluation result and not a new independent test.

## Chat examples

`data/processed/chat_intelligence/food_intent_examples.csv` is the existing labelled demonstration set. It is small and should not be presented as a production-grade model evaluation. Run `python scripts/train_food_intent.py` to save the model and its training report. The `/api/chat/send` endpoint stores message text because this is a simulated chat application; `/api/chat/analyze` continues to store only derived metadata and a one-way digest.

## Evaluation portion template

`data/processed/reference_meals_template.csv` is a template, not a labelled dataset. Replace its example row with weighed reference meals before interpreting `scripts/evaluate_portions.py` results.
