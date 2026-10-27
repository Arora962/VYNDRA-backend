# VYNDRA Backend — API Reference

Base URL: `http://127.0.0.1:5001` (on the host) or `http://<LAN-IP>:5001` (from a phone on the same Wi-Fi).

This is a **development API with no authentication**. Do not expose it publicly.

## Conventions

| Topic | Rule |
|---|---|
| Request bodies | JSON routes need `Content-Type: application/json` and a JSON **object** (≤ 64 KB). Photo routes use `multipart/form-data`. |
| Errors | Always `{"error": "message"}` with a suitable status: `400` invalid input, `404` not found, `413` too large, `415` wrong content type or image extension, `422` unknown food name, `503` required data or model file missing, `500` unexpected error. |
| Limits | Image ≤ 8 MB (JPG, JPEG, PNG, WEBP, BMP); chat text ≤ 4,000 chars; `conversation_id` ≤ 150 chars; food name ≤ 150 chars; portion 1–5,000 g; ≤ 30 items per manual meal. |
| IDs | User IDs are strings. If `id` is omitted on user creation, a UUID is generated. |

## Health and privacy

| Method | Path | Response |
|---|---|---|
| GET | `/api/health` | `{"status":"ok","service":"vyndra-backend","database":"sqlite"}` |
| GET | `/api/privacy` | Storage policy: `raw_chat_text_stored`, `message_text_stored_for_simulated_chat`, `image_retention`, limits |

## Users

| Method | Path | Notes |
|---|---|---|
| POST | `/api/users` | Create or update a user |
| GET | `/api/users` | `{"users":[{id,name,diet,allergens}, ...]}` |

```json
// POST /api/users
{ "id": "u1", "name": "Asha", "diet": "Vegetarian", "allergens": ["Gluten"] }
// 201
{ "id": "u1", "name": "Asha", "diet": "Vegetarian", "allergens": ["Gluten"] }
```

Allowed `diet` values: `Vegetarian, Vegan, Keto, Pescatarian, Mediterranean, Paleo`. Allowed `allergens`: `Lactose / Dairy, Eggs, Gluten, Nuts, Seafood / Shellfish, Soy`. `id` or `user_id` may be supplied for a stable identity; the legacy inputs `display_name` and `dietary_preferences` are also accepted. `diet` is returned as a string only when exactly one diet is stored, otherwise `null`.

## Nutrition and meals

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/foods/search?q=idli` | Case-insensitive substring search, up to 20 names → `{query,count,foods}` |
| GET | `/api/nutrition/<food_name>` | Exact normalised-name lookup (per 100 g) |
| POST | `/api/nutrition/calculate` | `{"food_name":"idli","grams":150}` → values scaled to the portion |
| POST | `/api/meals` | Save a manual meal: `{"items":[{"food_name","quantity_g"}],"notes","user_id"}` (1–30 items, exact food names, otherwise `422`) |
| GET | `/api/meals?limit=20&user_id=u1` | Recent meals (`limit` 1–100); `user_id` filters |
| GET | `/api/meals/<meal_id>` | One saved meal with its items |

```json
// POST /api/nutrition/calculate → 200 (abridged)
{ "food_name": "Idli", "portion_g": 150.0, "energy_kcal": 206.31, "protein_g": 6.96,
  "carb_g": 42.27, "fat_g": 0.49, "fibre_g": 3.46, "diet_tags": "Vegan, Vegetarian",
  "best_time_to_eat": "Breakfast", "match_type": "exact_normalized_name" }
```

## Photo analysis

`POST /api/meals/analyze` (legacy alias: `POST /upload`) — multipart form:

| Field | Required | Description |
|---|---|---|
| `image` | yes | JPG, JPEG, PNG, WEBP or BMP, ≤ 8 MB; verified with Pillow |
| `user_id` | no | Must exist (create it with `POST /api/users` first) |
| `grams` | no | JSON object keyed by predicted class or mapped food name, e.g. `{"idly": 180}` |
| `servings` | no | A number or a JSON object keyed like `grams`; multiplies the serving estimate (0 < x ≤ 20) |
| `keep` | no | `true` keeps the photo in `uploads/`; by default it is deleted after processing |

Behaviour:
- The model returns its **top 3** predictions; predictions below confidence **0.10** are dropped.
- `low_confidence` is `true` below **0.59** (a provisional threshold, see `docs/METHODOLOGY.md`).
- Nutrition is returned **only** for reviewed `exact`/`family` mappings. Otherwise the item has `needs_manual_selection: true`, `found: false` and up to five name `suggestions`; nutrition is never guessed.
- Portion priority: `grams` › `servings × serving estimate` › serving estimate › flagged 100 g default. `portion_source` and `default_portion` tell the client which was used.
- The meal is saved to SQLite and `meal_id` is returned.

```json
// 200 (abridged)
{ "message": "Image analysed. Review predictions and nutrition matches before relying on them.",
  "foods": [ { "food": "Idli", "class_name": "idly", "quantity_grams": 120.0, "confidence": 0.81,
               "match_type": "exact", "reviewed": true, "low_confidence": false, "found": true,
               "needs_manual_selection": false, "suggestions": [], "portion_source": "serving_estimate",
               "default_portion": false, "calories": 137.5, "protein": 4.6, "carbs": 28.2, "fat": 0.3, "fiber": 2.3 } ],
  "summary": { "energy": 137.5, "calories": 137.5, "protein": 4.6, "carbs": 28.2, "fat": 0.3, "fiber": 2.3 },
  "meal_id": 7, "model": "models/best.pt", "low_confidence_threshold": 0.59, "image_retained": false }
```
*(Illustrative values; field names are exact.)*

## Chat

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat/analyze` | Non-persistent analysis of one text (stores only derived metadata) |
| GET | `/api/chat/topics` | LDA topics on the bundled 20-document demo set |
| POST | `/api/chat/send` | Store a message, classify intent, return a nudge decision |
| POST | `/api/chat/send-meal` | Multipart: `image`, `user_id`, `conversation_id`, optional `text`; analyses the photo and attaches the meal |
| GET | `/api/chat/<conversation_id>` | `{"messages":[...]}` oldest first; messages with a meal carry a nested `meal` |

```json
// POST /api/chat/send   {"user_id":"u1","conversation_id":"lounge","text":"I am craving masala dosa for dinner"}
// 201
{ "message": { "id": 1, "user_id": "u1", "conversation_id": "lounge",
               "text": "I am craving masala dosa for dinner", "created_at": "2026-10-27T01:13:29.738Z",
               "intent_label": "food_intent", "probability": 0.9999998 },
  "nudge": true, "nudge_text": "You mentioned masala dosa. Want to share a photo of your meal?",
  "food": "masala dosa" }
```

`nudge` is `true` when the intent is `food_intent` or `food_discussion`, the probability is ≥ the calibrated threshold (`results/tables/part4/nudge_threshold.json`, currently **0.90**, fallback 0.45) **and** a food phrase was found. `intent_label` is one of `food_intent`, `food_discussion`, `not_food`.

`/api/chat/send-meal` returns the same fields plus `message.meal = {foods, summary:{calories,protein,carbs,fat,fiber}, low_confidence}` and `meal_id`.

```json
// POST /api/chat/analyze → 200 (abridged)
{ "food_intent": { "label": "food_intent", "confidence": 0.99999, "is_food_related": true,
                   "class_probabilities": { "food_intent": 0.99999, "food_discussion": 1.5e-07, "not_food": 3e-09 } },
  "food_mentions": ["masala dosa"],
  "sentiment": { "label": "neutral", "compound": 0.0, "scores": { "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0 } },
  "privacy": { "raw_text_persisted": false, "stored_fields": ["derived labels", "food mentions", "text length", "SHA-256 digest"] } }
```

## Insights

`GET /api/insights/<user_id>` → `404` if the user does not exist.

```json
{ "daily": [ { "date": "2026-10-27", "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0 } ],
  "weekly": { "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "meals": 0, "avg_daily_calories": 0 },
  "allergen_warnings": [ { "food": "Idli", "allergens": ["Gluten"] } ],
  "diet_fit": { "score": 0, "label": "Not assessed", "notes": ["..."] },
  "note": "Nutrition and tag-based guidance are estimates; tags are rule-generated and are not medical advice." }
```

The window is the last 7 days (UTC). `diet_fit.score` is the percentage of logged foods carrying at least one of the user's diet tags; the label is `Good fit` (≥ 80), `Mixed fit` (≥ 50), `Review foods`, or `Not assessed`.

## Food insights

`GET /api/foods/<food_name>/insights[?diet=Vegan]` → diet tags, allergen tags, best time to eat, nutrition cluster (after `scripts/cluster_foods.py`), and lower-calorie `swaps` with `kcal_saving_pct`. With `?diet=` a `diet_fit.scores` object is added. Tags are rule-generated, and a score of 0 means "not tagged", not "unsuitable". `404` if the food is not in the nutrition table.

## Social

| Method | Path | Returns |
|---|---|---|
| GET | `/api/social/influencers` | `{"influencers":[{user_id,name,pagerank,betweenness,closeness,eigenvector,degree}]}` sorted by PageRank |
| GET | `/api/social/communities` | `{"modularity":0.5,"communities":[{"id":0,"members":["u001",...]}]}` (Louvain, seed 42). IDs are run-specific labels. |
| GET | `/api/social/user/<id>` | Centrality and community details for one user (`404` if unknown); uses `results/tables/part5/social_centrality.csv` when present |
| GET | `/api/social/graph` | Graph metrics for the bundled graph |
| GET | `/api/social/clusters?n_clusters=3` | Food-interest clusters of users |
| GET | `/api/social/recommendations/<user_id>?limit=5` | `{"user_id","friends":[...],"foods":[...]}` (`limit` 0–50) |
| GET | `/api/social/dietary-guidance/<food>` | Curated general guidance |
| GET | `/api/social/healthier-swaps/<food>?limit=3` | Curated swap ideas (`limit` 0–20) |

The social routes read `data/processed/social_recommendations/social_edges.csv` and `user_food_profiles.csv`. The repository ships a **20-user demo graph**; running `scripts/make_social_data.py` replaces it with the **60-user synthetic network**. All social data is synthetic.

## Mobile-app contract notes

- Extra diagnostic fields may be added to responses; the fields documented here are stable.
- Use the server's LAN IP from a phone. The app should call `POST /api/users` before any route that needs a `user_id`.
- Message bubbles with meals read `message.meal.foods` and `message.meal.summary`.

## Privacy and data handling

| Endpoint | What is stored |
|---|---|
| `/api/chat/send`, `/api/chat/send-meal` | Message text, intent label, probability, nudge flag, meal reference (needed to display conversations) |
| `/api/chat/analyze` | Only intent, sentiment, food mentions, text length and a SHA-256 digest (a digest is **not** anonymisation) |
| Photo routes | Meal totals and items; the image is deleted unless `keep=true` |

SQLite content is user data: protect the file. Nutrition is an estimate, model predictions can be wrong, and tags depend on the source dataset.