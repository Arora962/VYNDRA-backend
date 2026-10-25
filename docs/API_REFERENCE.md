# VYNDRA Backend API Reference (Part 1)

Base URL: `http://127.0.0.1:5001` on the Mac itself. The server binds to `0.0.0.0:5001` so a phone on the same trusted Wi-Fi can reach it through the Mac LAN IP. JSON routes use `Content-Type: application/json`; photo routes use multipart form data. The API is a local development/demo backend and has no authentication. Do not expose it publicly.

## Health and privacy
- `GET /api/health` — database/backend status.
- `GET /api/privacy` — current data-storage notes.

## Users and insights
- `POST /api/users` — JSON `{ "name": "Kriti", "diet": "Vegetarian", "allergens": [] }`. `id` is optional; if omitted, the backend generates one. Response: `{id,name,diet,allergens}`. The legacy `display_name` and `dietary_preferences` inputs are also accepted.
- `GET /api/users` — list demo users.
- `GET /api/insights/<user_id>` — `{daily:[...], weekly:{...}, allergen_warnings:[...], diet_fit:{score,label,notes}}` for the frontend.

## Photo analysis
- `POST /api/meals/analyze` — multipart field `image` (JPG, JPEG, PNG, WEBP, BMP; maximum 8 MB). Optional fields: `user_id`; `grams` as a JSON object keyed by predicted class or mapped food name; `servings` as a number or JSON object keyed the same way; `keep=true` to retain the temporary image in `uploads/`.
- `POST /upload` — legacy alias for the same photo analysis flow.
- Predictions use top three model outputs at confidence >= 0.10. `low_confidence` is true below 0.59. This provisional threshold maximized F1 for top-1 correctness on the repository's existing 270-image result file; the mapping/test set is limited, so recalibrate after expanded vision evaluation.
- Nutrition is returned only for reviewed `exact`/`family` mapping entries that resolve to a nutrition row. `none` or unresolved mappings return `needs_manual_selection` and up to five search suggestions without fabricated nutrient values.
- Images are deleted after processing unless `keep=true`.

## Meals and nutrition
- `POST /api/meals` — create a manually logged meal from exact nutrition-table food names.
- `GET /api/meals?limit=20` — list recent meals; optional `user_id` filters by user.
- `GET /api/meals/<meal_id>` — retrieve a saved meal.
- `GET /api/foods/search?q=dosa` — search food names.
- `GET /api/nutrition/<food_name>` — exact normalized-name lookup.
- `POST /api/nutrition/calculate` — JSON `{ "food_name": "Hot tea (Garam Chai)", "grams": 150 }`.

## Chat
- `POST /api/chat/analyze` — non-persistent text analysis; raw text is not stored by this endpoint.
- `GET /api/chat/topics` — topic analysis of the bundled demonstration documents.
- `POST /api/chat/send` — JSON `{ "user_id": "u1", "conversation_id": "c1", "text": "I want biryani" }`; returns `{message,nudge,nudge_text,food}`. The message includes `id`, `user_id`, `conversation_id`, `text`, `created_at`, `intent_label`, and `probability`.
- `POST /api/chat/send-meal` — multipart `image`, `user_id`, `conversation_id`, optional `text`; returns a message with nested `meal:{foods,summary,low_confidence}`.
- `GET /api/chat/<conversation_id>` — `{messages:[...]}` oldest first; a saved meal is nested on the related message.

## Social routes
- `GET /api/social/influencers` — PageRank-sorted centrality records.
- `GET /api/social/communities` — communities and modularity.

## Food insights
- `GET /api/foods/<food_name>/insights` — diet tags, allergen tags, best time, and lower-kcal alternatives. The cluster description notes that food clustering is a later project part.

## Existing social routes
- `GET /api/social/graph`
- `GET /api/social/clusters?n_clusters=3`
- `GET /api/social/recommendations/<user_id>`
- `GET /api/social/dietary-guidance/<food>`
- `GET /api/social/healthier-swaps/<food>`

## Privacy and limitations
For this simulated app, `/api/chat/send` stores message text in local SQLite so a conversation can be displayed. The older `/api/chat/analyze` endpoint stores only derived labels, mention metadata, text length, and a SHA-256 digest. Meal photos are deleted by default. This demo has no authentication or account isolation; SQLite content is sensitive user data. Nutrition is an estimate, model predictions can be wrong, and tags depend on the source dataset.
