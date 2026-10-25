# Frontend/backend contract (Part 1)

This document records the JSON contract expected by the VYNDRA mobile app. The backend may return additional diagnostic fields, but the fields listed here are stable.

## Users

- `POST /api/users`: JSON `{ "name": "...", "diet": "Vegetarian", "allergens": [] }`; `id` or `user_id` may be supplied to keep a stable local demo identity. If omitted, the backend generates an ID.
- Response: `{ "id": string, "name": string, "diet": string|null, "allergens": string[] }`.
- `GET /api/users`: `{ "users": [user, ...] }`.
- Supported diet values: Vegetarian, Vegan, Keto, Pescatarian, Mediterranean, Paleo.
- Supported allergen values: `Lactose / Dairy`, `Eggs`, `Gluten`, `Nuts`, `Seafood / Shellfish`, `Soy`.

## Chat

- `POST /api/chat/send` JSON `{user_id, conversation_id, text}` returns `{message, nudge, nudge_text, food}`. `message` includes `id`, `user_id`, `conversation_id`, `text`, `created_at`, `intent_label`, and `probability`.
- `POST /api/chat/send-meal` multipart fields `image`, `user_id`, `conversation_id`, optional `text` returns the same message fields with `message.meal = {foods, summary, low_confidence}`.
- `GET /api/chat/<conversation_id>` returns `{messages: [...]}` oldest first. A message has a `meal` property only when a meal was attached.
- `/api/chat/send` stores message text because this is a simulated chat application. `/api/chat/analyze` stores only derived metadata and a SHA-256 digest. Do not expose the unauthenticated development server to the public internet.

## Meals and insights

- `POST /api/meals/analyze` and `POST /upload` accept multipart `image`, optional `user_id`, optional `grams` JSON object, and optional `servings` JSON number or object.
- Image extensions are JPG/JPEG, PNG, WEBP and BMP; maximum file size is 8 MiB. Pillow verifies the image bytes. Uploaded image bytes are deleted after processing unless `keep=true`.
- Unapproved or missing nutrition mappings never receive guessed nutrition values. Such foods return `needs_manual_selection: true` and food-search suggestions.
- `GET /api/meals?user_id=...&limit=...` returns `{meals: [...]}`. A supplied user ID filters the result.
- `GET /api/insights/<user_id>` returns `daily`, `weekly`, `allergen_warnings`, and `diet_fit` in the mobile-app shape.
- `GET /api/foods/<name>/insights` returns nutrition-tag insights and possible lower-kcal alternatives. The `cluster.description` explicitly states that food clustering is not yet implemented in Part 1.

## Social endpoints

- `GET /api/social/influencers` returns PageRank-sorted records with PageRank, betweenness, closeness and eigenvector centrality.
- `GET /api/social/communities` returns Louvain communities and modularity when supported by the installed NetworkX version (greedy modularity fallback otherwise).
- These results use the repository's bundled social example data in Part 1; large-scale temporal social analysis belongs to a later project part.

## Network access from a phone

`python app.py` binds to `0.0.0.0:5001` by default. On the same Wi-Fi network, use the Mac's LAN IP address (not `127.0.0.1`) in the mobile app, e.g. `http://192.168.x.x:5001`. macOS firewall rules may need to permit incoming connections. This development API has no authentication and should only be used on a trusted local network.
