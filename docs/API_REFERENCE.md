# VYNDRA API reference (Parts 6–7)

Base URL during local development: `http://127.0.0.1:5001`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | service status |
| GET | `/api/health` | SQLite/application health |
| GET | `/api/privacy` | privacy/retention summary |
| GET | `/api/foods/search?q=apple` | search supplied nutrition table |
| GET | `/api/nutrition/<food_name>` | exact normalized nutrition lookup |
| POST | `/api/nutrition/calculate` | scale nutrition to grams |
| POST | `/api/meals` | validate nutrition items and persist a meal |
| GET | `/api/meals?limit=20` | list recent meal summaries |
| GET | `/api/meals/<id>` | one saved meal with items |
| POST | `/api/chat/analyze` | intent, food mentions, sentiment; raw text is not stored |
| GET | `/api/chat/topics` | bundled-data topic model |
| GET | `/api/social/graph` | simulated graph summary (Part 5 blueprint) |
| GET | `/api/social/clusters?n_clusters=3` | simulated food-interest clusters |
| GET | `/api/social/recommendations/u001?limit=5` | synthetic recommendations |
| GET | `/api/social/dietary-guidance/pizza` | general guidance lookup |
| GET | `/api/social/healthier-swaps/soda?limit=3` | curated swap suggestions |

## Example requests

```bash
curl -s http://127.0.0.1:5001/api/health
curl -s 'http://127.0.0.1:5001/api/foods/search?q=apple'
curl -s -X POST http://127.0.0.1:5001/api/nutrition/calculate \
  -H 'Content-Type: application/json' \
  -d '{"food_name":"apple","grams":150}'
curl -s -X POST http://127.0.0.1:5001/api/meals \
  -H 'Content-Type: application/json' \
  -d '{"notes":"sample meal","items":[{"food_name":"apple","quantity_g":150}]}'
curl -s -X POST http://127.0.0.1:5001/api/chat/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"I love dosa; suggest a healthier meal"}'
```

## Contract and safety notes

- Nutrition lookup uses exact normalized names; it does not invent nutrition values for unknown names. A meal with an unknown food returns HTTP 422.
- Input validation includes JSON content type, body-size, string length, meal item count, gram range, and list limits.
- SQL values use parameterized queries. Raw chat text is not persisted; derived labels, mentions, length and SHA-256 digest are stored. A digest is still linkable metadata, not anonymization.
- This development version does not implement user accounts, authentication, authorization, rate limiting, deletion/export workflows, or production deployment controls. Do not expose it to the public internet or store sensitive health information until these are implemented and reviewed.
- Images are not uploaded/retained by the new Part 6 API. Image classification remains the separate Part 3 utility and requires local model weights.
- Nutrition values are estimates from the provided dataset and are not medical advice.
