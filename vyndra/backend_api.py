"""Flask API for nutrition lookup, meal logging, users, and insights."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from flask import Blueprint, jsonify, request

from .database import (
    create_user,
    get_meal,
    get_user,
    init_db,
    list_meals,
    list_users,
    save_meal,
    user_meal_rows,
)
from .nutrition import calculate_nutrition, find_food, get_food_table

bp = Blueprint("vyndra_api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)
MAX_JSON_BYTES = 64 * 1024


def _json_object():
    """Validate a bounded JSON object request body."""
    if request.content_length is not None and request.content_length > MAX_JSON_BYTES:
        return None, (jsonify({"error": "Request body too large; maximum is 64 KB."}), 413)
    if not request.is_json:
        return None, (jsonify({"error": "Content-Type must be application/json."}), 415)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, (jsonify({"error": "Body must be a JSON object."}), 400)
    return payload, None


@bp.get("/health")
def health():
    """Report backend and database availability."""
    try:
        init_db()
        return jsonify({"status": "ok", "service": "vyndra-backend", "database": "sqlite"})
    except Exception:
        logger.exception("Database health check failed")
        return jsonify({"status": "error", "error": "Database initialization failed."}), 500


@bp.get("/foods/search")
def foods_search():
    """Search food names using a literal case-insensitive substring."""
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Query parameter q is required."}), 400
    if len(query) > 100:
        return jsonify({"error": "q must be 100 characters or fewer."}), 400
    try:
        frame = get_food_table()
        hits = frame[frame["food_name"].str.contains(query, case=False, regex=False, na=False)].head(20)
        return jsonify({"query": query, "count": len(hits), "foods": hits["food_name"].astype(str).tolist()})
    except Exception:
        logger.exception("Food search failed")
        return jsonify({"error": "Nutrition database is unavailable. Check data/raw/Db.csv."}), 503


@bp.get("/nutrition/<path:food_name>")
def nutrition_lookup(food_name):
    """Return an exact normalized-name nutrition lookup."""
    if not food_name.strip() or len(food_name) > 150:
        return jsonify({"error": "food_name must be 1–150 characters."}), 400
    try:
        return jsonify(find_food(food_name))
    except FileNotFoundError:
        return jsonify({"error": "Nutrition database file is missing."}), 503
    except Exception:
        logger.exception("Nutrition lookup failed")
        return jsonify({"error": "Nutrition lookup failed."}), 500


@bp.post("/nutrition/calculate")
def nutrition_calculate():
    """Scale a food's per-100-gram values to a requested portion."""
    payload, error = _json_object()
    if error:
        return error
    name = payload.get("food_name")
    grams = payload.get("grams")
    if not isinstance(name, str) or not name.strip() or len(name) > 150:
        return jsonify({"error": "food_name must be a non-empty string up to 150 characters."}), 400
    if isinstance(grams, bool) or not isinstance(grams, (int, float)) or not 0 < grams <= 5000:
        return jsonify({"error": "grams must be a number greater than 0 and at most 5000."}), 400
    try:
        return jsonify(calculate_nutrition(name, float(grams)))
    except FileNotFoundError:
        return jsonify({"error": "Nutrition database file is missing."}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Nutrition calculation failed")
        return jsonify({"error": "Nutrition calculation failed."}), 500


@bp.post("/meals")
def create_meal_endpoint():
    """Create a meal from manually entered exact nutrition-table food names."""
    payload, error = _json_object()
    if error:
        return error
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 30:
        return jsonify({"error": "items must be a list containing 1–30 foods."}), 400
    items = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            return jsonify({"error": f"items[{index}] must be an object."}), 400
        name = item.get("food_name")
        grams = item.get("quantity_g", 100)
        if not isinstance(name, str) or not name.strip() or len(name) > 150:
            return jsonify({"error": f"items[{index}].food_name must be a non-empty string up to 150 characters."}), 400
        if isinstance(grams, bool) or not isinstance(grams, (int, float)) or not 0 < grams <= 5000:
            return jsonify({"error": f"items[{index}].quantity_g must be > 0 and <= 5000."}), 400
        try:
            nutrition = calculate_nutrition(name, float(grams))
        except FileNotFoundError:
            return jsonify({"error": "Nutrition database file is missing."}), 503
        if not nutrition.get("found"):
            return jsonify({"error": f"No exact nutrition match for {name!r}; correct the name rather than guessing nutrition values."}), 422
        items.append({
            "food_name": nutrition["food_name"], "quantity_g": float(grams),
            "calories": nutrition["energy_kcal"], "protein_g": nutrition["protein_g"],
            "carbs_g": nutrition["carb_g"], "fat_g": nutrition["fat_g"],
            "fiber_g": nutrition["fibre_g"], "match_type": "exact_normalized_name",
        })
    notes = payload.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 500:
        return jsonify({"error": "notes must be a string of at most 500 characters."}), 400
    user_id = payload.get("user_id")
    if user_id is not None and (not isinstance(user_id, str) or not get_user(user_id)):
        return jsonify({"error": "user_id must identify an existing user."}), 400
    try:
        return jsonify(save_meal(items, source="manual", notes=notes, user_id=user_id)), 201
    except Exception:
        logger.exception("Could not save manually logged meal")
        return jsonify({"error": "Could not save meal to SQLite."}), 500


@bp.get("/meals")
def meals_list():
    """List recent meals, optionally filtered to one user."""
    raw_limit = request.args.get("limit", "20")
    try:
        limit = int(raw_limit)
    except ValueError:
        return jsonify({"error": "limit must be an integer between 1 and 100."}), 400
    if not 1 <= limit <= 100:
        return jsonify({"error": "limit must be between 1 and 100."}), 400
    user_id = request.args.get("user_id")
    try:
        return jsonify({"meals": list_meals(limit, user_id=user_id)})
    except Exception:
        logger.exception("Could not read meals")
        return jsonify({"error": "Could not read meals from SQLite."}), 500


@bp.get("/meals/<int:meal_id>")
def meal_detail(meal_id):
    """Return one saved meal and its food items."""
    try:
        meal = get_meal(meal_id)
    except Exception:
        logger.exception("Could not read meal")
        return jsonify({"error": "Could not read meal from SQLite."}), 500
    return (jsonify(meal), 200) if meal else (jsonify({"error": "Meal not found."}), 404)


@bp.post("/users")
def create_user_endpoint():
    """Create or update a simulated-app user using the frontend contract."""
    payload, error = _json_object()
    if error:
        return error
    raw_id = payload.get("id") or payload.get("user_id")
    user_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else str(uuid4())
    name = payload.get("name", payload.get("display_name", ""))
    diet = payload.get("diet", payload.get("dietary_preferences", []))
    allergens = payload.get("allergens", [])
    if not isinstance(name, str) or not name.strip() or len(name) > 150:
        return jsonify({"error": "name must be a non-empty string up to 150 characters."}), 400
    if isinstance(diet, str):
        diet = [diet] if diet.strip() else []
    if not isinstance(diet, list) or not all(isinstance(value, str) for value in diet):
        return jsonify({"error": "diet must be a string or list of strings."}), 400
    if not isinstance(allergens, list) or not all(isinstance(value, str) for value in allergens):
        return jsonify({"error": "allergens must be a list of strings."}), 400
    allowed_diets = {"Vegetarian", "Vegan", "Keto", "Pescatarian", "Mediterranean", "Paleo"}
    allowed_allergens = {"Lactose / Dairy", "Eggs", "Gluten", "Nuts", "Seafood / Shellfish", "Soy"}
    if any(value not in allowed_diets for value in diet):
        return jsonify({"error": "Unsupported diet value.", "allowed": sorted(allowed_diets)}), 400
    if any(value not in allowed_allergens for value in allergens):
        return jsonify({"error": "Unsupported allergen value.", "allowed": sorted(allowed_allergens)}), 400
    try:
        stored = create_user(user_id, name.strip(), diet, allergens)
        return jsonify({"id": stored["id"], "name": stored["display_name"],
                        "diet": (stored["dietary_preferences"][0] if len(stored["dietary_preferences"]) == 1 else None),
                        "allergens": stored["allergens"]}), 201
    except Exception:
        logger.exception("Could not create user")
        return jsonify({"error": "Could not save user."}), 500


@bp.get("/users")
def list_users_endpoint():
    """List simulated-app users."""
    try:
        return jsonify({"users": [{"id": u["id"], "name": u["display_name"],
                                 "diet": (u["dietary_preferences"][0] if len(u["dietary_preferences"]) == 1 else None),
                                 "allergens": u["allergens"]} for u in list_users()]})
    except Exception:
        logger.exception("Could not list users")
        return jsonify({"error": "Could not read users."}), 500


@bp.get("/insights/<user_id>")
def user_insights(user_id):
    """Return frontend-compatible daily/weekly nutrition and conservative guidance."""
    user = get_user(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404
    now = datetime.now(timezone.utc)
    today = now.date()
    day_start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    week_start = (now - timedelta(days=7)).isoformat().replace("+00:00", "Z")
    rows = user_meal_rows(user_id, week_start)
    by_meal = {}
    for row in rows:
        by_meal.setdefault(row["meal_id"], row)
    daily_map = {}
    for row in by_meal.values():
        date_key = str(row["created_at"])[:10]
        if date_key not in daily_map:
            daily_map[date_key] = {"date": date_key, "calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
        item = daily_map[date_key]
        for out_key, db_key in (("calories", "total_calories"), ("protein", "total_protein_g"), ("carbs", "total_carbs_g"), ("fat", "total_fat_g"), ("fiber", "total_fiber_g")):
            item[out_key] += float(row.get(db_key) or 0)
    daily = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in day.items()}
             for day in sorted(daily_map.values(), key=lambda x: x["date"])]
    week_totals = {key: round(sum(float(row.get(db_key) or 0) for row in by_meal.values()), 2)
                   for key, db_key in (("calories", "total_calories"), ("protein", "total_protein_g"), ("carbs", "total_carbs_g"), ("fat", "total_fat_g"), ("fiber", "total_fiber_g"))}
    week_totals["meals"] = len(by_meal)
    week_totals["avg_daily_calories"] = round(week_totals["calories"] / 7, 2)
    food_names = sorted({str(row["food_name"]) for row in rows if row.get("food_name")})
    warnings = []
    fit_results = []
    allergen_aliases = {
        "lactose / dairy": {"lactose / dairy", "dairy", "milk", "lactose"},
        "eggs": {"eggs", "egg"}, "gluten": {"gluten", "wheat"},
        "nuts": {"nuts", "nut", "peanut", "tree nuts"},
        "seafood / shellfish": {"seafood / shellfish", "seafood", "shellfish", "fish", "prawn", "shrimp"},
        "soy": {"soy", "soya", "soybean"},
    }
    user_allergens = set()
    for value in user["allergens"]:
        key = str(value).strip().casefold()
        user_allergens.update(allergen_aliases.get(key, {key}))
    user_diets = {str(value).strip().casefold() for value in user["dietary_preferences"]}
    for food_name in food_names:
        food = find_food(food_name)
        raw_allergens = str(food.get("allergens", "")).replace(";", ",").split(",")
        food_allergens = {x.strip().casefold() for x in raw_allergens if x.strip() and x.strip().casefold() != "none"}
        conflicts = sorted(value for value in user["allergens"] if user_allergens.intersection(allergen_aliases.get(str(value).strip().casefold(), {str(value).strip().casefold()})) and food_allergens.intersection(allergen_aliases.get(str(value).strip().casefold(), {str(value).strip().casefold()})))
        if conflicts:
            warnings.append({"food": food_name, "allergens": conflicts})
        tags = {x.strip().casefold() for x in str(food.get("diet_tags", "")).replace(";", ",").split(",") if x.strip()}
        if user_diets:
            fit_results.append(bool(tags & user_diets))
    fit_score = round(100 * sum(fit_results) / len(fit_results)) if fit_results else 0
    fit_label = ("Good fit" if fit_score >= 80 else "Mixed fit" if fit_score >= 50 else "Review foods") if fit_results else "Not assessed"
    if fit_results:
        fit_notes = [f"{sum(fit_results)} of {len(fit_results)} logged food entries matched at least one selected diet tag."]
    elif not user_diets:
        fit_notes = ["No diet preference selected; the score is not a personalised assessment."]
    else:
        fit_notes = ["No tagged foods were logged in the last 7 days, so diet fit could not be assessed."]
    return jsonify({"daily": daily, "weekly": week_totals, "allergen_warnings": warnings,
                    "diet_fit": {"score": fit_score, "label": fit_label, "notes": fit_notes},
                    "note": "Nutrition and tag-based guidance are estimates; tags are rule-generated and are not medical advice."})


@bp.get("/foods/<path:food_name>/insights")
def food_insights(food_name):
    """Return conservative nutrition-tag insights without pretending clustering is complete."""
    from .nutrition import normalize_food_name
    try:
        food = find_food(food_name)
        if not food.get("found"):
            return jsonify({"error": "Food not found in the nutrition table."}), 404
        tags = [x.strip() for x in str(food.get("diet_tags", "")).replace(";", ",").split(",") if x.strip() and x.strip().casefold() != "none"]
        allergen_tags = [x.strip() for x in str(food.get("allergens", "")).replace(";", ",").split(",") if x.strip() and x.strip().casefold() != "none"]
        best_time = str(food.get("best_time_to_eat") or "Not specified")
        swaps = []
        try:
            table = get_food_table()
            source_kcal = float(food["energy_kcal"])
            if source_kcal > 0:
                candidates = table[(table["energy_kcal"] <= source_kcal * 0.85) & (table["_normalized_name"] != normalize_food_name(food_name))]
                for row in candidates.sort_values("energy_kcal").head(5).itertuples(index=False):
                    name = str(getattr(row, "food_name"))
                    kcal = float(getattr(row, "energy_kcal"))
                    swaps.append({"food": name, "kcal_saving_pct": round((source_kcal-kcal)/source_kcal*100, 1)})
        except Exception:
            logger.exception("Could not compute food swap candidates")
        return jsonify({"food": food["food_name"], "cluster": {"description": "Nutritional clustering will be added in Part 3; no cluster assignment is claimed yet."},
                        "allergens": allergen_tags, "diet_tags": tags, "best_time": best_time, "swaps": swaps,
                        "note": "Diet tags are rule-generated; lower calories alone do not guarantee a healthier choice."})
    except Exception:
        logger.exception("Could not load food insights")
        return jsonify({"error": "Food insights are unavailable."}), 500


@bp.get("/privacy")
def privacy():
    """Describe the different storage policies of the simulated chat endpoints."""
    return jsonify({
        "raw_chat_text_stored": False,
        "message_text_stored_for_simulated_chat": True,
        "image_retention": "Images are deleted after processing unless keep=true.",
        "database": "SQLite local file; protect it as user data. /api/chat/send stores message text; /api/chat/analyze stores only a hash and derived metadata.",
        "limits": {"chat_characters": 4000, "meal_items": 30, "food_name_characters": 150},
        "note": "No authentication is implemented. Do not expose this development API publicly without authentication, HTTPS, rate limiting, and a retention policy.",
    })
