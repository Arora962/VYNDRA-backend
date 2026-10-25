"""Photo upload and dish-to-nutrition pipeline."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request
from PIL import Image, UnidentifiedImageError

from .database import get_user, save_meal
from .matching import match_class_to_food
from .nutrition import calculate_nutrition, get_food_table
from .portions import portion_grams
from .vision import predict_image

logger = logging.getLogger(__name__)
photo_bp = Blueprint("photo_upload", __name__)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_PREDICTION_CONFIDENCE = 0.10
# Selected by maximizing F1 for correct-vs-incorrect top-1 predictions in the
# repository's existing 270-image vision_predictions.csv. It is provisional:
# the class mapping is imperfect and must be recalibrated on the expanded Part 4/7 test set.
LOW_CONFIDENCE_THRESHOLD = 0.59


def _suggestions(class_name: str) -> list[str]:
    """Return five search suggestions without inventing nutrition for a class."""
    try:
        foods = get_food_table()["food_name"].astype(str).tolist()
        query = str(class_name).replace("_", " ").lower().strip()
        starts = [name for name in foods if name.lower().startswith(query)]
        contains = [name for name in foods if query and query in name.lower() and name not in starts]
        ranked = []
        try:
            from rapidfuzz import fuzz, process
            ranked = [name for name, _score, _index in process.extract(query, foods, scorer=fuzz.WRatio, limit=20)]
        except ImportError:
            import difflib
            ranked = difflib.get_close_matches(query, foods, n=20, cutoff=0.0)
        ordered = starts + contains + ranked + foods
        result = []
        for name in ordered:
            if name not in result:
                result.append(name)
            if len(result) == 5:
                break
        return result
    except Exception:
        logger.exception("Could not generate food suggestions")
        return []


def _parse_json_field(name: str, default):
    raw = request.form.get(name)
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise ValueError(f"Form field '{name}' must contain valid JSON.")


def _json_grams_for(class_name: str, food_name: str, grams_map) -> float | None:
    if grams_map is None:
        return None
    if not isinstance(grams_map, dict):
        raise ValueError("grams must be a JSON object keyed by predicted class or food name.")
    raw = grams_map.get(class_name, grams_map.get(food_name))
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"grams for {class_name!r} must be numeric.")
    return float(raw)


def _analyze_upload():
    if "image" not in request.files:
        return jsonify({"error": "Multipart form field 'image' is required."}), 400
    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"error": "Choose an image file to upload."}), 400
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported image extension. Use JPG, PNG, WEBP, or BMP."}), 415
    user_id = (request.form.get("user_id") or "").strip() or None
    if user_id and not get_user(user_id):
        return jsonify({"error": "user_id does not exist. Create the user with POST /api/users first."}), 400
    keep = (request.form.get("keep") or "false").lower() in {"1", "true", "yes"}
    try:
        grams_map = _parse_json_field("grams", {})
        servings_map = _parse_json_field("servings", {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not isinstance(servings_map, (dict, int, float)) or isinstance(servings_map, bool):
        return jsonify({"error": "servings must be a number or a JSON object keyed by predicted class or food name."}), 400

    temp_path = None
    try:
        # Stream to a temporary file while enforcing the 8 MB limit.
        with tempfile.NamedTemporaryFile(prefix="vyndra_", suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            total = 0
            while True:
                chunk = uploaded.stream.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    return jsonify({"error": "Image exceeds the 8 MB limit."}), 413
                handle.write(chunk)
        if total == 0:
            return jsonify({"error": "Uploaded image is empty."}), 400
        try:
            with Image.open(temp_path) as image:
                image.verify()
        except Exception:
            return jsonify({"error": "Uploaded file is not a valid readable image."}), 400

        prediction_result = predict_image(temp_path, top_k=3)
        predictions = [p for p in prediction_result.get("top_predictions", [])
                       if float(p.get("confidence", 0)) >= MIN_PREDICTION_CONFIDENCE][:3]
        foods = []
        saved_items = []
        for prediction in predictions:
            class_name = str(prediction["class_name"])
            confidence = float(prediction["confidence"])
            match = match_class_to_food(class_name)
            match_type = match.get("match_type", "none")
            item = {
                "food": match.get("food_name") if match.get("found") else class_name,
                "class_name": class_name,
                "quantity_grams": None,
                "confidence": round(confidence, 6),
                "source": "yolo11n-cls",
                "match_type": match_type,
                "reviewed": bool(match.get("reviewed", False)),
                "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
                "found": bool(match.get("found")),
                "needs_manual_selection": False,
                "suggestions": [],
            }
            if not match.get("found") or match_type == "none":
                item.update({"found": False, "needs_manual_selection": True,
                             "suggestions": _suggestions(class_name),
                             "message": "No approved nutrition equivalent; nutrition was not guessed."})
                foods.append(item)
                continue
            if match_type not in {"exact", "family"}:
                item.update({"found": False, "needs_manual_selection": True,
                             "suggestions": _suggestions(class_name),
                             "message": "Mapping is not approved for automatic nutrition calculation."})
                foods.append(item)
                continue
            explicit_grams = _json_grams_for(class_name, match["food_name"], grams_map)
            if isinstance(servings_map, dict):
                servings = servings_map.get(class_name, servings_map.get(match["food_name"]))
            else:
                servings = servings_map
            portion = portion_grams(match["food_name"], grams=explicit_grams, servings=servings)
            grams = float(portion["quantity_grams"])
            if grams <= 0 or grams > 5000:
                raise ValueError("Resolved portion must be greater than 0 and at most 5000 grams.")
            nutrition = calculate_nutrition(match["food_name"], grams)
            if not nutrition.get("found"):
                item.update({"found": False, "needs_manual_selection": True,
                             "suggestions": _suggestions(class_name),
                             "message": "Approved mapping did not resolve to a nutrition row."})
                foods.append(item)
                continue
            item.update({
                "food": nutrition["food_name"], "quantity_grams": grams,
                "default_portion": bool(portion["default_portion"]),
                "portion_source": portion["portion_source"],
                "servings_unit": portion["servings_unit"],
                "energy": nutrition["energy_kcal"], "calories": nutrition["energy_kcal"],
                "protein": nutrition["protein_g"], "carbs": nutrition["carb_g"],
                "fat": nutrition["fat_g"], "fiber": nutrition["fibre_g"],
                "nutrition_basis": nutrition["basis"],
            })
            saved_items.append({"food_name": nutrition["food_name"], "quantity_g": grams,
                                "calories": nutrition["energy_kcal"], "protein_g": nutrition["protein_g"],
                                "carbs_g": nutrition["carb_g"], "fat_g": nutrition["fat_g"],
                                "fiber_g": nutrition["fibre_g"], "match_type": match_type})
            foods.append(item)

        totals = {key: round(sum(float(item.get(key, 0) or 0) for item in foods), 2)
                  for key in ("energy", "calories", "protein", "carbs", "fat", "fiber")}
        meal = save_meal(saved_items, source="photo", notes="Photo-based meal analysis", user_id=user_id,
                         metadata={"predictions": predictions, "model": "models/best.pt",
                                   "low_confidence": any(bool(item.get("low_confidence")) for item in foods)})
        response = {
            "message": "Image analysed. Review predictions and nutrition matches before relying on them.",
            "foods": foods, "summary": totals, "meal_id": meal["id"],
            "model": "models/best.pt", "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "image_retained": bool(keep),
        }
        if keep:
            retained_dir = Path(os.getenv("VYNDRA_UPLOAD_DIR", str(Path(__file__).resolve().parents[1] / "uploads")))
            retained_dir.mkdir(parents=True, exist_ok=True)
            destination = retained_dir / f"{uuid4().hex}{suffix}"
            temp_path.replace(destination)
            temp_path = None
            response["retained_image_name"] = destination.name
        return jsonify(response), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        logger.exception("Photo pipeline file missing")
        return jsonify({"error": "Photo analysis model or required data file is missing. Check models/best.pt and data/raw/Db.csv."}), 503
    except Exception:
        logger.exception("Photo analysis failed")
        return jsonify({"error": "Photo analysis failed. Check server logs and verify the model dependencies."}), 500
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete temporary image %s", temp_path)


@photo_bp.post("/api/meals/analyze")
def analyze_meal_photo():
    """Analyze a meal photo; temporary image data is deleted after processing by default."""
    return _analyze_upload()


@photo_bp.post("/upload")
def legacy_upload_alias():
    """Compatibility alias for the former mobile app upload route."""
    return _analyze_upload()
