"""Single chat blueprint for analysis, messages, nudges, and meal attachments."""
from __future__ import annotations
import logging
from uuid import uuid4
from flask import Blueprint, jsonify, request
from .database import create_user, get_user, get_conversation, save_message
from .chat_intelligence import analyze_sentiment, extract_food_mentions, run_topic_model, DEFAULT_CHAT_DATA, _read_csv

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")
NUDGE_CONFIDENCE_THRESHOLD = 0.45


def _body():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, (jsonify({"error": "Content-Type must be application/json with an object body."}), 400)
    return payload, None


def _intent(text: str):
    """Use the persisted model when present; otherwise train once per process as a safe fallback."""
    from .intent_model import predict_intent
    return predict_intent(text)


def _message_contract(message: dict, include_meal: bool = False) -> dict:
    """Project a database message into the stable frontend response contract."""
    result = {key: message.get(key) for key in
              ("id", "user_id", "conversation_id", "text", "created_at", "intent_label", "probability")}
    if include_meal and message.get("meal") is not None:
        result["meal"] = message["meal"]
    return result


@chat_bp.post("/analyze")
def analyze_message():
    payload, error = _body()
    if error:
        return error
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "text must be a non-empty string."}), 400
    if len(text) > 4000:
        return jsonify({"error": "text must be 4000 characters or fewer."}), 413
    try:
        from .database import save_chat_event
        intent = _intent(text)
        mentions = extract_food_mentions(text)
        sentiment = analyze_sentiment(text)
        save_chat_event(str(intent), str(sentiment), mentions, text)
        return jsonify({"food_intent": intent, "food_mentions": mentions, "sentiment": sentiment,
                        "privacy": {"raw_text_persisted": False, "stored_fields": ["derived labels", "food mentions", "text length", "SHA-256 digest"]}})
    except Exception:
        logger.exception("Chat analysis failed")
        return jsonify({"error": "Chat analysis is unavailable; check installed dependencies and bundled demo data."}), 503


@chat_bp.get("/topics")
def topics():
    try:
        frame = _read_csv(DEFAULT_CHAT_DATA)
        return jsonify(run_topic_model(frame["text"].astype(str).tolist()))
    except Exception:
        logger.exception("Topic analysis failed")
        return jsonify({"error": "Topic analysis is unavailable; check bundled topic data and dependencies."}), 503


@chat_bp.post("/send")
def send_message():
    payload, error = _body()
    if error:
        return error
    text = payload.get("text")
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        return jsonify({"error": "text must be a non-empty string up to 4000 characters."}), 400
    if not isinstance(user_id, str) or not user_id.strip() or not get_user(user_id.strip()):
        return jsonify({"error": "Create a valid user first using POST /api/users and pass user_id."}), 400
    if not isinstance(conversation_id, str) or not conversation_id.strip() or len(conversation_id) > 150:
        return jsonify({"error": "conversation_id must be a non-empty string up to 150 characters."}), 400
    try:
        prediction = _intent(text)
        label = prediction["label"]
        probability = float(prediction["confidence"])
        mentions = extract_food_mentions(text)
        food_name = str(mentions[0]) if mentions else ""
        nudge = label in {"food_intent", "food_discussion"} and probability >= NUDGE_CONFIDENCE_THRESHOLD and bool(food_name)
        nudge_text = f"You mentioned {food_name}. Want to share a photo of your meal?" if nudge else ""
        saved = save_message(user_id.strip(), conversation_id.strip(), text.strip(), label, probability, nudge, nudge_text)
        return jsonify({"message": _message_contract(saved), "nudge": bool(nudge), "nudge_text": nudge_text,
                        "food": food_name}), 201
    except Exception:
        logger.exception("Could not process chat message")
        return jsonify({"error": "Could not process chat message. Check server logs."}), 500


@chat_bp.post("/send-meal")
def send_meal_message():
    if "image" not in request.files:
        return jsonify({"error": "Multipart form field 'image' is required."}), 400
    user_id = (request.form.get("user_id") or "").strip()
    if not user_id or not get_user(user_id):
        return jsonify({"error": "Create a valid user first using POST /api/users and pass user_id."}), 400
    text = (request.form.get("text") or "").strip()
    conversation_id = (request.form.get("conversation_id") or "").strip()
    if not conversation_id or len(conversation_id) > 150:
        return jsonify({"error": "conversation_id is required and must be 150 characters or fewer."}), 400
    try:
        # Reuse the photo view under the current request so validation and cleanup are identical.
        from .photo_api import _analyze_upload
        response = _analyze_upload()
        body, status = response if isinstance(response, tuple) else (response, 200)
        if status != 200:
            return body, status
        result = body.get_json()
        prediction = _intent(text) if text else {"label": "not_food", "confidence": 1.0}
        mentions = extract_food_mentions(text) if text else []
        food_name = str(mentions[0]) if mentions else ""
        nudge = bool(text and food_name and prediction["label"] in {"food_intent", "food_discussion"}
                     and float(prediction["confidence"]) >= NUDGE_CONFIDENCE_THRESHOLD)
        nudge_text = f"You mentioned {food_name}. Want to share a photo of your meal?" if nudge else ""
        saved = save_message(user_id, conversation_id, text, prediction["label"], float(prediction["confidence"]),
                             nudge, nudge_text, result.get("meal_id"))
        meal = {"foods": result.get("foods", []), "summary": {
            "calories": result.get("summary", {}).get("calories", 0),
            "protein": result.get("summary", {}).get("protein", 0),
            "carbs": result.get("summary", {}).get("carbs", 0),
            "fat": result.get("summary", {}).get("fat", 0),
            "fiber": result.get("summary", {}).get("fiber", 0),
        }, "low_confidence": any(bool(food.get("low_confidence")) for food in result.get("foods", []))}
        message = _message_contract(saved)
        message["meal"] = meal
        return jsonify({"message": message, "nudge": bool(nudge), "nudge_text": nudge_text, "food": food_name,
                        "meal_id": result.get("meal_id")}), 201
    except Exception:
        logger.exception("Could not attach meal analysis to message")
        return jsonify({"error": "Could not attach meal analysis to message. Check server logs."}), 500


@chat_bp.get("/<conversation_id>")
def conversation(conversation_id):
    if not conversation_id or len(conversation_id) > 150:
        return jsonify({"error": "Invalid conversation_id."}), 400
    try:
        messages = get_conversation(conversation_id)
        from .database import get_meal
        projected = []
        for raw in messages:  # database query is oldest first
            item = _message_contract(raw)
            if raw.get("meal_id"):
                saved_meal = get_meal(int(raw["meal_id"]))
                if saved_meal:
                    item["meal"] = {"foods": saved_meal.get("items", []), "summary": {
                        "calories": saved_meal.get("total_calories", 0),
                        "protein": saved_meal.get("total_protein_g", 0),
                        "carbs": saved_meal.get("total_carbs_g", 0),
                        "fat": saved_meal.get("total_fat_g", 0),
                        "fiber": saved_meal.get("total_fiber_g", 0)},
                        "low_confidence": bool(saved_meal.get("metadata", {}).get("low_confidence", False))}
            projected.append(item)
        return jsonify({"messages": projected})
    except Exception:
        logger.exception("Could not load conversation")
        return jsonify({"error": "Could not load conversation."}), 500
