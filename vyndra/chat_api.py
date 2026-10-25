"""Optional Flask Blueprint for Part 4 endpoints. Register in app.py if desired."""
from flask import Blueprint, jsonify, request

from .chat_intelligence import (
    analyze_sentiment, extract_food_mentions, predict_food_intent, run_topic_model,
    DEFAULT_CHAT_DATA, _read_csv
)

chat_bp = Blueprint("chat_intelligence", __name__, url_prefix="/api/chat")


@chat_bp.post("/analyze")
def analyze_message():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "JSON body must include a non-empty string field named 'text'."}), 400
    try:
        return jsonify({
            "food_intent": predict_food_intent(text),
            "food_mentions": extract_food_mentions(text),
            "sentiment": analyze_sentiment(text),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@chat_bp.get("/topics")
def topics():
    try:
        frame = _read_csv(DEFAULT_CHAT_DATA)
        from .chat_intelligence import run_topic_model
        result = run_topic_model(frame["text"].astype(str).tolist())
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
