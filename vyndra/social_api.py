"""Optional Flask blueprint for Part 5; register after reviewing app.py."""
from flask import Blueprint, jsonify, request
from .social_recommendations import (
    load_edges, load_profiles, analyse_social_graph, cluster_food_profiles,
    recommend_friends, recommend_foods, dietary_guidance, healthier_swaps
)
social_bp = Blueprint("social_recommendations", __name__, url_prefix="/api/social")

@social_bp.get("/graph")
def graph_metrics():
    try:
        e, p = load_edges(), load_profiles()
        return jsonify(analyse_social_graph(e, p.user_id.tolist()))
    except Exception as exc: return jsonify({"error": str(exc)}), 500

@social_bp.get("/clusters")
def clusters():
    try:
        n = request.args.get("n_clusters", default=3, type=int)
        if n is None or n < 1: return jsonify({"error": "n_clusters must be positive"}), 400
        return jsonify(cluster_food_profiles(load_profiles(), n))
    except Exception as exc: return jsonify({"error": str(exc)}), 500

@social_bp.get("/recommendations/<user_id>")
def recommendations(user_id):
    try:
        e, p = load_edges(), load_profiles()
        limit = request.args.get("limit", default=5, type=int)
        if limit is None or not 0 <= limit <= 50: return jsonify({"error": "limit must be 0..50"}), 400
        return jsonify({"user_id": user_id, "friends": recommend_friends(user_id, e, p, limit),
                        "foods": recommend_foods(user_id, p, limit)})
    except KeyError: return jsonify({"error": f"Unknown user_id: {user_id}"}), 404
    except Exception as exc: return jsonify({"error": str(exc)}), 500

@social_bp.get("/dietary-guidance/<food>")
def guidance(food):
    try: return jsonify(dietary_guidance(food))
    except Exception as exc: return jsonify({"error": str(exc)}), 500

@social_bp.get("/healthier-swaps/<food>")
def swaps(food):
    try:
        limit = request.args.get("limit", default=3, type=int)
        if not 0 <= limit <= 20: return jsonify({"error": "limit must be 0..20"}), 400
        return jsonify({"food": food, "swaps": healthier_swaps(food, limit=limit)})
    except Exception as exc: return jsonify({"error": str(exc)}), 500
