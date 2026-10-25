"""Social analytics endpoints, including stable frontend contract routes."""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd
from flask import Blueprint, jsonify, request

from .database import list_users
from .social_recommendations import (
    analyse_social_graph,
    build_social_graph,
    cluster_food_profiles,
    dietary_guidance,
    healthier_swaps,
    load_edges,
    load_profiles,
    recommend_friends,
    recommend_foods,
)

logger = logging.getLogger(__name__)
social_bp = Blueprint("social_recommendations", __name__, url_prefix="/api/social")


def _graph_and_profiles():
    edges = load_edges()
    profiles = load_profiles()
    graph = build_social_graph(edges, profiles.user_id.astype(str).tolist())
    return edges, profiles, graph


@social_bp.get("/graph")
def graph_metrics():
    try:
        edges, profiles, _graph = _graph_and_profiles()
        return jsonify(analyse_social_graph(edges, profiles.user_id.tolist()))
    except Exception:
        logger.exception("Social graph metrics failed")
        return jsonify({"error": "Social graph metrics are unavailable."}), 500


@social_bp.get("/clusters")
def clusters():
    try:
        n = request.args.get("n_clusters", default=3, type=int)
        if n is None or n < 1:
            return jsonify({"error": "n_clusters must be positive"}), 400
        return jsonify(cluster_food_profiles(load_profiles(), n))
    except Exception:
        logger.exception("Social profile clustering failed")
        return jsonify({"error": "Social profile clustering is unavailable."}), 500


@social_bp.get("/recommendations/<user_id>")
def recommendations(user_id):
    try:
        edges, profiles, _graph = _graph_and_profiles()
        limit = request.args.get("limit", default=5, type=int)
        if limit is None or not 0 <= limit <= 50:
            return jsonify({"error": "limit must be 0..50"}), 400
        return jsonify({"user_id": user_id, "friends": recommend_friends(user_id, edges, profiles, limit),
                        "foods": recommend_foods(user_id, profiles, limit)})
    except KeyError:
        return jsonify({"error": "Unknown social profile user."}), 404
    except Exception:
        logger.exception("Social recommendations failed")
        return jsonify({"error": "Social recommendations are unavailable."}), 500


@social_bp.get("/dietary-guidance/<food>")
def guidance(food):
    try:
        return jsonify(dietary_guidance(food))
    except Exception:
        logger.exception("Dietary guidance failed")
        return jsonify({"error": "Dietary guidance is unavailable."}), 500


@social_bp.get("/healthier-swaps/<food>")
def swaps(food):
    limit = request.args.get("limit", default=3, type=int)
    if limit is None or not 0 <= limit <= 20:
        return jsonify({"error": "limit must be 0..20"}), 400
    try:
        return jsonify({"food": food, "swaps": healthier_swaps(food, limit=limit)})
    except Exception:
        logger.exception("Healthier swaps failed")
        return jsonify({"error": "Healthier swaps are unavailable."}), 500


@social_bp.get("/influencers")
def influencers():
    """Rank the bundled social graph by PageRank, with complementary centralities."""
    try:
        _edges, profiles, graph = _graph_and_profiles()
        if graph.number_of_nodes() == 0:
            return jsonify({"influencers": []})
        pagerank = nx.pagerank(graph) if graph.number_of_edges() else {node: 1 / graph.number_of_nodes() for node in graph}
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        closeness = nx.closeness_centrality(graph)
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
        except nx.NetworkXException:
            eigenvector = {node: 0.0 for node in graph}
        degree = nx.degree_centrality(graph)
        names = {str(user["id"]): user.get("display_name", "") for user in list_users()}
        rows = []
        for user_id in graph.nodes:
            rows.append({"user_id": str(user_id), "name": names.get(str(user_id)) or str(user_id),
                         "pagerank": round(float(pagerank[user_id]), 8),
                         "betweenness": round(float(betweenness[user_id]), 8),
                         "closeness": round(float(closeness[user_id]), 8),
                         "eigenvector": round(float(eigenvector[user_id]), 8),
                         "degree": round(float(degree[user_id]), 8)})
        rows.sort(key=lambda row: (-row["pagerank"], -row["betweenness"], row["user_id"]))
        return jsonify({"influencers": rows})
    except Exception:
        logger.exception("Influencer ranking failed")
        return jsonify({"error": "Influencer ranking is unavailable."}), 500


@social_bp.get("/communities")
def communities():
    """Return Louvain communities and modularity for the supplied bundled graph."""
    try:
        _edges, profiles, graph = _graph_and_profiles()
        if graph.number_of_nodes() == 0:
            return jsonify({"modularity": 0.0, "communities": []})
        if graph.number_of_edges():
            try:
                groups = list(nx.community.louvain_communities(graph, seed=42))
            except AttributeError:
                groups = list(nx.community.greedy_modularity_communities(graph))
            modularity = float(nx.community.modularity(graph, groups)) if groups else 0.0
        else:
            groups = [{node} for node in graph.nodes]
            modularity = 0.0
        groups = sorted((sorted(str(node) for node in group) for group in groups), key=lambda group: (-len(group), group))
        return jsonify({"modularity": round(modularity, 6),
                        "communities": [{"id": index, "members": members} for index, members in enumerate(groups)]})
    except Exception:
        logger.exception("Community detection failed")
        return jsonify({"error": "Community detection is unavailable."}), 500
