"""VYNDRA Part 5: social graph, food-interest clustering, dietary guidance and healthier swaps.

Bundled inputs are synthetic examples. This is an educational baseline, not medical advice.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "social_recommendations"
EDGE_FILE = DATA / "social_edges.csv"
PROFILE_FILE = DATA / "user_food_profiles.csv"
FOOD_GUIDANCE_FILE = DATA / "food_guidance.csv"
SWAPS_FILE = DATA / "healthier_swaps.csv"
REPORT_DIR = ROOT / "results" / "tables" / "part5"


def load_edges(path: Path = EDGE_FILE) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Social edge CSV not found: {path}")
    df = pd.read_csv(path).fillna("")
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError(f"{path} must contain source,target columns")
    df = df[["source", "target"]].astype(str)
    df = df[(df.source.str.strip() != "") & (df.target.str.strip() != "") & (df.source != df.target)]
    return df.drop_duplicates().reset_index(drop=True)


def _split_tags(value) -> list[str]:
    return sorted({x.strip().lower() for x in str(value).split("|") if x.strip()})


def load_profiles(path: Path = PROFILE_FILE) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"User profile CSV not found: {path}")
    df = pd.read_csv(path).fillna("")
    if not {"user_id", "food_interests"}.issubset(df.columns):
        raise ValueError(f"{path} must contain user_id,food_interests columns")
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["food_interests"] = df["food_interests"].apply(_split_tags)
    return df[df.user_id != ""].drop_duplicates("user_id").reset_index(drop=True)


def build_social_graph(edges: pd.DataFrame, users: Iterable[str] = ()) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(str(u) for u in users if str(u).strip())
    graph.add_edges_from((str(r.source), str(r.target)) for r in edges.itertuples(index=False))
    return graph


def analyse_social_graph(edges: pd.DataFrame, users: Iterable[str] = ()) -> dict:
    graph = build_social_graph(edges, users)
    n = graph.number_of_nodes()
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: (-len(c), c))
    if graph.number_of_edges():
        communities = [sorted(c) for c in nx.community.greedy_modularity_communities(graph)]
    else:
        communities = [[node] for node in graph.nodes()]
    communities.sort(key=lambda c: (-len(c), c))
    return {
        "nodes": n, "edges": graph.number_of_edges(),
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "connected_components": components,
        "community_count": len(communities), "communities": communities,
        "degree_centrality": {str(k): float(v) for k, v in sorted(
            nx.degree_centrality(graph).items(), key=lambda kv: (-kv[1], kv[0]))},
        "note": "Graph results describe supplied data; bundled network is synthetic."
    }


def cluster_food_profiles(profiles: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> dict:
    if profiles.empty: raise ValueError("At least one user profile is required.")
    tags = profiles["food_interests"].tolist()
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(tags)
    if matrix.shape[1] == 0: raise ValueError("No food interests found.")
    k = max(1, min(int(n_clusters), len(profiles)))
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(matrix)
    assignments, clusters = [], []
    for i, label in enumerate(labels):
        assignments.append({"user_id": str(profiles.iloc[i].user_id), "cluster_id": int(label)})
    for cid in range(k):
        members = [i for i, label in enumerate(labels) if int(label) == cid]
        counts = {}
        for i in members:
            for food in tags[i]: counts[food] = counts.get(food, 0) + 1
        clusters.append({"cluster_id": cid, "users": [str(profiles.iloc[i].user_id) for i in members],
                         "top_food_interests": [x for x, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:5]]})
    return {"n_users": len(profiles), "n_clusters": k, "assignments": assignments,
            "clusters": clusters, "note": "Exploratory clusters based on user-supplied food-interest tags."}


def recommend_friends(user_id: str, edges: pd.DataFrame, profiles: pd.DataFrame, limit: int = 5) -> list[dict]:
    user_id = str(user_id)
    graph = build_social_graph(edges, profiles.user_id.astype(str).tolist())
    if user_id not in graph: raise KeyError(f"Unknown user_id: {user_id}")
    lookup = {str(r.user_id): set(r.food_interests) for r in profiles.itertuples(index=False)}
    existing = set(graph.neighbors(user_id))
    candidates = []
    for other in graph.nodes:
        if other == user_id or other in existing: continue
        shared = sorted(lookup.get(user_id, set()) & lookup.get(other, set()))
        mutual = sorted(set(graph.neighbors(user_id)) & set(graph.neighbors(other)))
        score = 2 * len(shared) + len(mutual)
        if score:
            candidates.append({"user_id": other, "score": score, "shared_food_interests": shared,
                               "mutual_friends": mutual})
    return sorted(candidates, key=lambda r: (-r["score"], r["user_id"]))[:max(0, int(limit))]


def recommend_foods(user_id: str, profiles: pd.DataFrame, limit: int = 5) -> list[dict]:
    lookup = {str(r.user_id): set(r.food_interests) for r in profiles.itertuples(index=False)}
    user_id = str(user_id)
    if user_id not in lookup: raise KeyError(f"Unknown user_id: {user_id}")
    own = lookup[user_id]
    if not own: return []
    scores = {}
    for other, foods in lookup.items():
        if other == user_id: continue
        overlap = len(own & foods)
        if overlap:
            for food in foods - own: scores[food] = scores.get(food, 0) + overlap
    return [{"food": f, "score": int(s)} for f, s in sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:max(0, int(limit))]]


def load_guidance(path: Path = FOOD_GUIDANCE_FILE) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Food guidance file not found: {path}")
    df = pd.read_csv(path).fillna("")
    required = {"food", "guidance", "basis"}
    if not required.issubset(df.columns): raise ValueError(f"{path} must contain {sorted(required)}")
    df["food_key"] = df.food.astype(str).str.strip().str.lower()
    return df


def dietary_guidance(food: str, guidance: pd.DataFrame | None = None) -> dict:
    """Return general, non-medical guidance for an explicitly listed food."""
    df = guidance if guidance is not None else load_guidance()
    key = str(food).strip().lower()
    matches = df[df.food_key == key]
    if matches.empty:
        return {"food": food, "found": False, "guidance": None,
                "message": "No curated guidance entry found; do not infer dietary suitability from the food name."}
    row = matches.iloc[0]
    return {"food": str(row.food), "found": True, "guidance": str(row.guidance),
            "basis": str(row.basis), "disclaimer": "General educational information, not personalised medical advice."}


def healthier_swaps(food: str, swaps: pd.DataFrame | None = None, limit: int = 3) -> list[dict]:
    """Return curated alternative ideas; 'healthier' depends on portion, recipe and personal needs."""
    if swaps is None:
        if not SWAPS_FILE.exists(): raise FileNotFoundError(f"Swap data not found: {SWAPS_FILE}")
        swaps = pd.read_csv(SWAPS_FILE).fillna("")
    required = {"food", "alternative", "rationale"}
    if not required.issubset(swaps.columns): raise ValueError(f"Swap data must contain {sorted(required)}")
    key = str(food).strip().lower()
    rows = swaps[swaps.food.astype(str).str.strip().str.lower() == key].head(max(0, int(limit)))
    return [{"food": str(r.food), "alternative": str(r.alternative), "rationale": str(r.rationale),
             "disclaimer": "A general swap idea, not universally healthier for every person."}
            for r in rows.itertuples(index=False)]


def save_part5_reports() -> dict:
    edges, profiles = load_edges(), load_profiles()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    graph = analyse_social_graph(edges, profiles.user_id.tolist())
    clusters = cluster_food_profiles(profiles, 3, 42)
    friends, foods = [], []
    for uid in profiles.user_id:
        for rec in recommend_friends(uid, edges, profiles):
            friends.append({"user_id": uid, **rec})
        for rec in recommend_foods(uid, profiles):
            foods.append({"user_id": uid, **rec})
    guidance = load_guidance()
    guidance_examples = [dietary_guidance(x, guidance) for x in ["pizza", "dal", "dosa", "unknown food"]]
    swaps_df = pd.read_csv(SWAPS_FILE).fillna("")
    swap_examples = [{"input_food": x, "suggestions": healthier_swaps(x, swaps_df)}
                     for x in ["pizza", "soda", "white rice", "deep-fried samosa"]]
    (REPORT_DIR / "social_graph_metrics.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    (REPORT_DIR / "user_clusters.json").write_text(json.dumps(clusters, indent=2), encoding="utf-8")
    (REPORT_DIR / "dietary_guidance_examples.json").write_text(json.dumps(guidance_examples, indent=2), encoding="utf-8")
    (REPORT_DIR / "healthier_swap_examples.json").write_text(json.dumps(swap_examples, indent=2), encoding="utf-8")
    pd.DataFrame(clusters["assignments"]).to_csv(REPORT_DIR / "user_cluster_assignments.csv", index=False)
    pd.DataFrame(friends, columns=["user_id", "score", "shared_food_interests", "mutual_friends"]).to_csv(REPORT_DIR / "friend_recommendations.csv", index=False)
    pd.DataFrame(foods, columns=["user_id", "food", "score"]).to_csv(REPORT_DIR / "food_recommendations.csv", index=False)
    summary = {"users": len(profiles), "social_edges": len(edges), "graph_nodes": graph["nodes"],
               "graph_edges": graph["edges"], "communities": graph["community_count"],
               "clusters": clusters["n_clusters"], "friend_recommendation_rows": len(friends),
               "food_recommendation_rows": len(foods), "guidance_entries": len(guidance),
               "swap_rules": len(swaps_df),
               "data_note": "Synthetic demo social data and curated general guidance; not real user results or medical advice."}
    (REPORT_DIR / "part5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
