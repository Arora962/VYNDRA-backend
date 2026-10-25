import pandas as pd
from vyndra.social_recommendations import (
    analyse_social_graph, cluster_food_profiles, recommend_foods, recommend_friends,
    dietary_guidance, healthier_swaps, load_guidance, FOOD_GUIDANCE_FILE, SWAPS_FILE
)

def edges():
    return pd.DataFrame([{"source":"u1","target":"u2"},{"source":"u2","target":"u3"}])

def profiles():
    return pd.DataFrame([
        {"user_id":"u1","food_interests":["dosa","idli"]},
        {"user_id":"u2","food_interests":["dosa","pizza"]},
        {"user_id":"u3","food_interests":["pizza","burger"]},
        {"user_id":"u4","food_interests":["dosa","burger"]},
    ])

def test_graph_metrics():
    r = analyse_social_graph(edges(), ["u4"])
    assert r["nodes"] == 4 and r["edges"] == 2

def test_friend_recommendations_exclude_self_and_existing_connections():
    ids = [x["user_id"] for x in recommend_friends("u1", edges(), profiles(), 10)]
    assert "u1" not in ids and "u2" not in ids

def test_clustering_assigns_all_profiles():
    r = cluster_food_profiles(profiles(), 2)
    assert len(r["assignments"]) == 4

def test_food_recommendations_do_not_repeat_own_foods():
    own = set(profiles().iloc[0]["food_interests"])
    assert not own.intersection(x["food"] for x in recommend_foods("u1", profiles()))

def test_dietary_guidance_known_and_unknown():
    g = load_guidance()
    assert dietary_guidance("pizza", g)["found"] is True
    assert dietary_guidance("imaginary food", g)["found"] is False

def test_healthier_swaps_return_curated_alternatives():
    swaps = healthier_swaps("soda")
    assert len(swaps) >= 1
    assert "alternative" in swaps[0] and "rationale" in swaps[0]
