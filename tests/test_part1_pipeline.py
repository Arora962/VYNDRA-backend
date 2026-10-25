"""Tests for Part 1 photo, user, chat, and serving endpoints."""
from __future__ import annotations
import io
import os
from pathlib import Path
import pytest
from PIL import Image

@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("VYNDRA_DB_PATH", str(db_path))
    import vyndra.database as database
    database.DB_PATH = db_path
    from app import create_app
    application = create_app({"TESTING": True})
    with application.test_client() as test_client:
        yield test_client


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (100, 80, 50)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_user_create_and_list(client):
    response = client.post("/api/users", json={"id": "u1", "name": "Test", "diet": "Vegetarian", "allergens": ["Nuts"]})
    assert response.status_code == 201
    assert response.json == {"id": "u1", "name": "Test", "diet": "Vegetarian", "allergens": ["Nuts"]}
    listed = client.get("/api/users").json["users"][0]
    assert listed == {"id": "u1", "name": "Test", "diet": "Vegetarian", "allergens": ["Nuts"]}


def test_photo_endpoint_is_registered_and_calls_mocked_pipeline(client, monkeypatch):
    import vyndra.photo_api as photo_api
    monkeypatch.setattr(photo_api, "predict_image", lambda *_args, **_kwargs: {
        "top_predictions": [{"class_name": "sample dish", "confidence": 0.9}]
    })
    monkeypatch.setattr(photo_api, "match_class_to_food", lambda _name: {
        "found": True, "food_name": "Hot tea (Garam Chai)", "match_type": "exact", "reviewed": True
    })
    response = client.post("/api/meals/analyze", data={"image": (image_bytes(), "meal.png")}, content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    assert response.json["foods"][0]["food"] == "Hot tea (Garam Chai)"
    assert response.json["foods"][0]["quantity_grams"] > 0
    assert all(key in response.json["summary"] for key in ("energy", "calories", "protein", "carbs", "fat", "fiber"))
    assert {"match_type", "low_confidence", "needs_manual_selection", "suggestions"}.issubset(response.json["foods"][0])
    assert response.json["image_retained"] is False


def test_upload_alias_exists(client, monkeypatch):
    import vyndra.photo_api as photo_api
    monkeypatch.setattr(photo_api, "predict_image", lambda *_args, **_kwargs: {"top_predictions": []})
    response = client.post("/upload", data={"image": (image_bytes(), "meal.png")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.json["foods"] == []


def test_unmatched_class_does_not_receive_guessed_nutrition(client, monkeypatch):
    import vyndra.photo_api as photo_api
    monkeypatch.setattr(photo_api, "predict_image", lambda *_args, **_kwargs: {
        "top_predictions": [{"class_name": "unknown dish", "confidence": 0.9}]
    })
    monkeypatch.setattr(photo_api, "match_class_to_food", lambda _name: {
        "found": False, "match_type": "none", "reviewed": True
    })
    response = client.post("/api/meals/analyze", data={"image": (image_bytes(), "meal.png")}, content_type="multipart/form-data")
    assert response.status_code == 200
    food = response.json["foods"][0]
    assert food["needs_manual_selection"] is True
    assert "calories" not in food
    assert len(food["suggestions"]) == 5


def test_invalid_photo_extension_rejected(client):
    response = client.post("/upload", data={"image": (io.BytesIO(b"not an image"), "meal.gif")}, content_type="multipart/form-data")
    assert response.status_code == 415


def test_send_message_returns_nudge_for_food_intent(client, monkeypatch):
    client.post("/api/users", json={"id": "u1", "name": "Test User"})
    import vyndra.intent_model as intent_model
    monkeypatch.setattr(intent_model, "predict_intent", lambda text: {
        "label": "food_intent", "confidence": 0.9, "is_food_related": True,
        "class_probabilities": {"food_intent": 0.9, "not_food": 0.1}
    })
    response = client.post("/api/chat/send", json={"user_id": "u1", "conversation_id": "c1", "text": "I want biryani"})
    assert response.status_code == 201, response.get_json()
    assert response.json["nudge"] is True
    assert "biryani" in response.json["nudge_text"].lower()
    assert response.json["food"] == "biryani"
    assert set(response.json["message"]) == {"id", "user_id", "conversation_id", "text", "created_at", "intent_label", "probability"}
    assert client.get("/api/chat/c1").json["messages"][0]["text"] == "I want biryani"


def test_send_requires_existing_user(client):
    response = client.post("/api/chat/send", json={"user_id": "missing", "text": "I ate dosa"})
    assert response.status_code == 400


def test_serving_portion_uses_default_for_missing_mapping():
    from vyndra.portions import portion_grams
    result = portion_grams("food with no serving row")
    assert result["quantity_grams"] == 100.0
    assert result["default_portion"] is True


def test_privacy_endpoint_documents_both_chat_storage_behaviours(client):
    response = client.get("/api/privacy")
    assert response.status_code == 200
    assert response.json["raw_chat_text_stored"] is False
    assert response.json["message_text_stored_for_simulated_chat"] is True


def test_frontend_contract_food_insights_and_social_routes(client):
    assert client.get("/api/foods/Hot%20tea%20(Garam%20Chai)/insights").status_code == 200
    influencers = client.get("/api/social/influencers")
    assert influencers.status_code == 200 and isinstance(influencers.json["influencers"], list)
    communities = client.get("/api/social/communities")
    assert communities.status_code == 200
    assert "modularity" in communities.json and "communities" in communities.json


def test_insights_contract(client):
    created = client.post("/api/users", json={"name": "Test User", "diet": "Vegan", "allergens": []})
    assert created.status_code == 201
    response = client.get(f"/api/insights/{created.json['id']}")
    assert response.status_code == 200
    assert isinstance(response.json["daily"], list)
    assert {"calories", "protein", "carbs", "fat", "fiber", "meals", "avg_daily_calories"}.issubset(response.json["weekly"])
    assert {"score", "label", "notes"}.issubset(response.json["diet_fit"])


def test_send_meal_returns_nested_meal_contract(client, monkeypatch):
    client.post("/api/users", json={"id": "u-meal", "name": "Meal User"})
    import vyndra.photo_api as photo_api
    import vyndra.intent_model as intent_model
    monkeypatch.setattr(photo_api, "predict_image", lambda *_args, **_kwargs: {
        "top_predictions": [{"class_name": "sample dish", "confidence": 0.9}]
    })
    monkeypatch.setattr(photo_api, "match_class_to_food", lambda _name: {
        "found": True, "food_name": "Hot tea (Garam Chai)", "match_type": "exact", "reviewed": True
    })
    monkeypatch.setattr(intent_model, "predict_intent", lambda _text: {
        "label": "food_intent", "confidence": 0.9, "is_food_related": True,
        "class_probabilities": {"food_intent": 0.9, "not_food": 0.1}
    })
    response = client.post("/api/chat/send-meal", data={
        "user_id": "u-meal", "conversation_id": "meal-chat", "text": "I had tea",
        "image": (image_bytes(), "meal.png"),
    }, content_type="multipart/form-data")
    assert response.status_code == 201, response.get_json()
    assert set(response.json["message"]) == {"id", "user_id", "conversation_id", "text", "created_at", "intent_label", "probability", "meal"}
    assert {"foods", "summary", "low_confidence"}.issubset(response.json["message"]["meal"])
    assert {"calories", "protein", "carbs", "fat", "fiber"}.issubset(response.json["message"]["meal"]["summary"])


def test_meal_list_returns_expected_summary_fields(client):
    response = client.get("/api/meals?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json["meals"], list)
    if response.json["meals"]:
        assert {"id", "created_at", "source", "notes", "total_calories", "total_protein_g",
                "total_carbs_g", "total_fat_g", "total_fiber_g"}.issubset(response.json["meals"][0])
