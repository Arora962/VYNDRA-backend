from vyndra.chat_intelligence import (
    analyze_sentiment, extract_food_mentions, load_food_lexicon,
    predict_food_intent, run_topic_model, train_intent_model,
)


def test_food_mention_extraction_known_terms():
    found = extract_food_mentions(
        "I ate dosa and chicken tikka masala.",
        foods=["dosa", "chicken", "chicken tikka masala"]
    )
    assert found == ["dosa", "chicken tikka masala"]


def test_sentiment_returns_valid_label():
    result = analyze_sentiment("This meal was wonderful and delicious!")
    assert result["label"] in {"positive", "neutral", "negative"}
    assert -1.0 <= result["compound"] <= 1.0


def test_intent_model_can_predict():
    model, metrics = train_intent_model()
    result = predict_food_intent("I am craving dosa for dinner", model=model)
    assert result["label"] in {"food_intent", "food_discussion", "not_food"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert metrics["test_examples"] > 0


def test_topic_model_returns_assignments():
    result = run_topic_model([
        "I love dosa and idli for breakfast",
        "Pizza and burgers are my favourite foods",
        "I went running in the park today",
        "I need to finish my project and study"
    ], n_topics=2)
    assert len(result["topics"]) == 2
    assert len(result["assignments"]) == 4
