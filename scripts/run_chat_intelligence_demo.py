#!/usr/bin/env python3
"""Quick local demo of the Part 4 components."""
import json
from vyndra.chat_intelligence import (
    analyze_sentiment, extract_food_mentions, predict_food_intent, train_intent_model
)

def main():
    model, metrics = train_intent_model()
    messages = [
        "I'm craving dosa and chutney for dinner!",
        "That biryani was amazing.",
        "The meeting starts at ten."
    ]
    for text in messages:
        print(json.dumps({
            "text": text,
            "food_intent": predict_food_intent(text, model=model),
            "food_mentions": extract_food_mentions(text),
            "sentiment": analyze_sentiment(text),
        }, indent=2))
    print(f"Intent demo hold-out accuracy: {metrics['accuracy']:.3f}")
    print("Reminder: this is a small demonstration dataset, not production validation.")

if __name__ == "__main__":
    main()
