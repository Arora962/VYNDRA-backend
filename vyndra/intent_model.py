"""Load the food-intent classifier once and provide stable inference helpers."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import logging
import joblib
from .chat_intelligence import train_intent_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "food_intent.joblib"
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_intent_model():
    """Load the saved model or fit a single in-process fallback if it has not been built yet."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    logger.warning("%s is absent; training a one-time in-memory fallback. Run scripts/train_food_intent.py for reproducibility.", MODEL_PATH)
    model, _ = train_intent_model()
    return model

def predict_intent(text: str) -> dict:
    """Return class label and confidence from the persisted/fallback model."""
    model = load_intent_model()
    label = str(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))
        scores = {str(name): float(score) for name, score in zip(model.classes_, probabilities)}
    else:
        confidence, scores = 0.0, {}
    return {"label": label, "is_food_related": label != "not_food", "confidence": confidence, "class_probabilities": scores}
