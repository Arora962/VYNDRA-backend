"""Small, explicit wrapper for VYNDRA's trained Ultralytics classifier."""
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best.pt"


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    from ultralytics import YOLO
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    return YOLO(str(path))


def predict_image(image_path: str | Path, top_k: int = 3, model_path: str | Path = DEFAULT_MODEL) -> dict:
    """Return top-k classifier predictions. This is not nutrition matching or an accuracy score."""
    image = Path(image_path).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10")
    model = _load_model(str(Path(model_path).expanduser().resolve()))
    results = model.predict(source=str(image), verbose=False)
    if not results:
        raise RuntimeError(f"The model returned no result for {image}")
    result = results[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        raise RuntimeError("models/best.pt did not return classification probabilities. Check that this is the VYNDRA classification model.")
    indices = probs.top5[:top_k]
    values = probs.top5conf[:top_k]
    names = result.names
    predictions = []
    for index, confidence in zip(indices, values):
        name = names[int(index)] if isinstance(names, dict) else names[int(index)]
        predictions.append({"class_name": str(name), "confidence": round(float(confidence), 6)})
    return {"image": str(image), "model": str(Path(model_path).expanduser().resolve()), "top_predictions": predictions}
