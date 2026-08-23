# app.py — VYNDRA backend (hybrid: custom classifier + generic detectors, no Supabase/Firebase)

import difflib
import os
import re
import json
import logging
from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)

# ---- Config ----
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
FOOD_DB_PATH = "Db.csv"
DEFAULT_GRAMS = 100

# Your custom Indian-food classification model (single best-guess label per image)
CLASSIFY_WEIGHTS = "best.pt"
CLASSIFY_TOP_K = 3          # how many of its top guesses to consider
CLASSIFY_MIN_CONF = 0.10    # ignore guesses below this confidence

# Generic COCO-pretrained detectors (bounding boxes; know things like cake, pizza,
# banana, apple, orange, donut, hot dog, sandwich, broccoli, carrot)
DETECT_WEIGHTS = ["yolo11n.pt", "yolov8n.pt"]
DETECT_MIN_CONF = 0.25       # ultralytics' usual default detection threshold

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(text).lower().strip())


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---- Local food nutrition database (replaces Supabase "food_data" table) ----
logging.info(f"Loading food database from {FOOD_DB_PATH}")
food_data = pd.read_csv(FOOD_DB_PATH, encoding="utf-8-sig")
food_data["food_name"] = food_data["food_name"].astype(str).str.lower()
food_data["normalized_food_name"] = food_data["food_name"].apply(normalize_text)


def match_food(normalized_name: str):
    """Exact substring match first, fuzzy match as fallback. Returns a row or None."""
    match = food_data[food_data["normalized_food_name"].str.contains(normalized_name, na=False)]
    if not match.empty:
        return match.iloc[0]

    close = difflib.get_close_matches(
        normalized_name, food_data["normalized_food_name"].tolist(), n=1, cutoff=0.6
    )
    if close:
        return food_data[food_data["normalized_food_name"] == close[0]].iloc[0]

    return None


# ---- Load every model you have available ----
classify_model = None
if os.path.exists(CLASSIFY_WEIGHTS):
    logging.info(f"Loading classification model: {CLASSIFY_WEIGHTS}")
    classify_model = YOLO(CLASSIFY_WEIGHTS)
    logging.info(f"  task type: {classify_model.task}")
else:
    logging.warning(f"{CLASSIFY_WEIGHTS} not found — skipping custom classifier")

detect_models = []
for weights_path in DETECT_WEIGHTS:
    if os.path.exists(weights_path):
        logging.info(f"Loading detection model: {weights_path}")
        m = YOLO(weights_path)
        logging.info(f"  task type: {m.task}")
        detect_models.append((weights_path, m))
    else:
        logging.warning(f"{weights_path} not found — skipping")

if not classify_model and not detect_models:
    raise RuntimeError("No model weights found. Need at least one of: best.pt, yolo11n.pt, yolov8n.pt")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "VYNDRA backend is running.",
        "classify_model": CLASSIFY_WEIGHTS if classify_model else None,
        "detect_models": [w for w, _ in detect_models],
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    image_file = request.files["image"]
    if image_file.filename == "" or not allowed_file(image_file.filename):
        return jsonify({"error": "Invalid image type"}), 400

    weights = {}
    weights_str = request.form.get("weights")
    if weights_str:
        try:
            weights = json.loads(weights_str)
        except Exception:
            return jsonify({"error": 'weights must be JSON, e.g. {"cake": 150}'}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{secure_filename(image_file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(filepath)

    try:
        image = Image.open(filepath).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 500

    # candidate_name -> {class_name, confidence, source}
    candidates = {}

    def consider(class_name: str, confidence: float, source: str):
        normalized_name = normalize_text(class_name)
        existing = candidates.get(normalized_name)
        if existing is None or confidence > existing["confidence"]:
            candidates[normalized_name] = {
                "class_name": class_name,
                "confidence": confidence,
                "source": source,
            }

    # --- 1. Custom Indian-food classifier (single best-guess label per image) ---
    if classify_model:
        try:
            result = classify_model(image)[0]
            if result.probs is not None:
                top_indices = result.probs.top5[:CLASSIFY_TOP_K]
                top_confs = result.probs.top5conf[:CLASSIFY_TOP_K]
                for idx, conf in zip(top_indices, top_confs):
                    conf = float(conf)
                    if conf < CLASSIFY_MIN_CONF:
                        continue
                    class_name = classify_model.names[int(idx)].strip()
                    consider(class_name, conf, CLASSIFY_WEIGHTS)
        except Exception as e:
            logging.error(f"Classifier failed: {e}")

    # --- 2. Generic COCO detectors (can find multiple distinct food items) ---
    for weights_path, det_model in detect_models:
        try:
            results = det_model(image, conf=DETECT_MIN_CONF, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    class_name = det_model.names[int(box.cls[0])].strip()
                    consider(class_name, conf, weights_path)
        except Exception as e:
            logging.error(f"{weights_path} failed: {e}")

    # --- 3. Match every surviving candidate against the nutrition database ---
    detections = []
    for normalized_name, info in candidates.items():
        food = match_food(normalized_name)
        if food is None:
            continue

        grams = weights.get(normalized_name, DEFAULT_GRAMS)
        factor = grams / 100.0

        detections.append({
            "food": info["class_name"],
            "confidence": round(info["confidence"], 3),
            "source": info["source"],
            "quantity_grams": grams,
            "energy": food["energy_kj"] * factor,
            "calories": food["energy_kcal"] * factor,
            "protein": food["protein_g"] * factor,
            "carbs": food["carb_g"] * factor,
            "fat": food["fat_g"] * factor,
            "fiber": food["fibre_g"] * factor,
        })

    if not detections:
        return jsonify({
            "message": "No food items recognized in this image.",
            "foods": [],
            "summary": {"energy": 0, "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0},
        })

    summary = {
        "energy": sum(d["energy"] for d in detections),
        "calories": sum(d["calories"] for d in detections),
        "protein": sum(d["protein"] for d in detections),
        "carbs": sum(d["carbs"] for d in detections),
        "fat": sum(d["fat"] for d in detections),
        "fiber": sum(d["fiber"] for d in detections),
    }

    foods = [
        {
            "food": d["food"],
            "quantity_grams": d["quantity_grams"],
            "confidence": d["confidence"],
            "source": d["source"],
        }
        for d in detections
    ]

    return jsonify({"message": "Meal summary created", "foods": foods, "summary": summary})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)