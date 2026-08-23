# app.py — VYNDRA backend (classification model, no Supabase/Firebase required)

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
MODEL_WEIGHTS = "best.pt" if os.path.exists("best.pt") else "yolo11n.pt"
DEFAULT_GRAMS = 100

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(text).lower().strip())


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


logging.info(f"Loading food database from {FOOD_DB_PATH}")
food_data = pd.read_csv(FOOD_DB_PATH, encoding="utf-8-sig")
food_data["food_name"] = food_data["food_name"].astype(str).str.lower()
food_data["normalized_food_name"] = food_data["food_name"].apply(normalize_text)

logging.info(f"Loading YOLO model from {MODEL_WEIGHTS}")
model = YOLO(MODEL_WEIGHTS)
logging.info(f"Model task type: {model.task}")  # should print "classify"


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "VYNDRA backend is running.", "model": MODEL_WEIGHTS, "task": model.task})


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
            return jsonify({"error": 'weights must be JSON, e.g. {"kulfi": 150}'}), 400

    # how many top guesses to consider (default 1 — just the best guess)
    top_k = int(request.form.get("top_k", 1))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{secure_filename(image_file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(filepath)

    try:
        image = Image.open(filepath).convert("RGB")
        results = model(image)
        result = results[0]
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    if result.probs is None:
        return jsonify({"error": "Model did not return classification probabilities."}), 500

    top_indices = result.probs.top5[:top_k]
    top_confs = result.probs.top5conf[:top_k]

    detections = []
    for idx, conf in zip(top_indices, top_confs):
        class_name = model.names[int(idx)].strip()
        normalized_name = normalize_text(class_name)
        confidence = float(conf)

        match = food_data[food_data["normalized_food_name"].str.contains(normalized_name, na=False)]
        if match.empty:
            continue

        food = match.iloc[0]
        grams = weights.get(normalized_name, DEFAULT_GRAMS)
        factor = grams / 100.0

        detections.append({
            "food": class_name,
            "confidence": round(confidence, 3),
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
            "message": "Model's best guesses didn't match anything in the food database.",
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

    foods = [{"food": d["food"], "quantity_grams": d["quantity_grams"], "confidence": d["confidence"]} for d in detections]

    return jsonify({"message": "Meal summary created", "foods": foods, "summary": summary})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)