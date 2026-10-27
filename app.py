import os
import io
import base64
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "traffic_classifier.h5")
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
ICONS_DIR = os.path.join(BASE_DIR, "static", "icons")

# Complete GTSRB (German Traffic Sign Recognition Benchmark) 43 Classes Dictionary
SIGN_CLASSES = {
    0: {
        "name": "Speed limit (20km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Maintain vehicle speed at or strictly below 20 km/h. Watch for pedestrians in shared zones.",
        "risk_level": "Low"
    },
    1: {
        "name": "Speed limit (30km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Maintain speed under 30 km/h. Typical in residential zones, school zones, and city centers.",
        "risk_level": "Low"
    },
    2: {
        "name": "Speed limit (50km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Standard urban area speed limit. Keep speed under 50 km/h.",
        "risk_level": "Low"
    },
    3: {
        "name": "Speed limit (60km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Drive at or below 60 km/h. Often found on urban express routes.",
        "risk_level": "Moderate"
    },
    4: {
        "name": "Speed limit (70km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Drive at or below 70 km/h on designated peri-urban arterial roads.",
        "risk_level": "Moderate"
    },
    5: {
        "name": "Speed limit (80km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Maintain speed under 80 km/h. Standard maximum for rural roads without separation.",
        "risk_level": "Moderate"
    },
    6: {
        "name": "End of speed limit (80km/h)",
        "category": "De-restriction",
        "badge_color": "#94a3b8",
        "action": "80 km/h restriction lifted. Resume normal road speed according to highway rules.",
        "risk_level": "Info"
    },
    7: {
        "name": "Speed limit (100km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Maintain speed under 100 km/h. Applicable on dual carriageways and rural highways.",
        "risk_level": "High"
    },
    8: {
        "name": "Speed limit (120km/h)",
        "category": "Speed Limit",
        "badge_color": "#38bdf8",
        "action": "Maximum speed 120 km/h. Typical expressway / motorway speed limit.",
        "risk_level": "High"
    },
    9: {
        "name": "No passing",
        "category": "Prohibitory",
        "badge_color": "#f43f5e",
        "action": "Overtaking motor vehicles is strictly forbidden. Stay in your lane.",
        "risk_level": "High"
    },
    10: {
        "name": "No passing for vehicles over 3.5 metric tons",
        "category": "Prohibitory",
        "badge_color": "#f43f5e",
        "action": "Heavy trucks and vehicles over 3.5 tonnes must not overtake.",
        "risk_level": "Moderate"
    },
    11: {
        "name": "Right-of-way at the next intersection",
        "category": "Priority",
        "badge_color": "#10b981",
        "action": "You have priority at the upcoming crossroads only. Proceed cautiously.",
        "risk_level": "Moderate"
    },
    12: {
        "name": "Priority road",
        "category": "Priority",
        "badge_color": "#10b981",
        "action": "You are on a priority road. Cross-traffic must yield until end-of-priority sign.",
        "risk_level": "Low"
    },
    13: {
        "name": "Yield",
        "category": "Priority",
        "badge_color": "#f59e0b",
        "action": "Give way to oncoming and intersecting traffic. Slow down and stop if necessary.",
        "risk_level": "High"
    },
    14: {
        "name": "Stop",
        "category": "Mandatory",
        "badge_color": "#ef4444",
        "action": "Mandatory full vehicle stop before stop line. Proceed only when road is clear.",
        "risk_level": "Critical"
    },
    15: {
        "name": "No vehicles",
        "category": "Prohibitory",
        "badge_color": "#f43f5e",
        "action": "Road closed to all vehicular traffic in both directions. Turn around.",
        "risk_level": "High"
    },
    16: {
        "name": "Vehicles over 3.5 metric tons prohibited",
        "category": "Prohibitory",
        "badge_color": "#f43f5e",
        "action": "Vehicles exceeding 3.5t gross weight must not enter.",
        "risk_level": "Moderate"
    },
    17: {
        "name": "No entry",
        "category": "Prohibitory",
        "badge_color": "#dc2626",
        "action": "Strictly no entry! Driving the wrong way into a one-way road is hazardous.",
        "risk_level": "Critical"
    },
    18: {
        "name": "General caution",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Unspecified roadway hazard ahead. Reduce speed, scan ahead, and be prepared to stop.",
        "risk_level": "High"
    },
    19: {
        "name": "Dangerous curve to the left",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Sharp bend to the left ahead. Decelerate smoothly before entering the turn.",
        "risk_level": "High"
    },
    20: {
        "name": "Dangerous curve to the right",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Sharp bend to the right ahead. Decelerate smoothly before entering the turn.",
        "risk_level": "High"
    },
    21: {
        "name": "Double curve",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Winding road / series of curves ahead. Maintain steady speed and stay inside lane markings.",
        "risk_level": "High"
    },
    22: {
        "name": "Bumpy road",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Uneven road surface ahead. Slow down to avoid suspension damage or losing control.",
        "risk_level": "Moderate"
    },
    23: {
        "name": "Slippery road",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Road surface offers low friction. Avoid sudden acceleration, hard braking, or sharp turns.",
        "risk_level": "High"
    },
    24: {
        "name": "Road narrows on the right",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Right lane ends or road narrows on right side. Merge left safely using zipper method.",
        "risk_level": "Moderate"
    },
    25: {
        "name": "Road work",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Road maintenance or construction ahead. Watch for workers, equipment, and detours.",
        "risk_level": "High"
    },
    26: {
        "name": "Traffic signals",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Traffic light intersection ahead. Be ready to stop if signal turns amber or red.",
        "risk_level": "Moderate"
    },
    27: {
        "name": "Pedestrians",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Pedestrian activity area. Slow down and be ready to yield to people crossing.",
        "risk_level": "Critical"
    },
    28: {
        "name": "Children crossing",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "School zone or playground nearby. Exercise maximum vigilance and drive slowly.",
        "risk_level": "Critical"
    },
    29: {
        "name": "Bicycles crossing",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Cycle route crosses roadway ahead. Check blind spots and yield to cyclists.",
        "risk_level": "High"
    },
    30: {
        "name": "Beware of ice/snow",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Sub-zero conditions can cause invisible black ice. Increase following distance.",
        "risk_level": "Critical"
    },
    31: {
        "name": "Wild animals crossing",
        "category": "Danger",
        "badge_color": "#f97316",
        "action": "Deer / wildlife habitat zone. Keep headlights clean and watch road shoulders.",
        "risk_level": "Moderate"
    },
    32: {
        "name": "End of all speed and passing limits",
        "category": "De-restriction",
        "badge_color": "#94a3b8",
        "action": "All previous temporary speed and overtaking bans end. Standard rules apply.",
        "risk_level": "Info"
    },
    33: {
        "name": "Turn right ahead",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Mandatory direction: Turn right at the next junction.",
        "risk_level": "Moderate"
    },
    34: {
        "name": "Turn left ahead",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Mandatory direction: Turn left at the next junction.",
        "risk_level": "Moderate"
    },
    35: {
        "name": "Ahead only",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Mandatory direction: Proceed straight ahead only. Turns are prohibited.",
        "risk_level": "Moderate"
    },
    36: {
        "name": "Go straight or right",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Mandatory direction: Proceed straight or turn right. Left turn prohibited.",
        "risk_level": "Moderate"
    },
    37: {
        "name": "Go straight or left",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Mandatory direction: Proceed straight or turn left. Right turn prohibited.",
        "risk_level": "Moderate"
    },
    38: {
        "name": "Keep right",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Pass on the right-hand side of traffic island or divider.",
        "risk_level": "Low"
    },
    39: {
        "name": "Keep left",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Pass on the left-hand side of traffic island or divider.",
        "risk_level": "Low"
    },
    40: {
        "name": "Roundabout mandatory",
        "category": "Mandatory",
        "badge_color": "#0284c7",
        "action": "Traffic circle ahead. Circulate counter-clockwise and yield to traffic inside.",
        "risk_level": "Moderate"
    },
    41: {
        "name": "End of no passing",
        "category": "De-restriction",
        "badge_color": "#94a3b8",
        "action": "Overtaking restriction ends. Vehicles may overtake when safe.",
        "risk_level": "Info"
    },
    42: {
        "name": "End of no passing by vehicles over 3.5 metric tons",
        "category": "De-restriction",
        "badge_color": "#94a3b8",
        "action": "Truck overtaking ban ends. Normal lane rules resume.",
        "risk_level": "Info"
    }
}

# Global model holder
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        print(f"Loading trained CNN model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully into memory.")
    return model

def preprocess_image(pil_image):
    """
    Preprocess image exactly matching Modeltrain.ipynb:
    - Resize to (30, 30)
    - Convert to RGB
    - Convert to numpy array of uint8/float (0-255)
    - Add batch dimension: shape (1, 30, 30, 3)
    """
    img = pil_image.convert("RGB")
    img = img.resize((30, 30))
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return all 43 classes with metadata and icon URLs"""
    class_list = []
    for class_id, info in SIGN_CLASSES.items():
        icon_path = f"/static/icons/{class_id}.png"
        class_list.append({
            "class_id": class_id,
            "name": info["name"],
            "category": info["category"],
            "badge_color": info["badge_color"],
            "action": info["action"],
            "risk_level": info["risk_level"],
            "icon_url": icon_path
        })
    return jsonify({"classes": class_list, "total": len(class_list)})

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return curated test sample images for quick testing"""
    manifest_path = os.path.join(SAMPLES_DIR, "samples_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for item in manifest:
            cid = item["class_id"]
            item["name"] = SIGN_CLASSES.get(cid, {}).get("name", "Unknown")
            item["category"] = SIGN_CLASSES.get(cid, {}).get("category", "General")
            item["image_url"] = f"/static/samples/{item['filename']}"
            item["icon_url"] = f"/static/icons/{cid}.png"
        return jsonify({"samples": manifest})
    return jsonify({"samples": []})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        current_model = get_model()
        pil_img = None

        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            pil_img = Image.open(file.stream)
        elif request.is_json:
            data = request.get_json()
            if "image" not in data:
                return jsonify({"error": "Missing 'image' key in JSON payload"}), 400
            image_data = data["image"]
            if "," in image_data:
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            pil_img = Image.open(io.BytesIO(image_bytes))
        else:
            return jsonify({"error": "No valid image provided via form-data or JSON"}), 400

        # Run inference
        input_tensor = preprocess_image(pil_img)
        predictions = current_model.predict(input_tensor, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1][:5]
        predicted_class_id = int(top_indices[0])
        confidence = float(predictions[predicted_class_id])

        top_5 = []
        for idx in top_indices:
            cid = int(idx)
            info = SIGN_CLASSES.get(cid, {
                "name": f"Class {cid}",
                "category": "Unknown",
                "badge_color": "#94a3b8",
                "action": "Proceed with standard caution.",
                "risk_level": "Moderate"
            })
            top_5.append({
                "class_id": cid,
                "name": info["name"],
                "category": info["category"],
                "confidence": float(predictions[cid]),
                "confidence_percent": round(float(predictions[cid]) * 100, 2),
                "icon_url": f"/static/icons/{cid}.png"
            })

        best_info = SIGN_CLASSES.get(predicted_class_id, {
            "name": f"Sign #{predicted_class_id}",
            "category": "Unknown",
            "badge_color": "#00e5ff",
            "action": "Proceed with regular traffic awareness.",
            "risk_level": "Moderate"
        })

        return jsonify({
            "success": True,
            "prediction": {
                "class_id": predicted_class_id,
                "name": best_info["name"],
                "category": best_info["category"],
                "badge_color": best_info["badge_color"],
                "action": best_info["action"],
                "risk_level": best_info["risk_level"],
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 2),
                "icon_url": f"/static/icons/{predicted_class_id}.png"
            },
            "top_5": top_5,
            "model_metadata": {
                "architecture": "Deep CNN (4 Conv2D + 2 MaxPool2D + 3 Dropout + 2 Dense)",
                "input_shape": [30, 30, 3],
                "num_classes": 43,
                "framework": "TensorFlow / Keras"
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(SIGN_CLASSES)
    })

if __name__ == "__main__":
    # Preload model on startup
    get_model()
    print("=" * 60)
    print(" Traffic Sign Recognition AI Dashboard is starting...")
    print(" Open browser at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
