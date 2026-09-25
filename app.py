import os
import io
import json
import subprocess
import threading
import time
from flask import Flask, request, jsonify, render_template, send_file, abort
from werkzeug.utils import secure_filename
from PIL import Image
from model_inference import classifier
from agronomic_db import AGRONOMIC_DB, get_agronomic_analysis

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Curated list of sample classes across all 5 major crop families
SAMPLE_CATEGORIES = [
    # Rice
    "Rice Blast",
    "Becterial Blight in Rice",
    "Brownspot",
    "Tungro",
    # Cotton
    "American Bollworm on Cotton",
    "cotton whitefly",
    "cotton mealy bug",
    "Leaf Curl",
    "Healthy cotton",
    # Wheat
    "Wheat___Yellow_Rust",
    "Wheat black rust",
    "Wheat powdery mildew",
    "Wheat scab",
    "Healthy Wheat",
    # Maize
    "maize fall armyworm",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "maize ear rot",
    "Healthy Maize",
    # Sugarcane
    "RedRot sugarcane",
    "Mosaic sugarcane",
    "Yellow Rust Sugarcane",
    "Sugarcane Healthy"
]

training_lock = threading.Lock()
training_process = None

def run_training_subprocess(epochs=10):
    global training_process
    try:
        cmd = ["python3", "train.py", "--epochs", str(epochs)]
        training_process = subprocess.Popen(cmd)
        training_process.wait()
    except Exception as e:
        print(f"Training subprocess error: {e}")
    finally:
        training_process = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples")
def get_samples():
    """Return a curated gallery of sample images from the dataset categorized by crop."""
    samples = []
    base_dir = "Validation" if os.path.exists("Validation") else "Train"
    
    for category in SAMPLE_CATEGORIES:
        cat_dir = os.path.join(base_dir, category)
        if os.path.isdir(cat_dir):
            files = [f for f in sorted(os.listdir(cat_dir)) if not f.startswith(".") and any(f.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)]
            if files:
                rel_path = f"{base_dir}/{category}/{files[0]}"
                display_info = get_agronomic_analysis(category)
                samples.append({
                    "class_name": category,
                    "crop": display_info.get("crop", "Crop"),
                    "display_name": display_info.get("display_name", category),
                    "severity": display_info.get("severity", "Moderate"),
                    "category": display_info.get("category", "Condition"),
                    "image_url": f"/api/sample_image?path={rel_path}",
                    "sample_path": rel_path
                })
    return jsonify({"samples": samples})

@app.route("/api/sample_image")
def get_sample_image():
    """Serve a sample image safely from Train or Validation."""
    raw_path = request.args.get("path", "")
    norm_path = os.path.normpath(raw_path)
    if not (norm_path.startswith("Validation/") or norm_path.startswith("Train/")):
        abort(403)
    if not os.path.exists(norm_path):
        abort(404)
    return send_file(norm_path)

@app.route("/api/model_info")
def model_info():
    """Return model status, training history, and metrics."""
    history = {}
    if os.path.exists("training_history.json"):
        try:
            with open("training_history.json", "r") as f:
                history = json.load(f)
        except Exception:
            pass
            
    is_trained = os.path.exists("best_crop_model.pth")
    return jsonify({
        "trained": is_trained,
        "architecture": "MobileNetV3-Large (Transfer Learning)",
        "num_classes": len(classifier.class_names),
        "device": str(classifier.device),
        "history": history
    })

@app.route("/api/training_status")
def training_status():
    """Return real-time training progress or completed history."""
    status = {"is_training": False}
    if os.path.exists("training_status.json"):
        try:
            with open("training_status.json", "r") as f:
                status = json.load(f)
        except Exception:
            pass
            
    history = {}
    if os.path.exists("training_history.json"):
        try:
            with open("training_history.json", "r") as f:
                history = json.load(f)
        except Exception:
            pass
            
    status["history_data"] = history
    status["device"] = str(classifier.device)
    return jsonify(status)

@app.route("/api/start_training", methods=["POST"])
def start_training():
    """Trigger 10-epoch training in the background."""
    global training_process
    with training_lock:
        if training_process is not None and training_process.poll() is None:
            return jsonify({"status": "already_running", "message": "10-epoch training is already actively running in the background!"})
            
        data = request.get_json(silent=True) or {}
        epochs = int(data.get("epochs", 10))
        
        thread = threading.Thread(target=run_training_subprocess, args=(epochs,), daemon=True)
        thread.start()
        
        return jsonify({"status": "started", "epochs": epochs, "message": f"Training initiated for {epochs} epochs on {classifier.device}!"})

@app.route("/api/predict", methods=["POST"])
def predict():
    """Handle crop image upload or sample path analysis."""
    try:
        image = None
        
        json_data = request.get_json(silent=True) or {}
        sample_path = request.form.get("sample_path") or json_data.get("sample_path")
        if sample_path:
            norm_path = os.path.normpath(sample_path)
            if not (norm_path.startswith("Validation/") or norm_path.startswith("Train/")):
                return jsonify({"error": "Invalid sample path"}), 400
            if not os.path.exists(norm_path):
                return jsonify({"error": "Sample image not found"}), 404
            image = Image.open(norm_path)
            
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(file.filename):
                return jsonify({"error": "Unsupported file format. Please upload JPG, PNG, or WEBP."}), 400
            image = Image.open(file.stream)
            
        else:
            return jsonify({"error": "No image uploaded or sample provided"}), 400

        result = classifier.predict(image)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
