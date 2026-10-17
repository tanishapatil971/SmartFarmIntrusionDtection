from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from collections import Counter
import numpy as np, io, os, time

app = Flask(__name__)
default = Path("runs/detect/runs/smartfarm/animals_cpu/weights/best.pt")
weights = Path(os.environ.get("SMARTFARM_MODEL", str(default)))
if not weights.is_absolute():
    weights = (Path(__file__).resolve().parent / weights).resolve()
if not weights.exists() and (Path(__file__).resolve().parent / "best.pt").exists():
    weights = Path(__file__).resolve().parent / "best.pt"
if not weights.exists():
    raise FileNotFoundError(f"Model not found at {weights}. Copy best.pt beside app.py or set SMARTFARM_MODEL.")
print(f"Loading YOLO model: {weights}")
model = YOLO(str(weights))
frame_count = 0
@app.get("/")
def index():
    return render_template("index.html")
@app.get("/health")
def health():
    return jsonify({"status":"ok","model":weights.name,"frames_processed":frame_count})
@app.post("/detect")
def detect():
    global frame_count
    if "frame" not in request.files:
        return jsonify({"error":"No frame uploaded"}), 400
    image = Image.open(io.BytesIO(request.files["frame"].read())).convert("RGB")
    arr = np.asarray(image)
    t = time.perf_counter()
    result = model.predict(arr, imgsz=416, conf=float(request.form.get("conf", .40)), device="cpu", verbose=False)[0]
    latency = (time.perf_counter()-t)*1000
    detections = []
    if result.boxes is not None:
        for b in result.boxes:
            cid = int(b.cls[0].item())
            name = model.names[cid] if isinstance(model.names, dict) else model.names[cid]
            detections.append({"label":str(name),"confidence":float(b.conf[0].item()),"xyxy":[float(v) for v in b.xyxy[0].tolist()]})
    frame_count += 1
    return jsonify({"width":image.width,"height":image.height,"detections":detections,
                    "objects":len(detections),"latency_ms":round(latency,1),"frames_processed":frame_count,
                    "counts":dict(Counter(d["label"] for d in detections))})
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
