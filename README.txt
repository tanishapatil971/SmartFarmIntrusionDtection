# 🌱 Smart Farm Intrusion Detection System

An AI-powered computer vision prototype that detects people and selected animals in farm camera images using **YOLO11n**. The project integrates a trained object detection model with a Flask backend and a web dashboard for live webcam visualization.

## 📌 Project Overview

Traditional farm monitoring can require continuous human supervision. This project explores how object detection can support farm surveillance by identifying objects, displaying bounding boxes, and reporting confidence scores.

The prototype processes webcam frames on a laptop and displays detection results through a browser-based dashboard.

> **Note:** This is a software prototype. It has not yet been deployed on embedded hardware, and detections are not guaranteed to be correct in every live scene.

## ✨ Features

- **Multi-object detection:** Detects multiple objects in an image.
- **10 object classes:** Cat, chicken, cow, dog, fox, goat, horse, person, raccoon, and skunk.
- **Live webcam demonstration:** Processes camera frames through the trained model.
- **Bounding boxes and confidence scores:** Visualizes predicted classes and object locations.
- **Web dashboard:** Displays detection results and inference information.
- **Model evaluation:** Reports standard object detection metrics on a held-out test set.
- **CPU inference:** Runs without requiring a dedicated NVIDIA GPU.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Object detection | YOLO11n |
| Programming language | Python |
| Deep learning framework | PyTorch |
| Detection library | Ultralytics |
| Backend | Flask |
| Image processing | Pillow, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Dataset | Roboflow-100 — Animals dataset |

## 📂 Dataset

The project uses the Animals dataset from Roboflow-100, licensed under **CC BY 4.0**.

| Split | Images |
|---|---:|
| Training | 700 |
| Validation | 200 |
| Testing | 100 |
| **Total** | **1,000** |

**Classes:** cat, chicken, cow, dog, fox, goat, horse, person, raccoon, and skunk.

The dataset contains general animal and person examples; it is not exclusively a farm-intrusion dataset.

## 🧠 Model Details

- Model: YOLO11n
- Parameters: approximately 2.6 million
- Computational complexity: approximately 6.4 GFLOPs
- Inference device: CPU
- Input size: depends on the inference configuration

## 📊 Recorded Test Results

The following results were recorded on the held-out test set of 100 images containing 157 annotated objects.

| Metric | Result |
|---|---:|
| Precision | 81.6% |
| Recall | 71.4% |
| mAP@50 | 80.3% |
| mAP@50–95 | 62.3% |

- **Precision** measures the proportion of predicted detections that are correct.
- **Recall** measures the proportion of annotated objects that are detected.
- **mAP@50** evaluates mean average precision at an IoU threshold of 0.50.
- **mAP@50–95** evaluates mean average precision across IoU thresholds from 0.50 to 0.95.

These are test-set metrics, not a guarantee of live-camera performance. Class confusion, missed detections, lighting, and scene conditions can affect webcam results.

## ⚙️ Project Architecture

```text
Webcam / Image
      |
      v
Flask Backend
      |
      v
YOLO11n Model
      |
      v
Object Detection
(Class + Confidence + Bounding Box)
      |
      v
Web Dashboard
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- pip
- A webcam for live detection
- Git
- A trained YOLO model file (`best.pt`)

### 1. Clone the repository

```bash
git clone https://github.com/tanishapatil971/SmartFarmIntrusionDtection.git
cd SmartFarmIntrusionDetection
```

If your GitHub repository has a different name, replace the URL and folder name accordingly.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If your environment does not already contain PyTorch and the Ultralytics dependencies, ensure they are installed according to the requirements file and your system configuration.

### 4. Configure model weights

Place the trained `best.pt` file at:

```text
runs/detect/runs/smartfarm/animals_cpu/weights/best.pt
```

Alternatively, place `best.pt` beside `app.py`, or set the `SMARTFARM_MODEL` environment variable to the model's path.

### 5. Start the application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

Allow camera access and start the live detection demo from the dashboard.

Keep the terminal running while using the application.

## 🔍 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Opens the dashboard |
| `/health` | GET | Checks server and model status |
| `/detect` | POST | Accepts a camera frame and returns detections |

The `/detect` endpoint expects an uploaded image under the `frame` form field.

## ⚠️ Limitations

- Some visually similar classes may be confused.
- Confidence scores do not guarantee correct classifications.
- Live performance depends on lighting, camera quality, and scene conditions.
- CPU inference can introduce noticeable latency.
- The prototype has not been deployed or evaluated on Raspberry Pi, NVIDIA Jetson, or other embedded hardware.
- It does not independently determine whether a detected person is authorized to enter a farm.

## 🔮 Future Scope

- Expand the dataset with farm-specific and night-time images.
- Improve class-level detection performance and prediction stability.
- Add restricted-zone rules and configurable alerts.
- Evaluate inference on embedded hardware.
- Optimize the model for latency and resource usage.
- Test the system in real farm environments.

## 🎯 Applications

- Farm camera surveillance
- Livestock observation
- Agricultural area monitoring
- Computer vision experimentation
- Edge AI and embedded AI research

## 👩‍💻 Author

**Tanisha Patil**

B.Tech — Computer Science and Engineering (AI & ML)

GitHub: [@tanishapatil971](https://github.com/tanishapatil971)

## 📄 License and Attribution

The Animals dataset is licensed under Creative Commons Attribution 4.0. Follow the dataset provider's attribution requirements when redistributing dataset material.

The project source code and model weights may have separate licensing or redistribution terms; check the relevant dependencies, dataset terms, and training resources before redistributing them.

---

*Built as an Embedded AI lab project exploring object detection for smart agriculture.*
