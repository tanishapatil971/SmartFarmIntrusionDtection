from pathlib import Path
import yaml
from ultralytics import YOLO

DATA_YAML = Path("dataset/animals-ij5d2/data.yaml")

def main():
    if not DATA_YAML.exists():
        raise SystemExit(
            "Dataset not found. First run: python download_dataset.py"
        )
    with DATA_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Dataset classes:", cfg.get("names", "check data.yaml"))
    print("Training YOLO11n for a short baseline run. This may take time on CPU.")
    model = YOLO("yolo11n.pt")
    model.train(data=str(DATA_YAML), epochs=15, imgsz=640, batch=4, workers=2,
                project="runs", name="smartfarm_custom", patience=5)
    print("Training finished. Look for runs/smartfarm_custom/weights/best.pt")

if __name__ == "__main__":
    main()
