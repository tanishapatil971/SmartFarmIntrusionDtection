import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

INTRUSION_CLASSES_DEFAULT = {"person", "dog", "fox", "raccoon", "racoon", "skunk"}

def main():
    parser = argparse.ArgumentParser(description="Smart Farm multi-object live detector")
    parser.add_argument("--source", default="0", help="Camera index (0) or path to image/video")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model/weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--log", default="outputs/detections.csv", help="CSV log path")
    parser.add_argument("--intruders", default="person,dog,fox,raccoon,racoon,skunk",
                        help="Comma-separated class names that should trigger an alert")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    intruders = {x.strip().lower() for x in args.intruders.split(",") if x.strip()}

    is_camera = isinstance(source, int)
    cap = cv2.VideoCapture(source) if is_camera or (isinstance(source, str) and Path(source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}) else None

    if cap is not None:
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera/video source: {source}")
        csv_file = open(log_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if log_path.stat().st_size == 0:
            writer.writerow(["timestamp", "frame", "detected_objects", "object_count",
                             "intrusion_alert", "average_confidence_percent", "inference_ms", "fps"])
        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                start = time.perf_counter()
                result = model.predict(frame, conf=args.conf, verbose=False)[0]
                inference_ms = (time.perf_counter() - start) * 1000
                names = model.names
                detected = []
                confidences = []
                intrusion_found = False

                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        name = str(names[cls_id])
                        detected.append(f"{name}:{confidence:.2f}")
                        confidences.append(confidence)
                        if name.lower() in intruders:
                            intrusion_found = True

                annotated = result.plot()
                status = "INTRUSION ALERT" if intrusion_found else "No configured intruder detected"
                status_color = (0, 0, 255) if intrusion_found else (0, 150, 0)
                cv2.rectangle(annotated, (8, 8), (annotated.shape[1]-8, 58), (25, 25, 25), -1)
                cv2.putText(annotated, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                fps = 1000.0 / inference_ms if inference_ms > 0 else 0
                avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0
                cv2.putText(annotated, f"Objects: {len(detected)} | Inference: {inference_ms:.1f} ms | FPS*: {fps:.1f}",
                            (18, annotated.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow("Smart Farm Intrusion Detection - press Q to quit", annotated)

                writer.writerow([datetime.now().isoformat(timespec="seconds"), frame_id,
                                 "; ".join(detected), len(detected), int(intrusion_found),
                                 round(avg_conf, 2), round(inference_ms, 2), round(fps, 2)])
                csv_file.flush()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            csv_file.close()
            cv2.destroyAllWindows()
    else:
        result = model.predict(source=source, conf=args.conf, save=True, verbose=False)[0]
        names = model.names
        detected = []
        intrusion_found = False
        confidences = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = str(names[cls_id])
                detected.append(f"{name}:{conf:.2f}")
                confidences.append(conf)
                intrusion_found |= name.lower() in intruders
        print("Detected:", ", ".join(detected) if detected else "no objects")
        print("Intrusion alert:", "YES" if intrusion_found else "NO")
        print("Average confidence:", f"{sum(confidences)/len(confidences)*100:.2f}%" if confidences else "N/A")
        print("Annotated output is saved by Ultralytics in runs/detect/predict/")

if __name__ == "__main__":
    main()
