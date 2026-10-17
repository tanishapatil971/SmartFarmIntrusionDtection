SMART FARM MULTI-OBJECT LIVE DETECTION — QUICK START
====================================================

This package provides a fast live-camera demo using an Ultralytics pretrained
YOLO11n model, plus a separate script to download the Animals Ij5D2 dataset.

IMPORTANT HONESTY NOTE:
- The live demo initially uses yolo11n.pt pretrained on COCO. It can detect
  multiple objects in the same frame, including common classes such as person,
  dog, cat, cow, horse, sheep and bird.
- It is NOT trained on the Animals Ij5D2 dataset until you run train_custom.py.
- The dataset contains 1,000 annotated images across 10 classes. Dataset card:
  https://huggingface.co/datasets/LibreYOLO/animals-ij5d2
- Do not claim the pretrained model achieved a custom 95% accuracy. For object
  detection, report test-set precision/recall/mAP from a proper labelled test set.

1) Open this folder in VS Code.
2) Open a terminal in this folder.
3) Create/activate your virtual environment if you already have one.
4) Install:
      pip install -r requirements.txt

5) Start live webcam detection:
      python live_detect.py --source 0

   Press Q in the video window to stop. The first run downloads yolo11n.pt.
   Webcam availability depends on your camera and permissions.

6) Run detection on a video:
      python live_detect.py --source "test_video.mp4"

7) Run detection on one image:
      python live_detect.py --source "test_images/farm.jpg"

8) Change which classes trigger an intrusion alert:
      python live_detect.py --source 0 --intruders person,dog

   Alert classes must actually be supported by the chosen model. The default
   COCO model does not know every wildlife species in the custom dataset.

9) Logs:
   outputs/detections.csv records timestamp, detected classes, count, alert,
   average confidence, inference time, and an approximate per-frame FPS value.
   FPS here is calculated as 1000 / model inference time; it is not guaranteed
   to equal the full camera display frame rate.

10) Download and zip the annotated custom dataset:
      python download_dataset.py

   It creates:
      dataset/animals-ij5d2/
      dataset/animals-ij5d2.zip

11) Optional custom training (may be slow on CPU):
      python train_custom.py

   After training, run the custom model:
      python live_detect.py --source 0 --model runs/smartfarm_custom/weights/best.pt

DATASET DETAILS
---------------
Animals Ij5D2 is a multi-class object-detection dataset with train/validation/
test splits (700/200/100 images) and classes:
cat, chicken, cow, dog, fox, goat, horse, person, racoon, skunk.

The downloaded dataset archive is not bundled in this starter ZIP because it
must be retrieved from the dataset host at run time. The downloader above
fetches it and creates a ZIP locally. This keeps this project ZIP small and
avoids pretending an external dataset was downloaded when it was not.

For your presentation, distinguish:
- Live demonstration (pretrained model detecting multiple objects)
- Custom dataset training (only after training is actually completed)
- Evaluation results (only values produced by running validation/test evaluation)
