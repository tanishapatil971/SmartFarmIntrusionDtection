SMART FARM MODEL CONNECTION — QUICK START

1. Extract the ZIP contents into your existing project folder:
C:\Users\Admin\Downloads\SmartFarm_MultiObject_Live_Demo\SmartFarm_MultiObject_Live_Demo

You should have app.py, requirements.txt, templates\index.html in that folder.
Do not delete your existing dataset or runs folders.

2. Confirm the model exists here:
runs\detect\runs\smartfarm\animals_cpu\weights\best.pt

If the model is elsewhere, the easiest option is to copy best.pt beside app.py.
Alternatively, in PowerShell set:
$env:SMARTFARM_MODEL="C:\full\path\to\best.pt"

3. Open PowerShell in the project folder. Activate your existing venv if you have one, then run:
python -m pip install -r requirements.txt
python app.py

4. Wait for Flask to say it is running, then open Chrome:
http://127.0.0.1:5000

5. Click Start camera + AI and allow camera access. The dashboard should show real YOLO boxes, labels, confidence scores and live events.

IMPORTANT:
- Do not open templates\index.html directly. Use the localhost URL.
- Keep the PowerShell window running while demonstrating.
- This is a local lab demo; do not expose the Flask development server to the internet.
- If CPU inference is slow, change setInterval(process,450) to setInterval(process,700) in templates\index.html.
