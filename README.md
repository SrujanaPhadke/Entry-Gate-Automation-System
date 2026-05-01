# Entry Gate Automation System

AI-powered gate monitoring with YOLO number plate detection, OCR text extraction, and clean entry/exit logs.

## Live Demo

Render URL:

```text
https://entry-gate-automation.onrender.com
```

If the URL shows Render's `Not Found` page, create/connect the Render web service using the settings in the **Deploy On Render** section below. The codebase already includes `render.yaml`, `runtime.txt`, and a Render-safe `requirements.txt`.

## What This System Does

This project turns a normal gate camera workflow into an automated vehicle logging system:

- Detects number plates with Ultralytics YOLO
- Extracts text with EasyOCR
- Supports image upload, recorded video, and live camera input
- Saves entry and exit records in SQLite
- Filters duplicate detections with a cooldown window
- Exports logs and video detection reports as CSV
- Supports custom YOLO weights through `models/best.pt` or sidebar upload

## Core Modes

### Image Detection

Upload a vehicle image and get:

- Original image preview
- YOLO bounding boxes
- Cropped plate regions
- Enhanced OCR debug crops
- Extracted vehicle number
- Entry/exit log action

### Video Detection

Upload recorded footage and get:

- Frame-by-frame YOLO plate detection
- OCR output per detected plate
- Duplicate filtering
- Detection metrics
- Annotated sample frames
- Processed video preview
- CSV detection report

### Live Camera Detection

Use webcam source `0` or an IP/RTSP camera URL and get:

- Real-time plate detection
- Buffered crop selection for sharper OCR
- Best live crop preview
- Entry and exit log updates
- Duplicate detection protection

## Tech Stack

- Python 3.10
- Streamlit
- Ultralytics YOLO
- EasyOCR
- OpenCV headless
- Torch / Torchvision
- SQLite
- Pandas

## Project Structure

```text
entry-gate-automation/
  app.py
  train.py
  requirements.txt
  runtime.txt
  render.yaml
  README.md
  models/
    best.pt
  src/
    detector.py
    ocr.py
    database.py
    video_processor.py
    camera.py
    utils.py
  data/
    uploads/
    outputs/
    logs.db
```

## Run Locally

```powershell
cd "D:\AIML Internship\entry-gate-automation"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Deploy On Render

Create a new Render service with:

```text
Service Type: Web Service
Environment: Python
Branch: main
```

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Expected live URL:

```text
https://entry-gate-automation.onrender.com
```

The repository also includes `render.yaml`, so Render can read the same configuration from the repo.

## YOLO Model

The default model path is:

```text
models/best.pt
```

The included model is small enough for GitHub and Render deployment. For better accuracy, train your own model using images from the actual gate camera angle and lighting.

## Database

The app creates `data/logs.db` automatically.

```sql
CREATE TABLE vehicle_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  vehicle_number TEXT NOT NULL,
  entry_time TEXT NOT NULL,
  exit_time TEXT,
  source_type TEXT NOT NULL,
  image_path TEXT,
  confidence_score REAL,
  status TEXT NOT NULL CHECK(status IN ('inside', 'exited'))
);
```

## Entry And Exit Logic

- First detection of a vehicle creates an `inside` entry.
- Re-detection after the cooldown updates `exit_time` and marks the vehicle `exited`.
- Repeated detections inside the duplicate window are ignored.

## Train A Custom YOLO Plate Model

Dataset format:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Example `dataset/data.yaml`:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: number_plate
```

Train with Ultralytics:

```powershell
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Or use the helper:

```powershell
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --copy-best
```

Copy the trained weights to:

```text
models/best.pt
```

Then restart:

```powershell
streamlit run app.py
```

## Accuracy Tips

- Use clear, close plate images.
- Keep `Analyze every Nth frame` at `1` for video accuracy.
- Lower YOLO confidence if plates are missed.
- Raise OCR confidence to reduce weak OCR guesses.
- Train a custom YOLO model for your camera angle.

## Repository

```text
https://github.com/SrujanaPhadke/Entry-Gate-Automation-System.git
```
