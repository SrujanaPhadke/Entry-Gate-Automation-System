# Entry Gate Automation System

An intelligent vehicle entry/exit monitoring system that detects number plates with Ultralytics YOLO, extracts vehicle numbers with OCR, and stores entry/exit logs in SQLite.

## Features

- Image input mode for uploaded vehicle photos
- Video input mode for recorded gate footage
- Live camera mode for webcam or IP camera streams
- YOLO number plate detection using `models/best.pt` or an uploaded `.pt` file
- EasyOCR-based plate text extraction
- Indian vehicle number cleaning and normalization, such as `MH12AB1234`, `DL01CA1234`, and `GJ05XY9876`
- Entry/exit logging with duplicate filtering and configurable cooldown
- Searchable logs table with status filter
- CSV export for logs and video detection reports
- Custom YOLO training support through Ultralytics

## Project Structure

```text
entry-gate-automation/
  app.py
  train.py
  requirements.txt
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

## Technology Stack

- Python
- OpenCV
- Ultralytics YOLOv8 or YOLO11
- EasyOCR
- Streamlit
- SQLite
- Pandas

## Setup

Open PowerShell in the project folder:

```powershell
cd "D:\AIML Internship\entry-gate-automation"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the application:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## YOLO Model Weights

The application loads the default model from:

```text
models/best.pt
```

You can also upload a custom `.pt` model from the Streamlit sidebar. For best gate accuracy, train a model using your own camera angle, lighting, plate distance, and vehicle types.

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

## Entry/Exit Logic

- If a detected vehicle has no active `inside` record, the system creates an entry log.
- If the same vehicle is detected again after the configured cooldown and its status is `inside`, the system records `exit_time` and marks it `exited`.
- Repeated detections inside the duplicate window are ignored to prevent noisy logs.

## Image Mode Testing

1. Select `Image Detection`.
2. Upload a clear image of a vehicle.
3. The dashboard displays the original image, YOLO bounding boxes, cropped number plate, OCR text, and log action.

## Video Mode Testing

1. Select `Video Detection`.
2. Upload a video file such as `.mp4`, `.avi`, `.mov`, or `.mkv`.
3. Keep `Analyze every Nth frame` at `1` for full frame-by-frame processing.
4. Click `Process Video`.
5. The dashboard displays total frames, frames analyzed, YOLO detections, readable plates, frame-by-frame results, sample frames, and processed video preview.

## Live Camera Testing

1. Select `Live Camera Detection`.
2. Use camera source `0` for your webcam, or paste an IP camera URL.
3. Set the number of frames to process.
4. Click `Start Camera`.
5. The dashboard shows the live annotated feed and entry/exit events.

## Logs

1. Select `Logs`.
2. Search by vehicle number.
3. Filter by `all`, `inside`, or `exited`.
4. Click `Download CSV` to export logs.

## Train a Custom YOLO Number Plate Model

Prepare your dataset in Ultralytics YOLO format:

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

Train with the Ultralytics CLI:

```powershell
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Or use the included training helper:

```powershell
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --copy-best
```

YOLO11 example:

```powershell
python train.py --data dataset/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --copy-best
```

After training, copy:

```text
runs/detect/train/weights/best.pt
```

or the helper run output:

```text
runs/detect/number_plate/weights/best.pt
```

to:

```text
models/best.pt
```

Then restart:

```powershell
streamlit run app.py
```

## Notes for Better Accuracy

- Use high-resolution frames where the plate is visible and not blurred.
- Keep `Analyze every Nth frame` at `1` for best video detection.
- Lower YOLO confidence if plates are missed.
- Lower OCR confidence slightly if YOLO detects crops but text is not extracted.
- Train your own model for your exact gate camera angle and lighting.
