from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "best.pt"
DB_PATH = DATA_DIR / "logs.db"


def ensure_directories() -> None:
    """Create runtime folders used by uploads, annotated frames, and the database."""
    for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def unique_path(directory: Path, suffix: str, prefix: str = "file") -> Path:
    """Return a collision-resistant file path in the selected directory."""
    ensure_directories()
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}{safe_suffix}"


def save_uploaded_file(uploaded_file, directory: Path = UPLOAD_DIR) -> Path:
    """Persist a Streamlit uploaded file and return its local path."""
    suffix = Path(uploaded_file.name).suffix or ".bin"
    output_path = unique_path(directory, suffix, Path(uploaded_file.name).stem or "upload")
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def read_image(path: str | Path) -> np.ndarray:
    """Read an image with OpenCV and raise a useful error if it cannot be decoded."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def crop_box(image: np.ndarray, box: Iterable[int], padding: int = 4) -> np.ndarray:
    """Crop a bounding box from an image with optional padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2]


def draw_detections(image: np.ndarray, detections: list[dict], color=(20, 180, 80)) -> np.ndarray:
    """Draw number plate boxes and optional labels on a copy of the image."""
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        label = detection.get("text") or f"{detection.get('confidence', 0):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR images for Streamlit display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, prefix: str = "frame") -> Path:
    """Save an OpenCV image to the outputs folder."""
    path = unique_path(OUTPUT_DIR, ".jpg", prefix)
    cv2.imwrite(str(path), image)
    return path


def normalize_camera_source(source: str):
    """Convert numeric camera sources to int while preserving RTSP/HTTP camera URLs."""
    source = str(source).strip()
    if re.fullmatch(r"\d+", source):
        return int(source)
    return source


def file_exists(path: str | Path | None) -> bool:
    """Return true when a path points to an existing file."""
    return bool(path) and Path(path).exists() and Path(path).is_file()


def relative_or_absolute(path: str | Path | None) -> str | None:
    """Store friendly paths in SQLite without failing for external absolute paths."""
    if not path:
        return None
    try:
        return os.path.relpath(str(path), PROJECT_ROOT)
    except ValueError:
        return str(path)
