from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import DEFAULT_MODEL_PATH, file_exists


class NumberPlateDetector:
    """Thin wrapper around Ultralytics YOLO for number plate detection."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        if not file_exists(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found at {self.model_path}. Upload best.pt or place it in models/best.pt."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install ultralytics with: pip install ultralytics") from exc

        self.model: Any = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray, conf: float) -> list[dict]:
        """Return YOLO detections as dictionaries containing box, confidence, and class id."""
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to detector.")

        results = self.model.predict(image, conf=conf, verbose=False)
        detections: list[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < conf:
                    continue
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                class_id = int(box.cls[0]) if box.cls is not None else 0
                detections.append(
                    {
                        "box": xyxy,
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )
        return detections
