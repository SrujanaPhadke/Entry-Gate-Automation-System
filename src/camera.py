from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np

from .database import VehicleLogDB
from .detector import NumberPlateDetector
from .ocr import OCRResult, PlateOCR
from .utils import crop_box, draw_detections, normalize_camera_source, save_image


class LiveCameraProcessor:
    """Handle webcam or IP camera number plate detection with buffered OCR."""

    def __init__(self, detector: NumberPlateDetector, ocr: PlateOCR, db: VehicleLogDB):
        self.detector = detector
        self.ocr = ocr
        self.db = db

    def open_capture(self, source: str = "0") -> cv2.VideoCapture:
        """Open a local webcam index or an IP camera URL."""
        capture = cv2.VideoCapture(normalize_camera_source(source))
        if not capture.isOpened():
            raise ValueError(f"Camera not found or cannot be opened: {source}")
        return capture

    @staticmethod
    def sharpness(image: np.ndarray) -> float:
        """Measure crop sharpness using Laplacian variance."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def select_best_plate(candidates: list[dict]) -> dict | None:
        """Choose the sharpest plate crop from the recent frame buffer."""
        if not candidates:
            return None
        return max(candidates, key=lambda item: item["sharpness"])

    def record_ocr_result(
        self,
        frame: np.ndarray,
        detection: dict,
        ocr_result: OCRResult,
        cooldown_seconds: int,
        duplicate_window_seconds: int,
        recent_cache: dict[str, datetime],
    ) -> tuple[dict | None, dict[str, datetime]]:
        """Apply duplicate filtering and persist a successful live OCR result."""
        plate_number = ocr_result.text
        now = datetime.now()
        if plate_number in recent_cache and now - recent_cache[plate_number] < timedelta(seconds=duplicate_window_seconds):
            return None, recent_cache

        recent_cache[plate_number] = now
        detection["text"] = plate_number
        frame_path = save_image(draw_detections(frame, [detection]), "live_frame")
        action, log_id = self.db.record_detection(
            plate_number,
            "live",
            frame_path,
            detection["confidence"],
            cooldown_seconds=cooldown_seconds,
            duplicate_window_seconds=duplicate_window_seconds,
        )
        if action == "duplicate":
            return None, recent_cache

        return (
            {
                "vehicle_number": plate_number,
                "action": action,
                "log_id": log_id,
                "confidence": detection["confidence"],
                "ocr_confidence": ocr_result.confidence,
                "ocr_source": ocr_result.source,
                "sharpness": round(float(detection.get("sharpness", 0.0)), 2),
            },
            recent_cache,
        )

    def run_capture(
        self,
        source: str,
        conf_threshold: float,
        ocr_threshold: float,
        cooldown_seconds: int = 60,
        duplicate_window_seconds: int = 10,
        max_frames: int | None = None,
        buffer_size: int = 8,
        process_every_n_frames: int = 3,
        stability_delay: float = 0.03,
    ):
        """Yield annotated frames and live OCR status from a buffered camera stream."""
        capture = self.open_capture(source)
        recent_cache: dict[str, datetime] = {}
        crop_buffer: deque[dict] = deque(maxlen=max(1, buffer_size))
        count = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                detections = []
                status = {
                    "plate_detected": False,
                    "ocr_failed": False,
                    "best_text": None,
                    "best_crop": None,
                    "processed_crop": None,
                    "best_sharpness": None,
                    "buffer_count": len(crop_buffer),
                }
                should_process = count % max(1, process_every_n_frames) == 0

                if should_process:
                    detections = self.detector.detect(frame, conf=conf_threshold)
                    for detection in detections:
                        plate_crop = crop_box(frame, detection["box"])
                        crop_buffer.append(
                            {
                                "crop": plate_crop,
                                "frame": frame.copy(),
                                "detection": detection.copy(),
                                "sharpness": self.sharpness(plate_crop),
                                "frame_index": count,
                            }
                        )

                    status["plate_detected"] = bool(detections or crop_buffer)
                    status["buffer_count"] = len(crop_buffer)
                    best = self.select_best_plate(list(crop_buffer))

                    events = []
                    if best:
                        best_crop = best["crop"]
                        ocr_result, variants = self.ocr.read_plate_with_debug(best_crop, min_confidence=ocr_threshold)
                        status["best_crop"] = best_crop
                        status["processed_crop"] = variants.get("threshold") if variants else None
                        status["best_sharpness"] = round(best["sharpness"], 2)

                        best_detection = best["detection"].copy()
                        best_detection["sharpness"] = best["sharpness"]
                        if ocr_result:
                            status["best_text"] = ocr_result.text
                            best_detection["text"] = ocr_result.text
                            event, recent_cache = self.record_ocr_result(
                                best["frame"],
                                best_detection,
                                ocr_result,
                                cooldown_seconds,
                                duplicate_window_seconds,
                                recent_cache,
                            )
                            if event:
                                events.append(event)
                        else:
                            status["ocr_failed"] = True
                    else:
                        events = []
                else:
                    events = []

                annotated = draw_detections(frame, detections)
                yield annotated, events, status
                count += 1

                if stability_delay > 0:
                    time.sleep(stability_delay)
                if max_frames and count >= max_frames:
                    break
        finally:
            capture.release()
