from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2

from .database import VehicleLogDB
from .detector import NumberPlateDetector
from .ocr import PlateOCR
from .utils import OUTPUT_DIR, crop_box, draw_detections, save_image, unique_path


ProgressCallback = Callable[[float, str], None]


class VideoProcessor:
    """Process recorded video files frame-by-frame with deduplicated plate logging."""

    def __init__(self, detector: NumberPlateDetector, ocr: PlateOCR, db: VehicleLogDB):
        self.detector = detector
        self.ocr = ocr
        self.db = db

    def process(
        self,
        video_path: str | Path,
        conf_threshold: float,
        ocr_threshold: float,
        frame_skip: int = 5,
        cooldown_seconds: int = 60,
        duplicate_window_seconds: int = 10,
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        """Read a video, annotate detections, persist logs, and write a preview video."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = unique_path(OUTPUT_DIR, ".mp4", "processed_video")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        seen: dict[str, datetime] = {}
        events: list[dict] = []
        detections_report: list[dict] = []
        sample_frames: list[Path] = []
        sampled_frames_processed = 0
        yolo_detection_count = 0
        readable_plate_count = 0
        frame_index = 0
        last_detections: list[dict] = []
        last_frame_path = None

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                should_process = frame_index % max(1, frame_skip) == 0
                if should_process:
                    sampled_frames_processed += 1
                    detections = self.detector.detect(frame, conf=conf_threshold)
                    yolo_detection_count += len(detections)
                    last_detections = detections
                    for detection in detections:
                        plate_crop = crop_box(frame, detection["box"])
                        ocr_result = self.ocr.read_plate(plate_crop, min_confidence=ocr_threshold)
                        report_row = {
                            "frame": frame_index,
                            "yolo_confidence": round(detection["confidence"], 3),
                            "box": detection["box"],
                            "plate_text": "",
                            "ocr_confidence": "",
                            "log_action": "not_readable",
                        }
                        if not ocr_result:
                            detections_report.append(report_row)
                            continue

                        readable_plate_count += 1
                        plate_number = ocr_result.text
                        detection["text"] = plate_number
                        report_row["plate_text"] = plate_number
                        report_row["ocr_confidence"] = round(ocr_result.confidence, 3)
                        now = datetime.now()
                        if plate_number in seen and now - seen[plate_number] < timedelta(seconds=duplicate_window_seconds):
                            report_row["log_action"] = "duplicate"
                            detections_report.append(report_row)
                            continue
                        seen[plate_number] = now

                        last_frame_path = save_image(draw_detections(frame, [detection]), "video_frame")
                        action, log_id = self.db.record_detection(
                            plate_number,
                            "video",
                            last_frame_path,
                            detection["confidence"],
                            cooldown_seconds=cooldown_seconds,
                            duplicate_window_seconds=duplicate_window_seconds,
                        )
                        report_row["log_action"] = action
                        detections_report.append(report_row)
                        if action != "duplicate":
                            events.append(
                                {
                                    "vehicle_number": plate_number,
                                    "action": action,
                                    "log_id": log_id,
                                    "frame": frame_index,
                                    "confidence": detection["confidence"],
                                }
                            )

                annotated = draw_detections(frame, last_detections)
                if should_process and last_detections and len(sample_frames) < 6:
                    sample_frames.append(save_image(annotated, "video_sample"))
                writer.write(annotated)

                if progress_callback and total_frames:
                    progress_callback(min(frame_index / total_frames, 1.0), f"Processed frame {frame_index}/{total_frames}")
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        if progress_callback:
            progress_callback(1.0, "Video processing complete")

        return {
            "output_video": output_path,
            "events": events,
            "detections": detections_report,
            "frames_processed": frame_index,
            "sampled_frames_processed": sampled_frames_processed,
            "yolo_detection_count": yolo_detection_count,
            "readable_plate_count": readable_plate_count,
            "sample_frames": sample_frames,
            "last_frame_path": last_frame_path,
        }
