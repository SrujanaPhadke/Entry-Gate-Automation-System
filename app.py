from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.camera import LiveCameraProcessor
from src.database import VehicleLogDB
from src.detector import NumberPlateDetector
from src.ocr import PlateOCR
from src.utils import (
    DEFAULT_MODEL_PATH,
    MODEL_DIR,
    bgr_to_rgb,
    crop_box,
    draw_detections,
    ensure_directories,
    read_image,
    save_uploaded_file,
)
from src.video_processor import VideoProcessor


st.set_page_config(page_title="Entry Gate Automation System", layout="wide")
ensure_directories()


@st.cache_resource
def get_db() -> VehicleLogDB:
    return VehicleLogDB()


@st.cache_resource
def get_detector(model_path: str) -> NumberPlateDetector:
    return NumberPlateDetector(model_path)


@st.cache_resource
def get_ocr(use_gpu: bool = False) -> PlateOCR:
    return PlateOCR(gpu=use_gpu)


def save_uploaded_model(uploaded_model) -> Path:
    """Save a user-provided YOLO weights file for the current application run."""
    if uploaded_model is None:
        return DEFAULT_MODEL_PATH
    model_path = MODEL_DIR / "uploaded_best.pt"
    model_path.write_bytes(uploaded_model.getbuffer())
    return model_path


def load_services(model_path: Path, use_gpu_ocr: bool):
    """Create the database, YOLO detector, and OCR reader used by every mode."""
    db = get_db()
    detector = get_detector(str(model_path))
    ocr = get_ocr(use_gpu_ocr)
    return db, detector, ocr


def show_logs(db: VehicleLogDB) -> None:
    """Render the searchable, exportable vehicle log table."""
    st.subheader("Vehicle Entry/Exit Logs")
    filters = st.columns([2, 1])
    search = filters[0].text_input("Search vehicle number", placeholder="Example: MH12AB1234")
    status = filters[1].selectbox("Status", ["all", "inside", "exited"])

    logs = db.fetch_logs(search=search, status=status)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total records", len(logs))
    metric_cols[1].metric("Inside", int((logs["status"] == "inside").sum()) if not logs.empty else 0)
    metric_cols[2].metric("Exited", int((logs["status"] == "exited").sum()) if not logs.empty else 0)

    st.dataframe(logs, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=db.export_csv(search=search, status=status),
        file_name="vehicle_logs.csv",
        mime="text/csv",
    )


def image_page(
    detector: NumberPlateDetector,
    ocr: PlateOCR,
    db: VehicleLogDB,
    confidence: float,
    ocr_confidence: float,
    cooldown_seconds: int,
    duplicate_window_seconds: int,
    show_ocr_debug: bool,
) -> None:
    """Process one uploaded image and show bounding boxes, crops, OCR, and log action."""
    st.subheader("Image Detection")
    uploaded_image = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])
    if uploaded_image is None:
        return

    image_path = save_uploaded_file(uploaded_image)
    try:
        image = read_image(image_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    detections = detector.detect(image, conf=confidence)
    readable_results = []
    unreadable_crops = []
    debug_results = []

    for detection in detections:
        plate_crop = crop_box(image, detection["box"])
        ocr_result, debug_variants = ocr.read_plate_with_debug(plate_crop, min_confidence=ocr_confidence)
        if not ocr_result:
            unreadable_crops.append((plate_crop, detection["confidence"]))
            if debug_variants:
                debug_results.append((plate_crop, None, debug_variants))
            continue

        detection["text"] = ocr_result.text
        action, _ = db.record_detection(
            vehicle_number=ocr_result.text,
            source_type="image",
            image_path=image_path,
            confidence_score=detection["confidence"],
            cooldown_seconds=cooldown_seconds,
            duplicate_window_seconds=duplicate_window_seconds,
        )
        readable_results.append((plate_crop, ocr_result.text, detection["confidence"], ocr_result.confidence, action))
        debug_results.append((plate_crop, ocr_result, debug_variants))

    annotated = draw_detections(image, detections)
    metric_cols = st.columns(3)
    metric_cols[0].metric("YOLO detections", len(detections))
    metric_cols[1].metric("Readable plates", len(readable_results))
    metric_cols[2].metric("Unreadable crops", len(unreadable_crops))

    left, right = st.columns(2)
    left.image(bgr_to_rgb(image), caption="Original image", use_container_width=True)
    right.image(bgr_to_rgb(annotated), caption="Detected bounding boxes", use_container_width=True)

    if not detections:
        st.warning("No number plate was detected. Try lowering YOLO confidence or use a clearer image.")
        return
    if not readable_results:
        st.warning("YOLO detected plate regions, but OCR could not read a valid vehicle number.")
    else:
        best_text = max(readable_results, key=lambda item: item[3])[1]
        st.success(f"Vehicle Number: {best_text}")
        st.write("Extracted vehicle numbers")

    table_rows = []
    if readable_results:
        cols = st.columns(min(len(readable_results), 3))
        for index, (crop, number, yolo_confidence, ocr_score, action) in enumerate(readable_results):
            cols[index % len(cols)].image(bgr_to_rgb(crop), caption=number, use_container_width=True)
            table_rows.append(
                {
                    "vehicle_number": number,
                    "yolo_confidence": round(yolo_confidence, 3),
                    "ocr_confidence": round(ocr_score, 3),
                    "ocr_source": next(
                        (
                            result.source
                            for _, result, _ in debug_results
                            if result is not None and result.text == number
                        ),
                        "",
                    ),
                    "log_action": action,
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    if unreadable_crops:
        st.write("Unreadable plate crops")
        crop_cols = st.columns(min(len(unreadable_crops), 3))
        for index, (crop, score) in enumerate(unreadable_crops[:6]):
            crop_cols[index % len(crop_cols)].image(
                bgr_to_rgb(crop),
                caption=f"YOLO confidence {score:.2f}",
                use_container_width=True,
            )

    if show_ocr_debug and debug_results:
        st.subheader("OCR Debug")
        for index, (crop, result, variants) in enumerate(debug_results[:3], start=1):
            st.caption(f"Detection {index}: {result.text if result else 'No OCR candidate'}")
            debug_cols = st.columns(3)
            debug_cols[0].image(bgr_to_rgb(crop), caption="Original crop", use_container_width=True)
            debug_cols[1].image(
                bgr_to_rgb(variants["enhanced"]),
                caption="2x/3x sharpened crop",
                use_container_width=True,
            )
            debug_cols[2].image(variants["threshold"], caption="Processed threshold image", use_container_width=True)


def video_page(
    detector: NumberPlateDetector,
    ocr: PlateOCR,
    db: VehicleLogDB,
    confidence: float,
    ocr_confidence: float,
    cooldown_seconds: int,
    duplicate_window_seconds: int,
) -> None:
    """Process uploaded video footage frame-by-frame and generate logs plus reports."""
    st.subheader("Video Detection")
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    frame_skip = st.slider("Analyze every Nth frame", min_value=1, max_value=20, value=1)
    if uploaded_video is None:
        return

    video_path = save_uploaded_file(uploaded_video)
    if not st.button("Process Video", type="primary"):
        return

    processor = VideoProcessor(detector, ocr, db)
    progress = st.progress(0)
    status = st.empty()

    def update_progress(value: float, message: str) -> None:
        progress.progress(value)
        status.caption(message)

    try:
        result = processor.process(
            video_path=video_path,
            conf_threshold=confidence,
            ocr_threshold=ocr_confidence,
            frame_skip=frame_skip,
            cooldown_seconds=cooldown_seconds,
            duplicate_window_seconds=duplicate_window_seconds,
            progress_callback=update_progress,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success("Video processing complete.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total frames", result["frames_processed"])
    metric_cols[1].metric("Frames analyzed", result["sampled_frames_processed"])
    metric_cols[2].metric("YOLO plate detections", result["yolo_detection_count"])
    metric_cols[3].metric("Readable plates", result["readable_plate_count"])

    st.markdown(
        """
        - Process recorded video footage
        - Detect vehicles frame-by-frame
        - Identify and extract number plates
        """
    )

    if result["events"]:
        st.subheader("Entry/Exit Logs")
        st.dataframe(result["events"], use_container_width=True, hide_index=True)
        st.caption(f"Logged {len(result['events'])} entry/exit events.")
    elif result["yolo_detection_count"]:
        st.warning(
            "YOLO detected possible number plates, but no readable entry/exit log was created. "
            "Try processing every frame, lowering OCR confidence, or using a clearer/closer video."
        )
    else:
        st.warning("No number plates were detected. Try lowering YOLO confidence or use a clearer video.")

    if result["detections"]:
        st.subheader("Frame-by-frame Detection Output")
        report = pd.DataFrame(result["detections"])
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Detection Report CSV",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="video_detection_report.csv",
            mime="text/csv",
        )

    if result["sample_frames"]:
        st.subheader("Detected Frame Samples")
        sample_cols = st.columns(min(len(result["sample_frames"]), 3))
        for index, frame_path in enumerate(result["sample_frames"]):
            sample_cols[index % len(sample_cols)].image(str(frame_path), use_container_width=True)

    st.subheader("Processed Video Preview")
    st.video(str(result["output_video"]))


def live_page(
    detector: NumberPlateDetector,
    ocr: PlateOCR,
    db: VehicleLogDB,
    confidence: float,
    ocr_confidence: float,
    cooldown_seconds: int,
    duplicate_window_seconds: int,
) -> None:
    """Run real-time detection from webcam index or IP camera URL."""
    st.subheader("Live Camera Detection")
    camera_source = st.text_input("Camera source", value="0", help="Use 0 for webcam or paste an RTSP/HTTP URL.")
    frames_to_process = st.slider("Frames to process", min_value=20, max_value=1000, value=200)
    live_cols = st.columns(3)
    buffer_size = live_cols[0].slider("Frame buffer", min_value=5, max_value=10, value=8)
    process_every_n_frames = live_cols[1].slider("Process every Nth frame", min_value=1, max_value=10, value=3)
    stability_delay = live_cols[2].slider("Frame delay seconds", min_value=0.00, max_value=0.20, value=0.03, step=0.01)

    if not st.button("Start Camera", type="primary"):
        return

    processor = LiveCameraProcessor(detector, ocr, db)
    frame_box = st.empty()
    result_box = st.empty()
    status_box = st.empty()
    crop_box_ui = st.empty()
    event_box = st.empty()
    events = []
    processed_count = 0
    last_text = None
    ocr_failures = 0

    try:
        for frame, frame_events, live_status in processor.run_capture(
            source=camera_source,
            conf_threshold=confidence,
            ocr_threshold=ocr_confidence,
            cooldown_seconds=cooldown_seconds,
            duplicate_window_seconds=duplicate_window_seconds,
            max_frames=frames_to_process,
            buffer_size=buffer_size,
            process_every_n_frames=process_every_n_frames,
            stability_delay=stability_delay,
        ):
            processed_count += 1
            frame_box.image(bgr_to_rgb(frame), caption="Camera feed", use_container_width=True)
            status_box.caption(
                f"Processed live frame {processed_count}/{frames_to_process} | "
                f"buffer crops: {live_status.get('buffer_count', 0)} | "
                f"best sharpness: {live_status.get('best_sharpness') or '-'}"
            )
            if live_status.get("best_text"):
                last_text = live_status["best_text"]
                result_box.success(f"Vehicle Number: {last_text}")
            elif live_status.get("ocr_failed"):
                ocr_failures += 1
                result_box.warning("Live plate detected but OCR failed")

            if live_status.get("best_crop") is not None and live_status.get("processed_crop") is not None:
                crop_cols = crop_box_ui.columns(2)
                crop_cols[0].image(bgr_to_rgb(live_status["best_crop"]), caption="Sharpest live crop", use_container_width=True)
                crop_cols[1].image(live_status["processed_crop"], caption="Processed live crop", use_container_width=True)

            if frame_events:
                events.extend(frame_events)
                event_box.dataframe(events, use_container_width=True, hide_index=True)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success("Camera run finished.")
    if last_text:
        st.success(f"Last Vehicle Number: {last_text}")
    if ocr_failures:
        st.warning(f"Live plate detected but OCR failed {ocr_failures} time(s).")
    if not events:
        st.info("No entry/exit events were recorded during this live camera run.")


def main() -> None:
    st.title("Entry Gate Automation System")
    st.caption("YOLO number plate detection, OCR extraction, and SQLite entry/exit monitoring.")

    with st.sidebar:
        mode = st.radio("Mode", ["Image Detection", "Video Detection", "Live Camera Detection", "Logs"])
        confidence = st.slider("YOLO confidence", min_value=0.10, max_value=1.00, value=0.35, step=0.05)
        ocr_confidence = st.slider("OCR confidence", min_value=0.10, max_value=1.00, value=0.30, step=0.05)
        cooldown_seconds = int(st.number_input("Entry/exit cooldown seconds", min_value=1, max_value=3600, value=60))
        duplicate_window_seconds = int(st.number_input("Duplicate filter seconds", min_value=1, max_value=600, value=10))
        use_gpu_ocr = st.checkbox("Use GPU for OCR", value=False)
        show_ocr_debug = st.checkbox("Show OCR debug images", value=True)
        uploaded_model = st.file_uploader("Optional custom YOLO model (.pt)", type=["pt"])
        model_path = save_uploaded_model(uploaded_model)
        st.caption(f"Using model: {model_path}")
        st.caption(f"Active thresholds: YOLO {confidence:.2f}, OCR {ocr_confidence:.2f}")

    try:
        db, detector, ocr = load_services(model_path, use_gpu_ocr)
    except Exception as exc:
        st.error(str(exc))
        st.info("Place trained YOLO weights at models/best.pt or upload a .pt file from the sidebar.")
        show_logs(get_db())
        return

    if mode == "Image Detection":
        image_page(
            detector,
            ocr,
            db,
            confidence,
            ocr_confidence,
            cooldown_seconds,
            duplicate_window_seconds,
            show_ocr_debug,
        )
    elif mode == "Video Detection":
        video_page(detector, ocr, db, confidence, ocr_confidence, cooldown_seconds, duplicate_window_seconds)
    elif mode == "Live Camera Detection":
        live_page(detector, ocr, db, confidence, ocr_confidence, cooldown_seconds, duplicate_window_seconds)
    else:
        show_logs(db)


if __name__ == "__main__":
    main()
