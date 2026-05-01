from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np


INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")
GENERIC_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{5,12}$")
DIGIT_CORRECTIONS = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
LETTER_CORRECTIONS = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
INDIAN_PLATE_LAYOUTS = {
    9: "LLDDLDDDD",
    10: "LLDDLLDDDD",
}


@dataclass
class OCRResult:
    text: str
    confidence: float
    raw_text: str
    source: str = "unknown"
    pattern_distance: int = 99


class PlateOCR:
    """EasyOCR-based number plate reader with Indian plate normalization."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError("Install EasyOCR with: pip install easyocr") from exc
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove OCR noise and keep only uppercase letters and digits."""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
        return cleaned

    @staticmethod
    def matches_layout(text: str, layout: str) -> bool:
        """Check whether text follows one explicit letter/digit layout."""
        if len(text) != len(layout):
            return False
        for char, expected in zip(text, layout):
            if expected == "L" and not char.isalpha():
                return False
            if expected == "D" and not char.isdigit():
                return False
        return True

    @staticmethod
    def pattern_distance(text: str) -> int:
        """Return how far a candidate is from the 9/10-character Indian plate layouts."""
        cleaned = PlateOCR.clean_text(text)
        distances = []
        for length, layout in INDIAN_PLATE_LAYOUTS.items():
            distance = abs(len(cleaned) - length) * 2
            for index, expected in enumerate(layout[: len(cleaned)]):
                char = cleaned[index]
                if expected == "L" and not char.isalpha():
                    distance += 1
                if expected == "D" and not char.isdigit():
                    distance += 1
            distances.append(distance)
        return min(distances) if distances else 99

    @staticmethod
    def correct_to_layout(text: str, layout: str) -> str:
        """Apply position-based letter/digit correction for one plate layout."""
        chars = list(PlateOCR.clean_text(text))
        if len(chars) != len(layout):
            return "".join(chars)

        for index, expected in enumerate(layout):
            value = chars[index]
            if expected == "D" and not value.isdigit() and value in DIGIT_CORRECTIONS:
                chars[index] = DIGIT_CORRECTIONS[value]
            elif expected == "L" and not value.isalpha() and value in LETTER_CORRECTIONS:
                chars[index] = LETTER_CORRECTIONS[value]

        return "".join(chars)

    @staticmethod
    def pattern_correct_candidates(text: str) -> list[str]:
        """Generate corrected plate candidates using Indian plate position rules."""
        cleaned = PlateOCR.clean_text(text)
        candidates = [cleaned] if cleaned else []

        if len(cleaned) in INDIAN_PLATE_LAYOUTS:
            candidates.append(PlateOCR.correct_to_layout(cleaned, INDIAN_PLATE_LAYOUTS[len(cleaned)]))

        # Some Karnataka plates are read with the KA prefix collapsed into a leading I/1.
        # Example: I70M1248 -> KA70M1248.
        if re.fullmatch(r"[I1][0-9A-Z]{2}[A-Z0-9][0-9A-Z]{4}", cleaned):
            candidates.append("KA" + cleaned[1:])

        corrected = []
        for candidate in candidates:
            if not candidate:
                continue
            candidate = PlateOCR.clean_text(candidate)
            if len(candidate) in INDIAN_PLATE_LAYOUTS:
                candidate = PlateOCR.correct_to_layout(candidate, INDIAN_PLATE_LAYOUTS[len(candidate)])
            if candidate not in corrected:
                corrected.append(candidate)
        return corrected

    @staticmethod
    def indian_pattern_score(text: str) -> int:
        """Score how closely text matches an Indian plate after position correction."""
        cleaned = PlateOCR.clean_text(text)
        if len(cleaned) not in {8, 9, 10}:
            return 0
        corrected = PlateOCR.normalize_indian_plate(text)
        if INDIAN_PLATE_PATTERN.match(corrected):
            return 2
        return 1 if PlateOCR.pattern_distance(text) <= 2 and len(cleaned) in INDIAN_PLATE_LAYOUTS else 0

    @staticmethod
    def generic_pattern_score(text: str) -> int:
        """Score generic alphanumeric plate validity without applying corrections."""
        return 1 if GENERIC_PLATE_PATTERN.match(PlateOCR.clean_text(text)) else 0

    @staticmethod
    def build_plate_candidates(text: str) -> list[tuple[str, str, int]]:
        """Return candidate tuples as (text, pattern_type, pattern_score)."""
        cleaned = PlateOCR.clean_text(text)
        candidates: list[tuple[str, str, int]] = []

        if PlateOCR.generic_pattern_score(cleaned):
            candidates.append((cleaned, "generic", 1))

        if PlateOCR.indian_pattern_score(cleaned):
            for corrected in PlateOCR.pattern_correct_candidates(cleaned):
                normalized = PlateOCR.normalize_indian_plate(corrected)
                if INDIAN_PLATE_PATTERN.match(normalized):
                    candidates.append((normalized, "indian", 2))

        unique = []
        seen = set()
        for value, pattern_type, score in candidates:
            key = (value, pattern_type)
            if key not in seen:
                seen.add(key)
                unique.append((value, pattern_type, score))
        return unique

    @staticmethod
    def extract_candidates(text: str) -> list[str]:
        """Return flexible full or partial plate-like tokens from OCR text."""
        cleaned = PlateOCR.clean_text(text)
        candidates: list[str] = []
        if 6 <= len(cleaned) <= 12:
            candidates.append(cleaned)
        for match in re.findall(r"[A-Z0-9]{6,12}", cleaned):
            candidates.append(match)
        if len(cleaned) > 12:
            for size in range(12, 5, -1):
                for start in range(0, len(cleaned) - size + 1):
                    candidates.append(cleaned[start : start + size])
        return list(dict.fromkeys(candidates))

    @staticmethod
    def normalize_indian_plate(text: str) -> str:
        """Return the best pattern-corrected Indian plate candidate."""
        candidates = PlateOCR.pattern_correct_candidates(text)
        if not candidates:
            return PlateOCR.clean_text(text)
        return min(candidates, key=lambda item: (PlateOCR.pattern_distance(item), -len(item)))

    @staticmethod
    def is_valid_plate(text: str) -> bool:
        """Check whether text resembles an Indian or generic vehicle number."""
        cleaned = PlateOCR.clean_text(text)
        return bool(INDIAN_PLATE_PATTERN.match(cleaned) or GENERIC_PLATE_PATTERN.match(cleaned))

    @staticmethod
    def is_strict_indian_plate(text: str) -> bool:
        """Check the common Indian state/RTO/series/number structure."""
        return bool(INDIAN_PLATE_PATTERN.match(PlateOCR.normalize_indian_plate(text)))

    @staticmethod
    def enhance_crop(plate_image: np.ndarray, scale: int = 3) -> np.ndarray:
        """Upscale and sharpen a detected plate crop before OCR."""
        plate = cv2.resize(plate_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(plate, -1, kernel)

    @staticmethod
    def preprocess_variants(plate_image: np.ndarray) -> dict[str, np.ndarray]:
        """Build multiple OCR inputs from one crop for robust text extraction."""
        enhanced = PlateOCR.enhance_crop(plate_image)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return {
            "original": plate_image,
            "enhanced": enhanced,
            "grayscale": gray,
            "threshold": thresh,
        }

    @staticmethod
    def preprocess_plate(plate_image: np.ndarray) -> np.ndarray:
        """Return the thresholded OCR variant for callers that need one processed image."""
        return PlateOCR.preprocess_variants(plate_image)["threshold"]

    @staticmethod
    def score_candidate(candidate: OCRResult) -> tuple[float, int, int, int]:
        """Rank candidates by OCR confidence first, then pattern quality."""
        strict = PlateOCR.is_strict_indian_plate(candidate.text)
        generic = PlateOCR.generic_pattern_score(candidate.text)
        pattern_score = 2 if strict else generic
        return candidate.confidence, pattern_score, -candidate.pattern_distance, len(candidate.text)

    def read_plate(self, plate_image: np.ndarray, min_confidence: float) -> OCRResult | None:
        """Run multi-pass OCR and return the best cleaned number plate candidate."""
        result, _ = self.read_plate_with_debug(plate_image, min_confidence=min_confidence)
        return result

    def read_plate_with_debug(
        self,
        plate_image: np.ndarray,
        min_confidence: float,
    ) -> tuple[OCRResult | None, dict[str, np.ndarray]]:
        """Run OCR on original, grayscale, and threshold variants and return debug images."""
        if plate_image is None or plate_image.size == 0:
            return None, {}

        variants = self.preprocess_variants(plate_image)
        ocr_inputs = {
            "original": variants["original"],
            "grayscale": variants["grayscale"],
            "threshold": variants["threshold"],
        }

        candidates: list[OCRResult] = []
        for source, image in ocr_inputs.items():
            try:
                results = self.reader.readtext(
                    image,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=1,
                )
            except Exception:
                continue
            for _, raw_text, confidence in results:
                confidence_value = float(confidence)
                if confidence_value < min_confidence:
                    continue
                for candidate_text in self.extract_candidates(raw_text):
                    for plate_text, pattern_type, pattern_score in self.build_plate_candidates(candidate_text):
                        pattern_distance = 0 if pattern_type == "generic" else self.pattern_distance(plate_text)
                        candidates.append(
                            OCRResult(
                                text=plate_text,
                                confidence=confidence_value,
                                raw_text=raw_text,
                                source=f"{source}:{pattern_type}:{pattern_score}",
                                pattern_distance=pattern_distance,
                            )
                        )

        if not candidates:
            return None, variants

        return max(candidates, key=self.score_candidate), variants
