from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kavach.config import (
    FACE_COLOR_SCORE_MINIMUM,
    FACE_CORRELATION_MINIMUM,
    FACE_IMAGE_SIZE,
    FACE_MATCH_THRESHOLD,
    FACE_ORB_MATCH_MINIMUM,
    FACE_RAW_CORRELATION_MINIMUM,
    FACE_SIGNATURE_SIZE,
    FACE_STRONG_CORRELATION_MINIMUM,
    FACE_STRONG_EDGE_MINIMUM,
    FACE_STRONG_MATCH_THRESHOLD,
    FACE_STRONG_RAW_CORRELATION_MINIMUM,
    FACE_STRONG_SIGNATURE_MINIMUM,
    KNOWN_FACES_DIR,
    PHONE_CONFIDENCE_THRESHOLD,
    YOLO_MODEL_NAME,
)
from kavach.runtime import import_optional_module


def decode_uploaded_image(uploaded_file: Any) -> np.ndarray | None:
    if uploaded_file is None:
        return None

    raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    nparray = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    return image


@lru_cache(maxsize=1)
def face_cascade() -> cv2.CascadeClassifier:
    classifier = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    return classifier


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    faces = face_cascade().detectMultiScale(
        blurred,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    return sorted((int(x), int(y), int(w), int(h)) for x, y, w, h in faces)


def crop_face(image: np.ndarray, bbox: tuple[int, int, int, int], padding: float = 0.18) -> np.ndarray:
    x, y, width, height = bbox
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image.shape[1], x + width + pad_x)
    bottom = min(image.shape[0], y + height + pad_y)
    face = image[top:bottom, left:right]
    return cv2.resize(face, FACE_IMAGE_SIZE)


def extract_primary_face(image: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    faces = detect_faces(image)
    if not faces:
        return None, None
    primary = max(faces, key=lambda item: item[2] * item[3])
    return crop_face(image, primary), primary


def normalize_face_image(face_image: np.ndarray) -> np.ndarray:
    return cv2.resize(face_image, FACE_IMAGE_SIZE)


@lru_cache(maxsize=4)
def face_focus_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (max(int(width * 0.34), 1), max(int(height * 0.44), 1)),
        0,
        0,
        360,
        255,
        -1,
    )
    return mask


def preprocess_face(face_image: np.ndarray) -> np.ndarray:
    normalized = normalize_face_image(face_image)
    grayscale = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    resized = cv2.resize(equalized, FACE_SIGNATURE_SIZE)
    mask = face_focus_mask(resized.shape)
    return cv2.bitwise_and(resized, resized, mask=mask)


def local_binary_pattern_histogram(processed_face: np.ndarray, grid_size: int = 4) -> np.ndarray:
    height, width = processed_face.shape
    padded = np.pad(processed_face, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    for bit, (delta_y, delta_x) in enumerate(offsets):
        neighbor = padded[1 + delta_y : height + 1 + delta_y, 1 + delta_x : width + 1 + delta_x]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    cell_height = max(height // grid_size, 1)
    cell_width = max(width // grid_size, 1)
    histograms: list[np.ndarray] = []
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            cell = lbp[
                grid_y * cell_height : (grid_y + 1) * cell_height,
                grid_x * cell_width : (grid_x + 1) * cell_width,
            ]
            histogram = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            histogram = histogram / (histogram.sum() + 1e-8)
            histograms.append(histogram)
    return np.concatenate(histograms).astype(np.float32)


def gradient_histogram(processed_face: np.ndarray, grid_size: int = 4, bins: int = 9) -> np.ndarray:
    grad_x = cv2.Sobel(processed_face, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(processed_face, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle %= 180.0

    height, width = processed_face.shape
    cell_height = max(height // grid_size, 1)
    cell_width = max(width // grid_size, 1)
    bin_width = 180.0 / bins
    histograms: list[np.ndarray] = []

    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            magnitude_cell = magnitude[
                grid_y * cell_height : (grid_y + 1) * cell_height,
                grid_x * cell_width : (grid_x + 1) * cell_width,
            ]
            angle_cell = angle[
                grid_y * cell_height : (grid_y + 1) * cell_height,
                grid_x * cell_width : (grid_x + 1) * cell_width,
            ]
            histogram = np.zeros(bins, dtype=np.float32)
            bucket_indices = np.clip((angle_cell / bin_width).astype(np.int32), 0, bins - 1)
            for bucket in range(bins):
                histogram[bucket] = float(magnitude_cell[bucket_indices == bucket].sum())
            histogram = histogram / (np.linalg.norm(histogram) + 1e-8)
            histograms.append(histogram)
    return np.concatenate(histograms).astype(np.float32)


def _build_face_signature_from_processed(processed_face: np.ndarray) -> np.ndarray:
    mask = face_focus_mask(processed_face.shape)
    histogram = cv2.calcHist([processed_face], [0], mask, [32], [0, 256]).flatten().astype(np.float32)
    histogram = histogram / (np.linalg.norm(histogram) + 1e-8)

    lbp_features = local_binary_pattern_histogram(processed_face)
    gradient_features = gradient_histogram(processed_face)
    height, width = processed_face.shape
    border_y = max(height // 6, 1)
    border_x = max(width // 6, 1)
    center_patch = processed_face[border_y : height - border_y, border_x : width - border_x]
    center_map = cv2.resize(center_patch, (24, 24)).astype(np.float32).flatten() / 255.0

    signature = np.concatenate([histogram, lbp_features, gradient_features, center_map]).astype(np.float32)
    signature = signature / (np.linalg.norm(signature) + 1e-8)
    return signature


def build_face_signature(face_image: np.ndarray) -> np.ndarray:
    processed = preprocess_face(face_image)
    return _build_face_signature_from_processed(processed)


def save_face_assets(student_id: int, face_image: np.ndarray) -> tuple[str, str]:
    face_path = KNOWN_FACES_DIR / f"student_{student_id}.png"
    signature_path = KNOWN_FACES_DIR / f"student_{student_id}.npy"
    normalized_face = normalize_face_image(face_image)
    cv2.imwrite(str(face_path), normalized_face)
    np.save(signature_path, build_face_signature(normalized_face))
    return str(face_path), str(signature_path)


def load_face_signature(signature_path: str) -> np.ndarray | None:
    try:
        return np.load(signature_path)
    except Exception:
        return None


def compute_correlation_score(live_processed: np.ndarray, stored_processed: np.ndarray) -> tuple[float, float]:
    mask = face_focus_mask(live_processed.shape) > 0
    live_values = live_processed[mask].astype(np.float32)
    stored_values = stored_processed[mask].astype(np.float32)
    live_values = (live_values - live_values.mean()) / (live_values.std() + 1e-8)
    stored_values = (stored_values - stored_values.mean()) / (stored_values.std() + 1e-8)
    correlation = float(np.mean(live_values * stored_values))
    normalized_score = max(0.0, min((correlation + 1.0) / 2.0, 1.0))
    return normalized_score, correlation


def compute_edge_score(live_processed: np.ndarray, stored_processed: np.ndarray) -> float:
    mask = face_focus_mask(live_processed.shape) > 0
    live_edges = cv2.Canny(live_processed, 60, 150).astype(np.float32) / 255.0
    stored_edges = cv2.Canny(stored_processed, 60, 150).astype(np.float32) / 255.0
    difference = float(np.mean(np.abs(live_edges[mask] - stored_edges[mask])))
    return max(0.0, 1.0 - min(difference * 2.2, 1.0))


def compute_orb_score(live_processed: np.ndarray, stored_processed: np.ndarray) -> float:
    detector = cv2.ORB_create(nfeatures=256, scaleFactor=1.2, nlevels=8)
    live_keypoints, live_descriptors = detector.detectAndCompute(live_processed, None)
    stored_keypoints, stored_descriptors = detector.detectAndCompute(stored_processed, None)
    if (
        live_descriptors is None
        or stored_descriptors is None
        or not live_keypoints
        or not stored_keypoints
    ):
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(live_descriptors, stored_descriptors)
    if not matches:
        return 0.0

    good_matches = [match for match in matches if match.distance <= 48]
    coverage = min(len(good_matches) / max(min(len(live_keypoints), len(stored_keypoints)), 1), 1.0)
    if not good_matches:
        return max(0.0, coverage * 0.5)

    average_distance = sum(match.distance for match in good_matches) / len(good_matches)
    quality = max(0.0, 1.0 - (average_distance / 48.0))
    return max(0.0, min((coverage * 0.65) + (quality * 0.35), 1.0))


def compute_color_score(live_face: np.ndarray, stored_face: np.ndarray) -> float:
    live_resized = normalize_face_image(live_face)
    stored_resized = normalize_face_image(stored_face)
    live_hsv = cv2.cvtColor(live_resized, cv2.COLOR_BGR2HSV)
    stored_hsv = cv2.cvtColor(stored_resized, cv2.COLOR_BGR2HSV)

    mask = np.zeros(FACE_IMAGE_SIZE[::-1], dtype=np.uint8)
    center_x = FACE_IMAGE_SIZE[0] // 2
    center_y = FACE_IMAGE_SIZE[1] // 2
    cv2.ellipse(mask, (center_x, center_y), (76, 96), 0, 0, 360, 255, -1)

    live_hist = cv2.calcHist([live_hsv], [0, 1], mask, [24, 24], [0, 180, 0, 256])
    stored_hist = cv2.calcHist([stored_hsv], [0, 1], mask, [24, 24], [0, 180, 0, 256])
    live_hist = cv2.normalize(live_hist, None).flatten()
    stored_hist = cv2.normalize(stored_hist, None).flatten()
    histogram_distance = float(cv2.compareHist(live_hist, stored_hist, cv2.HISTCMP_BHATTACHARYYA))
    histogram_score = max(0.0, 1.0 - min(histogram_distance, 1.0))

    live_mean = cv2.mean(live_resized, mask=mask)[:3]
    stored_mean = cv2.mean(stored_resized, mask=mask)[:3]
    mean_difference = float(np.mean([abs(left - right) for left, right in zip(live_mean, stored_mean)]))
    mean_score = max(0.0, 1.0 - (mean_difference / 80.0))

    return max(0.0, min((histogram_score * 0.65) + (mean_score * 0.35), 1.0))


def translate_face_image(face_image: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    if shift_x == 0 and shift_y == 0:
        return face_image.copy()
    transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(
        face_image,
        transform,
        (face_image.shape[1], face_image.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )


def rotate_face_image(face_image: np.ndarray, angle: float = 0.0) -> np.ndarray:
    if abs(angle) < 1e-6:
        return face_image.copy()
    height, width = face_image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        face_image,
        transform,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )


def face_variant_candidates(face_image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    seen_signatures: set[bytes] = set()
    offsets = [
        (0, 0),
        (-20, 0),
        (20, 0),
        (-12, 0),
        (12, 0),
        (-8, 0),
        (8, 0),
        (0, -16),
        (0, 16),
        (0, -12),
        (0, 12),
        (0, -6),
        (0, 6),
        (-8, -8),
        (8, -8),
        (-8, 8),
        (8, 8),
    ]
    angles = (-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0)
    scales = (0.86, 0.92, 1.0, 1.08, 1.14)

    for mirrored in (False, True):
        mirrored_face = cv2.flip(face_image, 1) if mirrored else face_image.copy()
        mirror_label = "mirrored" if mirrored else "direct"
        for angle in angles:
            rotated_face = rotate_face_image(mirrored_face, angle=angle)
            for scale in scales:
                if abs(scale - 1.0) > 1e-6:
                    height, width = rotated_face.shape[:2]
                    scaled_face = cv2.warpAffine(
                        rotated_face,
                        cv2.getRotationMatrix2D((width / 2, height / 2), 0.0, scale),
                        (width, height),
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                else:
                    scaled_face = rotated_face.copy()

                for shift_x, shift_y in offsets:
                    variant = translate_face_image(scaled_face, shift_x=shift_x, shift_y=shift_y)
                    variant_key = variant.tobytes()
                    if variant_key in seen_signatures:
                        continue
                    seen_signatures.add(variant_key)
                    variants.append((f"{mirror_label}:{angle:.0f}:{scale:.2f}:{shift_x},{shift_y}", variant))
    return variants


def _resolve_stored_signature(
    stored_processed: np.ndarray,
    stored_signature: np.ndarray | None,
) -> np.ndarray:
    expected_stored_signature = _build_face_signature_from_processed(stored_processed)
    if stored_signature is None or stored_signature.shape != expected_stored_signature.shape:
        return expected_stored_signature
    return stored_signature


def _face_match_details_from_processed(
    live_face: np.ndarray,
    stored_face: np.ndarray,
    live_processed: np.ndarray,
    stored_processed: np.ndarray,
    stored_signature: np.ndarray,
) -> dict[str, float]:
    live_signature = _build_face_signature_from_processed(live_processed)
    signature_score = float(np.dot(live_signature, stored_signature))
    correlation_score, raw_correlation = compute_correlation_score(live_processed, stored_processed)
    edge_score = compute_edge_score(live_processed, stored_processed)
    orb_score = compute_orb_score(live_processed, stored_processed)
    color_score = compute_color_score(live_face, stored_face)

    final_score = (
        (signature_score * 0.25)
        + (correlation_score * 0.28)
        + (edge_score * 0.07)
        + (orb_score * 0.10)
        + (color_score * 0.30)
    ) * 100.0

    if raw_correlation < FACE_RAW_CORRELATION_MINIMUM:
        final_score *= max(raw_correlation / max(FACE_RAW_CORRELATION_MINIMUM, 1e-8), 0.1)

    return {
        "score": round(final_score, 2),
        "signature_score": round(signature_score, 4),
        "correlation_score": round(correlation_score, 4),
        "raw_correlation": round(raw_correlation, 4),
        "edge_score": round(edge_score, 4),
        "orb_score": round(orb_score, 4),
        "color_score": round(color_score, 4),
    }


def face_match_details(
    live_face: np.ndarray,
    stored_face: np.ndarray,
    stored_signature: np.ndarray | None = None,
) -> dict[str, float]:
    live_processed = preprocess_face(live_face)
    stored_processed = preprocess_face(stored_face)
    resolved_signature = _resolve_stored_signature(stored_processed, stored_signature)
    return _face_match_details_from_processed(
        live_face,
        stored_face,
        live_processed,
        stored_processed,
        resolved_signature,
    )


def best_face_match_details(
    live_face: np.ndarray,
    stored_face: np.ndarray,
    stored_signature: np.ndarray | None = None,
) -> dict[str, float | str]:
    stored_processed = preprocess_face(stored_face)
    resolved_signature = _resolve_stored_signature(stored_processed, stored_signature)
    best_details: dict[str, float | str] | None = None
    for variant_label, candidate_face in face_variant_candidates(live_face):
        candidate_details = _face_match_details_from_processed(
            candidate_face,
            stored_face,
            preprocess_face(candidate_face),
            stored_processed,
            resolved_signature,
        )
        candidate_details["variant"] = variant_label
        if best_details is None:
            best_details = candidate_details
            continue
        if (
            candidate_details["score"],
            candidate_details["orb_score"],
            candidate_details["correlation_score"],
        ) > (
            best_details["score"],
            best_details["orb_score"],
            best_details["correlation_score"],
        ):
            best_details = candidate_details

    assert best_details is not None
    return best_details


def compare_face_signatures(
    live_face: np.ndarray,
    stored_face: np.ndarray,
    stored_signature: np.ndarray | None,
) -> float:
    return float(best_face_match_details(live_face, stored_face, stored_signature)["score"])


def verify_face(
    live_image: np.ndarray,
    stored_face_path: str,
    stored_signature_path: str,
) -> dict[str, Any]:
    faces = detect_faces(live_image)
    if not faces:
        return {"match": False, "score": 0.0, "reason": "No face detected in the live image."}
    if len(faces) > 1:
        return {"match": False, "score": 0.0, "reason": "Multiple faces detected in the live image."}

    bbox = faces[0]
    live_face = crop_face(live_image, bbox)

    stored_face = cv2.imread(stored_face_path)
    if stored_face is None:
        return {"match": False, "score": 0.0, "reason": "Registered face data could not be loaded."}

    stored_signature = load_face_signature(stored_signature_path)
    diagnostics = best_face_match_details(live_face, stored_face, stored_signature)
    score = diagnostics["score"]
    orb_score = diagnostics["orb_score"]
    correlation_score = diagnostics["correlation_score"]
    raw_correlation = diagnostics["raw_correlation"]
    primary_match = (
        score >= FACE_MATCH_THRESHOLD
        and correlation_score >= FACE_CORRELATION_MINIMUM
        and raw_correlation >= FACE_RAW_CORRELATION_MINIMUM
        and diagnostics["color_score"] >= FACE_COLOR_SCORE_MINIMUM
    )
    strong_match = (
        score >= FACE_STRONG_MATCH_THRESHOLD
        and diagnostics["signature_score"] >= FACE_STRONG_SIGNATURE_MINIMUM
        and correlation_score >= FACE_STRONG_CORRELATION_MINIMUM
        and raw_correlation >= FACE_STRONG_RAW_CORRELATION_MINIMUM
        and diagnostics["edge_score"] >= FACE_STRONG_EDGE_MINIMUM
    )
    match = primary_match or strong_match

    if match:
        reason = "Face verified successfully."
    elif (
        correlation_score < FACE_CORRELATION_MINIMUM
        or raw_correlation < FACE_RAW_CORRELATION_MINIMUM
        or diagnostics["color_score"] < FACE_COLOR_SCORE_MINIMUM
    ):
        reason = "Face mismatch detected. Registered facial features do not align strongly enough."
    else:
        reason = "Face mismatch detected."

    print(
        "[Kavach Face] "
        f"match={match} "
        f"score={score:.2f} "
        f"variant={diagnostics.get('variant', 'n/a')} "
        f"signature={diagnostics.get('signature_score', 0.0):.4f} "
        f"orb={orb_score:.4f} "
        f"corr={correlation_score:.4f} "
        f"raw_corr={raw_correlation:.4f} "
        f"color={diagnostics.get('color_score', 0.0):.4f} "
        f"edge={diagnostics.get('edge_score', 0.0):.4f} "
        f"primary={primary_match} "
        f"strong={strong_match} "
        f"bbox={bbox} "
        f"stored_face={Path(stored_face_path).name}"
    )

    return {
        "match": match,
        "score": score,
        "reason": reason,
        "bbox": bbox,
        "diagnostics": diagnostics,
    }


class PhoneDetector:
    def __init__(self) -> None:
        self.model = None
        self.available = False
        self.load_error = ""
        self._attempted = False

    def _ensure_loaded(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        try:
            YOLO = import_optional_module("ultralytics").YOLO

            self.model = YOLO(YOLO_MODEL_NAME)
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on environment
            self.available = False
            self.load_error = str(exc)

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if not self.available or self.model is None:
            return []

        detections: list[dict[str, Any]] = []
        try:  # pragma: no cover - model inference depends on environment
            results = self.model.predict(
                image,
                verbose=False,
                classes=[67],
                conf=PHONE_CONFIDENCE_THRESHOLD,
                imgsz=640,
            )
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "bbox": (x1, y1, x2 - x1, y2 - y1),
                            "confidence": round(float(box.conf[0]), 2),
                        }
                    )
        except Exception as exc:
            self.available = False
            self.load_error = str(exc)
        return detections


PHONE_DETECTOR = PhoneDetector()
