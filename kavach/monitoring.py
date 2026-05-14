from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import av
import cv2
import numpy as np

from kavach.runtime import import_optional_module

try:  # pragma: no cover - depends on local runtime
    streamlit_webrtc = import_optional_module("streamlit_webrtc")
    VideoProcessorBase = streamlit_webrtc.VideoProcessorBase

    WEBRTC_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local runtime
    WEBRTC_IMPORT_ERROR = str(exc)

    class VideoProcessorBase:  # type: ignore[override]
        pass

from kavach.config import (
    ALERT_COOLDOWNS,
    ALERT_MESSAGES,
    EVIDENCE_DIR,
    FRAME_ANALYSIS_INTERVAL,
    PHONE_ANALYSIS_INTERVAL,
    SUSPICION_POINTS,
)
from kavach.database import (
    add_alert,
    current_session_score,
    list_alerts,
    update_session_ml_assessment,
)
from kavach.vision import PHONE_DETECTOR, detect_faces

RISK_CLASSES = ("Low Risk", "Medium Risk", "High Risk")


def session_feature_summary(session_id: int) -> dict[str, Any]:
    alerts = list_alerts(session_id=session_id)
    counts = {
        "no_face": 0,
        "multiple_faces": 0,
        "fullscreen_exit": 0,
        "tab_switch": 0,
        "mobile_phone": 0,
    }
    for alert in alerts:
        counts[alert["alert_type"]] = counts.get(alert["alert_type"], 0) + 1

    first_alert = alerts[-1]["created_at"] if alerts else None
    last_alert = alerts[0]["created_at"] if alerts else None

    return {
        "suspicion_score": current_session_score(session_id),
        "total_alerts": len(alerts),
        "first_alert": first_alert,
        "last_alert": last_alert,
        **counts,
    }


def predict_session_risk(session_id: int, elapsed_minutes: int = 0) -> tuple[str, float]:
    summary = session_feature_summary(session_id)
    risk_signal = (
        float(summary["suspicion_score"])
        + (summary["multiple_faces"] * 12.0)
        + (summary["tab_switch"] * 10.0)
        + (summary["mobile_phone"] * 18.0)
        + (summary["fullscreen_exit"] * 8.0)
        + (summary["no_face"] * 4.0)
        + (max(summary["total_alerts"] - 1, 0) * 4.0)
        + (min(max(elapsed_minutes, 0), 180) * 0.08)
    )

    high_trigger = (
        risk_signal >= 90.0
        or summary["mobile_phone"] >= 2
        or summary["tab_switch"] >= 3
        or summary["multiple_faces"] >= 2
    )
    medium_trigger = (
        risk_signal >= 38.0
        or summary["total_alerts"] >= 3
        or (summary["tab_switch"] >= 1 and summary["mobile_phone"] >= 1)
    )

    if high_trigger:
        margin = (
            max(risk_signal - 90.0, 0.0)
            + (summary["mobile_phone"] * 12.0)
            + (summary["tab_switch"] * 7.0)
            + (summary["multiple_faces"] * 8.0)
        )
        confidence = min(0.99, 0.62 + (margin / 160.0))
        return RISK_CLASSES[2], float(confidence)

    if medium_trigger:
        margin = max(risk_signal - 38.0, 0.0) + (summary["total_alerts"] * 5.0)
        confidence = min(0.93, 0.56 + (margin / 150.0))
        return RISK_CLASSES[1], float(confidence)

    stability = max(0.0, 35.0 - risk_signal) + (max(2 - summary["total_alerts"], 0) * 6.0)
    confidence = min(0.88, 0.52 + (stability / 140.0))
    return RISK_CLASSES[0], float(confidence)


def refresh_session_ml_assessment(
    session_id: int,
    elapsed_minutes: int = 0,
    *,
    persist: bool = True,
) -> tuple[str, float]:
    level, confidence = predict_session_risk(session_id, elapsed_minutes=elapsed_minutes)
    if persist:
        update_session_ml_assessment(session_id, level, confidence)
    return level, confidence


def phone_detector_status() -> str:
    if PHONE_DETECTOR.available:
        return "YOLOv8 phone detection loaded."
    if PHONE_DETECTOR.load_error:
        return f"Phone detector unavailable: {PHONE_DETECTOR.load_error}"
    return "YOLOv8 phone detection will initialize on first inference."


def save_evidence_frame(session_id: int, annotated_frame: np.ndarray, event_type: str) -> str:
    target_dir = EVIDENCE_DIR / f"session_{session_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = target_dir / f"{event_type}_{timestamp}.png"
    cv2.imwrite(str(target), annotated_frame)
    return str(target)


@dataclass
class SessionMonitor:
    session_id: int
    frame_counter: int = 0
    last_alert_times: dict[str, float] = field(default_factory=dict)
    last_phone_detections: list[dict[str, Any]] = field(default_factory=list)
    latest_stats: dict[str, Any] = field(
        default_factory=lambda: {
            "face_count": 0,
            "phone_count": 0,
            "last_processed_at": None,
            "new_alerts": [],
        }
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest_stats)

    def _can_emit(self, event_type: str) -> bool:
        now = time.monotonic()
        last_seen = self.last_alert_times.get(event_type, 0.0)
        cooldown = ALERT_COOLDOWNS.get(event_type, 8)
        if now - last_seen < cooldown:
            return False
        self.last_alert_times[event_type] = now
        return True

    def _emit_alert(self, event_type: str, annotated_frame: np.ndarray | None = None) -> str | None:
        if not self._can_emit(event_type):
            return None
        evidence_path = None
        if annotated_frame is not None:
            evidence_path = save_evidence_frame(self.session_id, annotated_frame, event_type)
        add_alert(
            self.session_id,
            event_type,
            SUSPICION_POINTS[event_type],
            ALERT_MESSAGES[event_type],
            evidence_path=evidence_path,
        )
        refresh_session_ml_assessment(self.session_id)
        return ALERT_MESSAGES[event_type]

    def record_browser_event(self, event_type: str, message: str) -> None:
        if event_type not in SUSPICION_POINTS:
            return
        if not self._can_emit(event_type):
            return
        add_alert(
            self.session_id,
            event_type,
            SUSPICION_POINTS[event_type],
            message,
            evidence_path=None,
        )
        refresh_session_ml_assessment(self.session_id)
        with self.lock:
            self.latest_stats["new_alerts"] = [message]

    def process_frame(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        self.frame_counter += 1
        annotated = image.copy()

        if not force and self.frame_counter % FRAME_ANALYSIS_INTERVAL != 0:
            return annotated

        new_alerts: list[str] = []
        faces = detect_faces(image)
        for x, y, width, height in faces:
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (59, 130, 246), 2)
            cv2.putText(
                annotated,
                "Face",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (147, 197, 253),
                2,
            )

        if len(faces) == 0:
            alert = self._emit_alert("no_face", annotated)
            if alert:
                new_alerts.append(alert)
        elif len(faces) > 1:
            alert = self._emit_alert("multiple_faces", annotated)
            if alert:
                new_alerts.append(alert)

        analysis_pass = max(self.frame_counter // FRAME_ANALYSIS_INTERVAL, 1)
        run_phone_detection = force or analysis_pass == 1 or analysis_pass % PHONE_ANALYSIS_INTERVAL == 0
        fresh_phone_detections: list[dict[str, Any]] = []
        if run_phone_detection:
            fresh_phone_detections = PHONE_DETECTOR.detect(image)
            self.last_phone_detections = fresh_phone_detections
        phones = self.last_phone_detections
        for item in phones:
            x, y, width, height = item["bbox"]
            label = f"Phone {item['confidence']:.2f}"
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (239, 68, 68), 2)
            cv2.putText(
                annotated,
                label,
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (254, 202, 202),
                2,
            )
        if fresh_phone_detections:
            alert = self._emit_alert("mobile_phone", annotated)
            if alert:
                new_alerts.append(alert)

        with self.lock:
            self.latest_stats = {
                "face_count": len(faces),
                "phone_count": len(phones),
                "last_processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "new_alerts": new_alerts,
            }
        return annotated


class MonitorVideoProcessor(VideoProcessorBase):
    def __init__(self, monitor: SessionMonitor) -> None:
        self.monitor = monitor

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        processed = self.monitor.process_frame(image)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")
