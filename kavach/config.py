import os
from pathlib import Path

BASE_DIR = Path(os.getenv("KAVACH_BASE_DIR", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.getenv("KAVACH_DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("KAVACH_DB_PATH", DATA_DIR / "kavach.db"))
KNOWN_FACES_DIR = Path(os.getenv("KAVACH_KNOWN_FACES_DIR", BASE_DIR / "assets" / "known_faces"))
REPORTS_DIR = Path(os.getenv("KAVACH_REPORTS_DIR", BASE_DIR / "reports"))
EVIDENCE_DIR = REPORTS_DIR / "evidence"
STYLE_PATH = BASE_DIR / "assets" / "styles" / "main.css"
IMAGES_DIR = BASE_DIR / "assets" / "images"
HERO_IMAGE_PATH = IMAGES_DIR / "image3.png"
ANALYTICS_IMAGE_PATH = IMAGES_DIR / "image2.png"
SECURE_ROOM_IMAGE_PATH = IMAGES_DIR / "image1.png"
BROWSER_COMPONENT_DIR = (
    BASE_DIR / "kavach" / "components" / "browser_security" / "frontend"
)

APP_TITLE = "Kavach"
APP_SUBTITLE = "AI Powered Online Exam Monitoring System"

ADMIN_DEFAULTS = {
    "username": "admin",
    "password": "Admin@123",
    "name": "System Administrator",
}

SUSPICION_POINTS = {
    "no_face": 10,
    "multiple_faces": 25,
    "fullscreen_exit": 25,
    "tab_switch": 30,
    "mobile_phone": 40,
}

ALERT_MESSAGES = {
    "no_face": "No face detected in the monitoring frame.",
    "multiple_faces": "Multiple faces detected during the exam.",
    "fullscreen_exit": "Fullscreen mode was exited.",
    "tab_switch": "Browser tab switched or lost focus.",
    "mobile_phone": "Mobile phone detected near the candidate.",
}

RISK_THRESHOLDS = (
    (70, "High Risk"),
    (30, "Medium Risk"),
    (0, "Low Risk"),
)

ALERT_COOLDOWNS = {
    "no_face": 12,
    "multiple_faces": 10,
    "fullscreen_exit": 8,
    "tab_switch": 8,
    "mobile_phone": 12,
}

FACE_MATCH_THRESHOLD = 74.0
FACE_ORB_MATCH_MINIMUM = 0.0
FACE_CORRELATION_MINIMUM = 0.72
FACE_RAW_CORRELATION_MINIMUM = 0.44
FACE_COLOR_SCORE_MINIMUM = 0.65
FACE_STRONG_MATCH_THRESHOLD = 68.0
FACE_STRONG_SIGNATURE_MINIMUM = 0.945
FACE_STRONG_CORRELATION_MINIMUM = 0.84
FACE_STRONG_RAW_CORRELATION_MINIMUM = 0.70
FACE_STRONG_EDGE_MINIMUM = 0.36
FACE_IMAGE_SIZE = (224, 224)
FACE_SIGNATURE_SIZE = (96, 96)
FRAME_ANALYSIS_INTERVAL = 8
PHONE_ANALYSIS_INTERVAL = 3
PHONE_CONFIDENCE_THRESHOLD = 0.40
YOLO_MODEL_NAME = "yolov8n.pt"

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}


def ensure_directories() -> None:
    for path in (DATA_DIR, KNOWN_FACES_DIR, REPORTS_DIR, EVIDENCE_DIR):
        path.mkdir(parents=True, exist_ok=True)


ensure_directories()
