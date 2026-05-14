from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch

TEMP_ROOT = Path(tempfile.mkdtemp(prefix="kavach_tests_"))
os.environ["KAVACH_BASE_DIR"] = str(TEMP_ROOT)
os.environ["KAVACH_DATA_DIR"] = str(TEMP_ROOT / "data")
os.environ["KAVACH_DB_PATH"] = str(TEMP_ROOT / "data" / "kavach_test.db")
os.environ["KAVACH_KNOWN_FACES_DIR"] = str(TEMP_ROOT / "assets" / "known_faces")
os.environ["KAVACH_REPORTS_DIR"] = str(TEMP_ROOT / "reports")

from kavach.auth import hash_password, verify_password
from kavach.config import FACE_MATCH_THRESHOLD
from kavach.database import (
    add_alert,
    complete_session,
    create_exam,
    create_session,
    create_student,
    get_connection,
    get_session_by_id,
    init_database,
    session_report_bundle,
    update_session_notes,
)
from kavach.monitoring import refresh_session_ml_assessment
from kavach.reporting import build_session_report_pdf, save_report
from kavach.vision import build_face_signature, compare_face_signatures, verify_face


def reset_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            DELETE FROM alerts;
            DELETE FROM sessions;
            DELETE FROM students;
            DELETE FROM exams;
            DELETE FROM admins;
            """
        )
        connection.commit()
    init_database()


class KavachCoreTests(unittest.TestCase):
    @staticmethod
    def _reference_face() -> np.ndarray:
        face = np.full((224, 224, 3), (120, 140, 175), dtype=np.uint8)
        cv2.rectangle(face, (0, 0), (224, 52), (40, 55, 75), -1)
        cv2.circle(face, (112, 96), 19, (195, 205, 220), -1)
        cv2.circle(face, (76, 96), 15, (50, 60, 70), -1)
        cv2.circle(face, (148, 96), 15, (50, 60, 70), -1)
        cv2.ellipse(face, (112, 156), (44, 18), 0, 0, 180, (45, 55, 65), 4)
        return face

    @staticmethod
    def _impostor_face() -> np.ndarray:
        face = np.full((224, 224, 3), (85, 110, 135), dtype=np.uint8)
        cv2.rectangle(face, (0, 0), (224, 68), (18, 28, 38), -1)
        cv2.circle(face, (112, 90), 22, (150, 165, 185), -1)
        cv2.circle(face, (68, 104), 11, (10, 20, 30), -1)
        cv2.circle(face, (156, 82), 20, (10, 20, 30), -1)
        cv2.line(face, (60, 154), (162, 166), (20, 30, 40), 8)
        cv2.line(face, (82, 72), (132, 70), (25, 35, 45), 5)
        return face

    @classmethod
    def setUpClass(cls) -> None:
        init_database()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    def setUp(self) -> None:
        reset_database()

    def test_password_hash_round_trip(self) -> None:
        password_hash = hash_password("SecurePass123")
        self.assertTrue(verify_password("SecurePass123", password_hash))
        self.assertFalse(verify_password("WrongPass999", password_hash))

    def test_face_signature_self_match_scores_high(self) -> None:
        face = self._reference_face()
        signature = build_face_signature(face)
        score = compare_face_signatures(face, face.copy(), signature)
        self.assertGreaterEqual(score, 95.0)

    def test_face_signature_rejects_different_face(self) -> None:
        stored_face = self._reference_face()
        different_face = self._impostor_face()
        signature = build_face_signature(stored_face)
        score = compare_face_signatures(different_face, stored_face, signature)
        self.assertLess(score, FACE_MATCH_THRESHOLD)

    def test_face_signature_accepts_mirrored_rotated_same_face(self) -> None:
        face = self._reference_face()
        mirrored_face = cv2.flip(face, 1)
        rotation = cv2.getRotationMatrix2D((112, 112), 8, 1.0)
        mirrored_face = cv2.warpAffine(
            mirrored_face,
            rotation,
            (224, 224),
            borderMode=cv2.BORDER_REPLICATE,
        )
        signature = build_face_signature(face)
        score = compare_face_signatures(mirrored_face, face, signature)
        self.assertGreaterEqual(score, FACE_MATCH_THRESHOLD)

    def test_verify_face_rejects_multiple_faces_and_wrong_face(self) -> None:
        stored_face = self._reference_face()
        wrong_face = self._impostor_face()

        stored_face_path = TEMP_ROOT / "assets" / "known_faces" / "verify_face_student.png"
        stored_signature_path = TEMP_ROOT / "assets" / "known_faces" / "verify_face_student.npy"
        cv2.imwrite(str(stored_face_path), stored_face)
        np.save(stored_signature_path, build_face_signature(stored_face))

        with patch("kavach.vision.detect_faces", return_value=[(0, 0, 224, 224)]):
            wrong_face_result = verify_face(
                wrong_face,
                str(stored_face_path),
                str(stored_signature_path),
            )

        self.assertFalse(wrong_face_result["match"])
        self.assertLess(wrong_face_result["score"], FACE_MATCH_THRESHOLD)

        with patch("kavach.vision.detect_faces", return_value=[(0, 0, 224, 224), (10, 10, 80, 80)]):
            multiple_face_result = verify_face(
                stored_face,
                str(stored_face_path),
                str(stored_signature_path),
            )

        self.assertFalse(multiple_face_result["match"])
        self.assertIn("Multiple faces", multiple_face_result["reason"])

    def test_verify_face_accepts_borderline_same_face_with_lighting_shift(self) -> None:
        stored_face = self._reference_face()
        stored_face_path = TEMP_ROOT / "assets" / "known_faces" / "verify_face_borderline.png"
        stored_signature_path = TEMP_ROOT / "assets" / "known_faces" / "verify_face_borderline.npy"
        cv2.imwrite(str(stored_face_path), stored_face)
        np.save(stored_signature_path, build_face_signature(stored_face))

        transformed_face = cv2.convertScaleAbs(stored_face, alpha=0.82, beta=-16)
        transformed_hsv = cv2.cvtColor(transformed_face, cv2.COLOR_BGR2HSV).astype(np.float32)
        transformed_hsv[..., 1] = np.clip(transformed_hsv[..., 1] * 0.55, 0, 255)
        transformed_face = cv2.cvtColor(transformed_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        rotation = cv2.getRotationMatrix2D((112, 112), -10, 1.0)
        transformed_face = cv2.warpAffine(
            transformed_face,
            rotation,
            (224, 224),
            borderMode=cv2.BORDER_REPLICATE,
        )
        translation = np.float32([[1, 0, -10], [0, 1, 0]])
        transformed_face = cv2.warpAffine(
            transformed_face,
            translation,
            (224, 224),
            borderMode=cv2.BORDER_REPLICATE,
        )

        with patch("kavach.vision.detect_faces", return_value=[(0, 0, 224, 224)]):
            result = verify_face(
                transformed_face,
                str(stored_face_path),
                str(stored_signature_path),
            )

        self.assertTrue(result["match"])
        self.assertGreaterEqual(result["score"], 68.0)

    def test_session_report_generation(self) -> None:
        student_id = create_student(
            "Aanya Sharma",
            "CS101",
            "aanya@example.com",
            "B.Tech CSE",
            hash_password("SecurePass123"),
        )
        exam_id = create_exam(
            "Networks Midterm",
            "Computer Networks",
            60,
            100,
            "Remain in fullscreen and stay visible to the webcam.",
        )
        session_id = create_session(student_id, exam_id)
        add_alert(session_id, "tab_switch", 30, "Browser tab lost focus.")
        update_session_notes(session_id, "Draft answer text.")
        complete_session(session_id, "Final answer text.")

        bundle = session_report_bundle(session_id)
        self.assertIsNotNone(bundle)
        report_bytes = build_session_report_pdf(bundle)
        self.assertGreater(len(report_bytes), 1500)

        saved_path = save_report(session_id, report_bytes)
        self.assertTrue(Path(saved_path).exists())

    def test_ml_risk_assessment_updates_session(self) -> None:
        student_id = create_student(
            "Rohan Mehta",
            "CS102",
            "rohan@example.com",
            "B.Tech CSE",
            hash_password("SecurePass123"),
        )
        exam_id = create_exam(
            "AI Proctoring Lab",
            "AI Systems",
            45,
            50,
            "Use only the current tab during the assessment.",
        )
        session_id = create_session(student_id, exam_id)
        add_alert(session_id, "mobile_phone", 40, "Mobile phone detected.")
        add_alert(session_id, "tab_switch", 30, "Tab switch detected.")
        level, confidence = refresh_session_ml_assessment(session_id, elapsed_minutes=25)
        session = get_session_by_id(session_id)

        self.assertIn(level, {"Medium Risk", "High Risk"})
        self.assertGreaterEqual(confidence, 0.4)
        self.assertEqual(session["ml_risk_level"], level)


if __name__ == "__main__":
    unittest.main()
