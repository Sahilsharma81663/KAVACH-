from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kavach.auth import hash_password
from kavach.config import ADMIN_DEFAULTS, DB_PATH, REPORTS_DIR, RISK_THRESHOLDS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def risk_level_for_score(score: int) -> str:
    for threshold, label in RISK_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low Risk"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with closing(get_connection()) as connection:
        row = connection.execute(query, params).fetchone()
        return row_to_dict(row)


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with closing(get_connection()) as connection:
        rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def init_database() -> None:
    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                roll_number TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                course TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                face_path TEXT,
                face_signature_path TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                subject TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                total_marks INTEGER NOT NULL,
                instructions TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                exam_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                suspicion_score INTEGER NOT NULL DEFAULT 0,
                risk_level TEXT NOT NULL DEFAULT 'Low Risk',
                ml_risk_level TEXT NOT NULL DEFAULT 'Low Risk',
                ml_confidence REAL NOT NULL DEFAULT 0,
                response_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY(exam_id) REFERENCES exams(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                alert_type TEXT NOT NULL,
                points INTEGER NOT NULL,
                message TEXT NOT NULL,
                evidence_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
            CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
            """
        )
        connection.commit()
    seed_defaults()


def seed_defaults() -> None:
    existing_admin = get_admin_by_username(ADMIN_DEFAULTS["username"])
    if not existing_admin:
        now = utc_now()
        with closing(get_connection()) as connection:
            connection.execute(
                """
                INSERT INTO admins (username, password_hash, name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ADMIN_DEFAULTS["username"],
                    hash_password(ADMIN_DEFAULTS["password"]),
                    ADMIN_DEFAULTS["name"],
                    now,
                ),
            )
            connection.commit()

    existing_exam = fetch_one("SELECT id FROM exams LIMIT 1")
    if not existing_exam:
        create_exam(
            title="Foundations of Secure Computing",
            subject="Computer Science",
            duration_minutes=45,
            total_marks=50,
            instructions=(
                "Answer all questions honestly. Stay visible in the webcam feed, "
                "remain in fullscreen mode, and avoid tab switching."
            ),
        )


def create_student(
    name: str,
    roll_number: str,
    email: str,
    course: str,
    password_hash: str,
) -> int:
    now = utc_now()
    with closing(get_connection()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO students (name, roll_number, email, course, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name.strip(), roll_number.strip(), email.strip().lower(), course.strip(), password_hash, now),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_student_by_email(email: str) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM students WHERE email = ?", (email.strip().lower(),))


def get_student_by_id(student_id: int) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM students WHERE id = ?", (student_id,))


def list_students() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT
            id,
            name,
            roll_number,
            email,
            course,
            CASE WHEN face_path IS NOT NULL THEN 'Registered' ELSE 'Pending' END AS face_status,
            created_at
        FROM students
        ORDER BY created_at DESC
        """
    )


def set_student_face(student_id: int, face_path: str, signature_path: str) -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            UPDATE students
            SET face_path = ?, face_signature_path = ?
            WHERE id = ?
            """,
            (face_path, signature_path, student_id),
        )
        connection.commit()


def delete_student(student_id: int) -> None:
    with closing(get_connection()) as connection:
        connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
        connection.commit()


def get_admin_by_username(username: str) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM admins WHERE username = ?", (username.strip(),))


def create_exam(
    title: str,
    subject: str,
    duration_minutes: int,
    total_marks: int,
    instructions: str,
) -> int:
    now = utc_now()
    with closing(get_connection()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO exams (title, subject, duration_minutes, total_marks, instructions, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title.strip(), subject.strip(), duration_minutes, total_marks, instructions.strip(), now),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_exam_by_id(exam_id: int) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM exams WHERE id = ?", (exam_id,))


def list_exams(active_only: bool = False) -> list[dict[str, Any]]:
    query = "SELECT * FROM exams"
    params: tuple[Any, ...] = ()
    if active_only:
        query += " WHERE is_active = 1"
    query += " ORDER BY created_at DESC"
    return fetch_all(query, params)


def delete_exam(exam_id: int) -> None:
    with closing(get_connection()) as connection:
        connection.execute("DELETE FROM exams WHERE id = ?", (exam_id,))
        connection.commit()


def get_active_session(student_id: int, exam_id: int) -> dict[str, Any] | None:
    return fetch_one(
        """
        SELECT * FROM sessions
        WHERE student_id = ? AND exam_id = ? AND status = 'active'
        ORDER BY id DESC
        LIMIT 1
        """,
        (student_id, exam_id),
    )


def create_session(student_id: int, exam_id: int) -> int:
    existing = get_active_session(student_id, exam_id)
    if existing:
        return int(existing["id"])

    now = utc_now()
    with closing(get_connection()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO sessions (
                student_id,
                exam_id,
                status,
                start_time,
                created_at,
                updated_at
            )
            VALUES (?, ?, 'active', ?, ?, ?)
            """,
            (student_id, exam_id, now, now, now),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_session_by_id(session_id: int) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM sessions WHERE id = ?", (session_id,))


def list_sessions(status: str | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT
            sessions.*,
            students.name AS student_name,
            students.roll_number,
            students.email,
            exams.title AS exam_title,
            exams.subject,
            exams.duration_minutes,
            (
                SELECT COUNT(*)
                FROM alerts
                WHERE alerts.session_id = sessions.id
            ) AS total_alerts
        FROM sessions
        JOIN students ON students.id = sessions.student_id
        JOIN exams ON exams.id = sessions.exam_id
    """
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE sessions.status = ?"
        params = (status,)
    query += " ORDER BY sessions.updated_at DESC"
    return fetch_all(query, params)


def list_student_sessions(student_id: int) -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT
            sessions.*,
            exams.title AS exam_title,
            exams.subject
        FROM sessions
        JOIN exams ON exams.id = sessions.exam_id
        WHERE sessions.student_id = ?
        ORDER BY sessions.updated_at DESC
        """,
        (student_id,),
    )


def complete_session(session_id: int, response_notes: str = "") -> None:
    now = utc_now()
    with closing(get_connection()) as connection:
        connection.execute(
            """
            UPDATE sessions
            SET status = 'completed',
                end_time = ?,
                response_notes = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (now, response_notes.strip(), now, session_id),
        )
        connection.commit()


def update_session_notes(session_id: int, response_notes: str) -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            UPDATE sessions
            SET response_notes = ?, updated_at = ?
            WHERE id = ?
            """,
            (response_notes.strip(), utc_now(), session_id),
        )
        connection.commit()


def add_alert(
    session_id: int,
    alert_type: str,
    points: int,
    message: str,
    evidence_path: str | None = None,
) -> int:
    now = utc_now()
    with closing(get_connection()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO alerts (session_id, alert_type, points, message, evidence_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, alert_type, points, message, evidence_path, now),
        )
        cursor.execute(
            """
            UPDATE sessions
            SET suspicion_score = suspicion_score + ?,
                risk_level = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                points,
                risk_level_for_score(current_session_score(session_id) + points),
                now,
                session_id,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def current_session_score(session_id: int) -> int:
    session = get_session_by_id(session_id)
    if not session:
        return 0
    return int(session["suspicion_score"])


def list_alerts(session_id: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT
            alerts.*,
            students.name AS student_name,
            exams.title AS exam_title
        FROM alerts
        JOIN sessions ON sessions.id = alerts.session_id
        JOIN students ON students.id = sessions.student_id
        JOIN exams ON exams.id = sessions.exam_id
    """
    params: list[Any] = []
    if session_id is not None:
        query += " WHERE alerts.session_id = ?"
        params.append(session_id)
    query += " ORDER BY alerts.created_at DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return fetch_all(query, tuple(params))


def update_session_ml_assessment(session_id: int, level: str, confidence: float) -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            UPDATE sessions
            SET ml_risk_level = ?, ml_confidence = ?, updated_at = ?
            WHERE id = ?
            """,
            (level, confidence, utc_now(), session_id),
        )
        connection.commit()


def dashboard_metrics() -> dict[str, int]:
    with closing(get_connection()) as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM students) AS total_students,
                (SELECT COUNT(*) FROM exams) AS total_exams,
                (SELECT COUNT(*) FROM sessions) AS total_sessions,
                (SELECT COUNT(*) FROM alerts) AS total_alerts,
                (SELECT COUNT(*) FROM sessions WHERE status = 'active') AS active_sessions
            """
        ).fetchone()
        return dict(row)


def recent_activity(limit: int = 8) -> list[dict[str, Any]]:
    return fetch_all(
        f"""
        SELECT
            alerts.created_at,
            alerts.alert_type,
            alerts.message,
            alerts.points,
            students.name AS student_name,
            exams.title AS exam_title
        FROM alerts
        JOIN sessions ON sessions.id = alerts.session_id
        JOIN students ON students.id = sessions.student_id
        JOIN exams ON exams.id = sessions.exam_id
        ORDER BY alerts.created_at DESC
        LIMIT {int(limit)}
        """
    )


def session_report_bundle(session_id: int) -> dict[str, Any] | None:
    session = fetch_one(
        """
        SELECT
            sessions.*,
            students.name AS student_name,
            students.roll_number,
            students.email,
            students.course,
            exams.title AS exam_title,
            exams.subject,
            exams.duration_minutes,
            exams.total_marks,
            exams.instructions
        FROM sessions
        JOIN students ON students.id = sessions.student_id
        JOIN exams ON exams.id = sessions.exam_id
        WHERE sessions.id = ?
        """,
        (session_id,),
    )
    if not session:
        return None

    alerts = list_alerts(session_id=session_id)
    return {"session": session, "alerts": alerts}


def analytics_frames() -> dict[str, list[dict[str, Any]]]:
    return {
        "sessions": list_sessions(),
        "alerts": list_alerts(),
        "students": list_students(),
        "exams": list_exams(),
    }


def list_saved_reports() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for file_path in sorted(REPORTS_DIR.glob("session_*.pdf"), reverse=True):
        reports.append(
            {
                "name": file_path.name,
                "path": str(file_path),
                "created_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
                "size_kb": round(file_path.stat().st_size / 1024, 1),
            }
        )
    return reports
