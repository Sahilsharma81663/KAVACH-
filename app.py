from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from kavach.runtime import import_optional_module

try:  # pragma: no cover - depends on local runtime
    streamlit_webrtc = import_optional_module("streamlit_webrtc")
    WebRtcMode = streamlit_webrtc.WebRtcMode
    webrtc_streamer = streamlit_webrtc.webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local runtime
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False
    WEBRTC_ERROR = str(exc)

from kavach.auth import hash_password, validate_email, validate_password_strength, verify_password
from kavach.components.browser_security.component import browser_security
from kavach.config import (
    ADMIN_DEFAULTS,
    ANALYTICS_IMAGE_PATH,
    APP_SUBTITLE,
    APP_TITLE,
    HERO_IMAGE_PATH,
    SECURE_ROOM_IMAGE_PATH,
)
from kavach.database import (
    analytics_frames,
    complete_session,
    create_exam,
    create_session,
    create_student,
    dashboard_metrics,
    delete_exam,
    delete_student,
    fetch_one,
    get_admin_by_username,
    get_exam_by_id,
    get_session_by_id,
    get_student_by_email,
    get_student_by_id,
    init_database,
    list_alerts,
    list_exams,
    list_saved_reports,
    list_sessions,
    list_student_sessions,
    list_students,
    recent_activity,
    session_report_bundle,
    set_student_face,
    update_session_notes,
)
from kavach.monitoring import MonitorVideoProcessor, SessionMonitor, phone_detector_status, refresh_session_ml_assessment
from kavach.reporting import build_session_report_pdf, save_report
from kavach.ui import (
    friendly_dataframe,
    inject_global_styles,
    mask_email,
    mask_identifier,
    metric_cards,
    render_masthead,
    render_three_scene,
    render_top_ribbon,
    render_visual_gallery,
    risk_badge,
    section_banner,
)
from kavach.vision import decode_uploaded_image, detect_faces, extract_primary_face, save_face_assets, verify_face
from kavach.webrtc import rtc_configuration_status, resolved_rtc_configuration

try:  # pragma: no cover - optional dependency
    px = import_optional_module("plotly.express")

    HAS_PLOTLY = True
except Exception:  # pragma: no cover - optional dependency
    px = None
    HAS_PLOTLY = False


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_database()
inject_global_styles()


def initialize_state() -> None:
    defaults = {
        "role": None,
        "user": None,
        "current_page": "Home",
        "pending_face_student_id": None,
        "active_exam_id": None,
        "active_session_id": None,
        "monitor_instances": {},
        "browser_events_seen": {},
        "last_face_score": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialize_state()


def hero_panel() -> None:
    metrics = dashboard_metrics()
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
            <div class="hero-grid">
                <div class="hero-chip">{metrics['total_students']} registered students</div>
                <div class="hero-chip">{metrics['total_exams']} exam templates</div>
                <div class="hero-chip">{metrics['active_sessions']} active monitoring sessions</div>
                <div class="hero-chip">{metrics['total_alerts']} alerts recorded</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_timestamp(value: str | None) -> str:
    if not value:
        return "In Progress"
    cleaned = str(value).replace("T", " ")
    if "+" in cleaned:
        cleaned = cleaned.split("+", maxsplit=1)[0]
    return cleaned


def format_student_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Student": row["name"],
            "Roll Number": mask_identifier(row.get("roll_number", ""), prefix=2, suffix=2),
            "Email": mask_email(row.get("email", "")),
            "Course": row.get("course", ""),
            "Face Status": row.get("face_status", ""),
            "Registered": format_timestamp(row.get("created_at")),
        }
        for row in rows
    ]


def format_student_session_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Exam": row.get("exam_title", ""),
            "Subject": row.get("subject", ""),
            "Status": str(row.get("status", "")).title(),
            "Suspicion Score": row.get("suspicion_score", 0),
            "Risk": row.get("risk_level", ""),
            "Started": format_timestamp(row.get("start_time")),
            "Ended": format_timestamp(row.get("end_time")),
        }
        for row in rows
    ]


def format_session_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Session": f"S-{row.get('id')}",
            "Student": row.get("student_name", ""),
            "Roll Number": mask_identifier(row.get("roll_number", ""), prefix=2, suffix=2),
            "Exam": row.get("exam_title", ""),
            "Status": str(row.get("status", "")).title(),
            "Suspicion Score": row.get("suspicion_score", 0),
            "Rule Risk": row.get("risk_level", ""),
            "ML Risk": row.get("ml_risk_level", ""),
            "Alerts": row.get("total_alerts", 0),
            "Started": format_timestamp(row.get("start_time")),
            "Updated": format_timestamp(row.get("updated_at")),
        }
        for row in rows
    ]


def format_alert_rows(rows: list[dict], *, include_student: bool = False) -> list[dict]:
    formatted: list[dict] = []
    for row in rows:
        item = {
            "Time": format_timestamp(row.get("created_at")),
            "Alert": str(row.get("alert_type", "")).replace("_", " ").title(),
            "Points": row.get("points", 0),
            "Message": row.get("message", ""),
        }
        if include_student:
            item["Student"] = row.get("student_name", "")
            item["Exam"] = row.get("exam_title", "")
        formatted.append(item)
    return formatted


def format_recent_activity_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Time": format_timestamp(row.get("created_at")),
            "Student": row.get("student_name", ""),
            "Exam": row.get("exam_title", ""),
            "Alert": str(row.get("alert_type", "")).replace("_", " ").title(),
            "Points": row.get("points", 0),
            "Summary": row.get("message", ""),
        }
        for row in rows
    ]


def format_saved_report_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Report": row.get("name", ""),
            "Generated": format_timestamp(row.get("created_at")),
            "Size (KB)": row.get("size_kb", 0),
        }
        for row in rows
    ]


def masthead_image_for_page(page: str, role: str | None) -> Path:
    if role == "student" and page in {"Dashboard", "Face Registration", "Exam Console"}:
        return SECURE_ROOM_IMAGE_PATH
    if role == "admin":
        return ANALYTICS_IMAGE_PATH
    return HERO_IMAGE_PATH


def masthead_status_line() -> str:
    role = st.session_state.role
    user = st.session_state.user
    if role == "admin" and user:
        return f"Admin workspace active · {user['name']}"
    if role == "student" and user:
        return f"Student workspace active · {user['name']}"
    return "Secure assessment operations with live AI monitoring, browser control, and analytics."


def set_page(page_name: str) -> None:
    st.session_state.current_page = page_name
    st.rerun()


def clear_auth() -> None:
    st.session_state.role = None
    st.session_state.user = None
    st.session_state.active_exam_id = None
    st.session_state.active_session_id = None
    st.session_state.current_page = "Home"
    st.rerun()


def get_monitor(session_id: int) -> SessionMonitor:
    monitors = st.session_state.monitor_instances
    if session_id not in monitors:
        monitors[session_id] = SessionMonitor(session_id=session_id)
    return monitors[session_id]


def format_countdown(start_time: str, duration_minutes: int) -> tuple[str, int]:
    started = datetime.fromisoformat(start_time)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    end_time = started + timedelta(minutes=int(duration_minutes))
    remaining = end_time - datetime.now(timezone.utc)
    total_seconds = int(remaining.total_seconds())
    minutes, seconds = divmod(max(total_seconds, 0), 60)
    return f"{minutes:02d}:{seconds:02d}", total_seconds


@st.fragment(run_every="3s")
def exam_timer_fragment(session_id: int, duration_minutes: int, notes_key: str) -> None:
    session = get_session_by_id(session_id)
    if not session or session["status"] != "active":
        return
    countdown, seconds_left = format_countdown(session["start_time"], duration_minutes)
    started = datetime.fromisoformat(session["start_time"])
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    level, confidence = refresh_session_ml_assessment(
        session_id,
        elapsed_minutes=max(int((datetime.now(timezone.utc) - started).total_seconds() // 60), 0),
        persist=False,
    )
    score_items = [
        ("Time Remaining", countdown, "Live countdown"),
        ("Suspicion Score", str(session["suspicion_score"]), "Rule based score"),
        ("Rule Risk", session["risk_level"], "Threshold classification"),
        ("ML Risk", f"{level} ({confidence:.0%})", "Predictive classification"),
    ]
    metric_cards(score_items)
    if seconds_left <= 0:
        complete_session(session_id, st.session_state.get(notes_key, ""))
        st.session_state.active_session_id = None
        st.session_state.active_exam_id = None
        st.warning("Time is up. The session has been submitted automatically.")
        st.session_state.current_page = "Dashboard"
        st.rerun()


@st.fragment(run_every="5s")
def session_alerts_fragment(session_id: int) -> None:
    alerts = list_alerts(session_id=session_id, limit=8)
    if alerts:
        friendly_dataframe(format_alert_rows(alerts))
    else:
        st.info("No suspicious activity has been recorded for this session yet.")


@st.fragment(run_every="5s")
def live_sessions_fragment() -> None:
    sessions = list_sessions(status="active")
    if sessions:
        friendly_dataframe(format_session_rows(sessions))
    else:
        st.info("No active sessions are running right now.")


def render_home_page() -> None:
    hero_panel()
    render_three_scene(height=330, key="home")
    section_banner(
        "Operational Overview",
        "Kavach combines exam access control, browser hardening, webcam analytics, alert logging, and admin-side review.",
    )
    render_visual_gallery(
        [
            (
                HERO_IMAGE_PATH,
                "Exam Monitoring Hub",
                "A cleaner overview of camera verification, browser control, and AI-backed session trust.",
            ),
            (
                ANALYTICS_IMAGE_PATH,
                "Risk & Analytics",
                "Session scoring, integrity signals, and operational review stay visible without exposing sensitive records.",
            ),
            (
                SECURE_ROOM_IMAGE_PATH,
                "Candidate Workspace",
                "A focused exam setup that keeps the student experience calm while the monitoring layer stays active.",
            ),
        ]
    )
    metric_cards(
        [
            ("Authentication", "Dual Step", "Password + face verification"),
            ("Monitoring", "Real Time", "Webcam, fullscreen, focus, and alerts"),
            ("Analytics", "Live", "Risk scoring, session review, and reports"),
        ]
    )

    col1, col2 = st.columns((1.15, 0.85))
    with col1:
        st.subheader("Platform Coverage")
        st.markdown(
            """
            - Student registration with duplicate prevention and face enrollment
            - Face-based login verification before exam access
            - Admin exam creation, monitoring, analytics, and PDF reports
            - Suspicion scoring for no face, multiple faces, tab switch, fullscreen exit, and phone detection
            - Live exam console with answer drafting and session persistence
            """
        )
    with col2:
        st.subheader("Privacy-First Surface")
        st.markdown(
            """
            - Public landing views no longer expose default credentials
            - Student email and roll data are masked in dashboards
            - Runtime traces and file-system paths stay hidden from the exam surface
            """
        )
        st.caption("Administrative credentials are now intentionally absent from the public UI.")


def render_student_registration_page() -> None:
    section_banner(
        "Student Registration",
        "Create a new student identity. Successful registration routes directly to face enrollment.",
    )
    with st.form("student-registration-form"):
        name = st.text_input("Full Name")
        roll_number = st.text_input("Roll Number")
        email = st.text_input("Email")
        course = st.text_input("Course")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register Student", width="stretch")

    if not submitted:
        return

    if not all([name.strip(), roll_number.strip(), email.strip(), course.strip(), password, confirm_password]):
        st.error("Please complete every field before submitting.")
        return
    if password != confirm_password:
        st.error("Password confirmation does not match.")
        return
    if not validate_email(email):
        st.error("Please enter a valid email address.")
        return
    password_ok, password_message = validate_password_strength(password)
    if not password_ok:
        st.error(password_message)
        return
    if get_student_by_email(email):
        st.error("A student with this email already exists.")
        return
    if fetch_one("SELECT id FROM students WHERE roll_number = ?", (roll_number.strip(),)):
        st.error("A student with this roll number already exists.")
        return

    student_id = create_student(
        name=name,
        roll_number=roll_number,
        email=email,
        course=course,
        password_hash=hash_password(password),
    )
    st.session_state.pending_face_student_id = student_id
    st.success("Student created successfully. Continue with face registration.")
    set_page("Face Registration")


def render_face_registration_page() -> None:
    target_student_id = st.session_state.pending_face_student_id
    if st.session_state.role == "student" and st.session_state.user:
        target_student_id = st.session_state.user["id"]

    if not target_student_id:
        st.info("Register a student first, or sign in as a student to update face data.")
        return

    student = get_student_by_id(target_student_id)
    if not student:
        st.error("Student record not found.")
        return

    section_banner(
        "AI Face Registration",
        f"Capture a clear face image for {student['name']}. A single face must be visible before it can be stored.",
    )
    camera_image = st.camera_input("Capture Face Image")
    if st.button("Save Registered Face", width="stretch"):
        image = decode_uploaded_image(camera_image)
        if image is None:
            st.error("Please capture an image from the webcam first.")
            return
        faces = detect_faces(image)
        if len(faces) == 0:
            st.error("No face was detected. Please recapture in better lighting.")
            return
        if len(faces) > 1:
            st.error("Multiple faces were detected. Please capture only the candidate.")
            return

        face_image, _ = extract_primary_face(image)
        face_path, signature_path = save_face_assets(target_student_id, face_image)
        set_student_face(target_student_id, face_path, signature_path)
        if st.session_state.role == "student":
            st.session_state.user = get_student_by_id(target_student_id)
            st.success("Face profile updated successfully.")
            set_page("Dashboard")
        else:
            st.session_state.pending_face_student_id = None
            st.success("Face registered successfully. Continue to student login.")
            set_page("Student Login")


def render_student_login_page() -> None:
    section_banner(
        "Student Login",
        "Password and face verification are both required before a student can access the exam dashboard.",
    )
    email = st.text_input("Student Email")
    password = st.text_input("Password", type="password")
    camera_image = st.camera_input("Face Verification Capture")
    if st.button("Verify and Login", width="stretch"):
        student = get_student_by_email(email)
        if not student or not verify_password(password, student["password_hash"]):
            st.error("Invalid email or password.")
            return
        if not student["face_path"] or not student["face_signature_path"]:
            st.error("This student has not completed face registration yet.")
            st.session_state.pending_face_student_id = student["id"]
            return
        live_image = decode_uploaded_image(camera_image)
        if live_image is None:
            st.error("Please capture a live webcam image to continue.")
            return
        result = verify_face(live_image, student["face_path"], student["face_signature_path"])
        st.session_state.last_face_score = result["score"]
        if not result["match"]:
            st.error(f"{result['reason']} Match score: {result['score']:.2f}%")
            return
        st.success(f"Face verified. Match score: {result['score']:.2f}%")
        st.session_state.role = "student"
        st.session_state.user = get_student_by_id(student["id"])
        st.session_state.current_page = "Dashboard"
        st.rerun()

    if st.session_state.last_face_score is not None:
        st.progress(min(float(st.session_state.last_face_score) / 100.0, 1.0))
        st.caption(f"Latest verification score: {st.session_state.last_face_score:.2f}%")


def render_admin_login_page() -> None:
    section_banner(
        "Admin Login",
        "Administrative access unlocks student management, live monitoring, analytics, and reporting.",
    )
    username = st.text_input("Admin Username")
    password = st.text_input("Admin Password", type="password")
    if st.button("Login as Admin", width="stretch"):
        admin = get_admin_by_username(username)
        if not admin or not verify_password(password, admin["password_hash"]):
            st.error("Invalid admin credentials.")
            return
        st.session_state.role = "admin"
        st.session_state.user = admin
        st.session_state.current_page = "Admin Overview"
        st.rerun()


def start_exam(exam_id: int) -> None:
    student = st.session_state.user
    if not student["face_path"]:
        st.warning("Face registration must be completed before starting an exam.")
        st.session_state.pending_face_student_id = student["id"]
        set_page("Face Registration")
        return
    session_id = create_session(student["id"], exam_id)
    st.session_state.active_exam_id = exam_id
    st.session_state.active_session_id = session_id
    set_page("Exam Console")


def render_student_dashboard_page() -> None:
    student = st.session_state.user
    hero_panel()
    section_banner(
        "Student Dashboard",
        f"Welcome back, {student['name']}. Start an exam, review session history, or update your face profile.",
    )
    exams = list_exams(active_only=True)
    if exams:
        st.subheader("Available Exams")
        for exam in exams:
            with st.container(border=True):
                col1, col2 = st.columns((0.78, 0.22))
                with col1:
                    st.markdown(f"**{exam['title']}**")
                    st.caption(
                        f"{exam['subject']} | {exam['duration_minutes']} minutes | {exam['total_marks']} marks"
                    )
                    st.write(exam["instructions"])
                with col2:
                    st.button(
                        "Start Exam",
                        key=f"start_exam_{exam['id']}",
                        width="stretch",
                        on_click=start_exam,
                        args=(exam["id"],),
                    )
    else:
        st.info("No active exams are available right now.")

    st.subheader("My Sessions")
    friendly_dataframe(format_student_session_rows(list_student_sessions(student["id"])))


def render_exam_console_page() -> None:
    session_id = st.session_state.active_session_id
    exam_id = st.session_state.active_exam_id
    if not session_id or not exam_id:
        st.info("Start an exam from the dashboard to open the exam console.")
        return

    session = get_session_by_id(session_id)
    exam = get_exam_by_id(exam_id)
    if not session or not exam or session["status"] != "active":
        st.warning("This exam session is no longer active.")
        st.session_state.active_session_id = None
        st.session_state.active_exam_id = None
        return

    notes_key = f"answer_notes_{session_id}"
    if notes_key not in st.session_state:
        st.session_state[notes_key] = session["response_notes"] or ""

    section_banner(
        "Secure Exam Console",
        f"{exam['title']} | {exam['subject']} | {exam['duration_minutes']} minutes",
    )
    st.markdown(risk_badge(session["risk_level"]), unsafe_allow_html=True)
    st.caption(phone_detector_status())

    exam_timer_fragment(session_id, exam["duration_minutes"], notes_key)

    left_column, right_column = st.columns((1.25, 0.75))
    with left_column:
        st.subheader("Exam Workspace")
        st.write(exam["instructions"])
        st.text_area(
            "Candidate Response Notes",
            key=notes_key,
            height=260,
            help="This acts as a working answer sheet in the MVP platform.",
        )
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Save Progress", width="stretch"):
                update_session_notes(session_id, st.session_state[notes_key])
                st.success("Progress saved.")
        with action_col2:
            if st.button("Submit Exam", width="stretch"):
                complete_session(session_id, st.session_state[notes_key])
                refresh_session_ml_assessment(session_id)
                st.session_state.active_session_id = None
                st.session_state.active_exam_id = None
                st.success("Exam submitted successfully.")
                set_page("Dashboard")

        st.subheader("Live Webcam Monitoring")
        st.caption("The camera feed is sampled a little more gently here so live monitoring stays responsive during the exam.")
        st.caption(rtc_configuration_status())
        monitor = get_monitor(session_id)
        if WEBRTC_AVAILABLE and webrtc_streamer and WebRtcMode:
            webrtc_streamer(
                key=f"webrtc-monitor-{session_id}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=resolved_rtc_configuration(),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 960},
                        "height": {"ideal": 540},
                        "frameRate": {"ideal": 15, "max": 20},
                    },
                    "audio": False,
                },
                video_processor_factory=lambda: MonitorVideoProcessor(monitor),
                async_processing=True,
                desired_playing_state=True,
            )
        else:
            st.warning(
                "Live WebRTC monitoring is unavailable in this runtime. "
                "Fallback frame analysis is enabled instead."
            )
            if WEBRTC_ERROR:
                st.caption(f"WebRTC import detail: {WEBRTC_ERROR}")
            st.caption("The app switched to still-frame analysis mode for this runtime.")
            snapshot = st.camera_input("Capture Monitoring Frame")
            if st.button("Analyze Captured Frame", width="stretch"):
                image = decode_uploaded_image(snapshot)
                if image is None:
                    st.error("Capture a frame before running analysis.")
                else:
                    analyzed = monitor.process_frame(image, force=True)
                    st.image(analyzed[:, :, ::-1], caption="Analyzed Monitoring Frame", width="stretch")

    with right_column:
        st.subheader("Browser Security")
        st.caption("Use Start Secure Mode once in this panel. Browsers usually require a direct click before fullscreen can begin.")
        browser_event = browser_security(
            session_id=session_id,
            active=True,
            key=f"browser_security_{session_id}",
            auto_request=True,
            prompt="Click Start Secure Mode once to arm fullscreen monitoring for this exam tab.",
        )
        if browser_event:
            seen = st.session_state.browser_events_seen.setdefault(session_id, set())
            event_token = f"{browser_event.get('event_type')}::{browser_event.get('timestamp')}"
            if event_token not in seen:
                seen.add(event_token)
                monitor = get_monitor(session_id)
                monitor.record_browser_event(
                    browser_event.get("event_type", ""),
                    browser_event.get("message", "Browser activity detected."),
                )
        st.subheader("Latest Monitoring Snapshot")
        snapshot = get_monitor(session_id).snapshot()
        metric_cards(
            [
                ("Visible Faces", str(snapshot.get("face_count", 0)), "Frame level count"),
                ("Phones Seen", str(snapshot.get("phone_count", 0)), "YOLOv8 detection"),
                (
                    "Last Frame Scan",
                    snapshot.get("last_processed_at") or "Waiting",
                    "Computer vision update",
                ),
            ]
        )
        st.subheader("Recent Alerts")
        session_alerts_fragment(session_id)


def render_admin_overview_page() -> None:
    hero_panel()
    render_three_scene(height=280, key="admin")
    metrics = dashboard_metrics()
    section_banner(
        "Admin Monitoring Dashboard",
        "Watch platform activity, manage exams, review risk trends, and generate downloadable reports.",
    )
    metric_cards(
        [
            ("Students", str(metrics["total_students"]), "Registered candidates"),
            ("Exams", str(metrics["total_exams"]), "Configured assessments"),
            ("Sessions", str(metrics["total_sessions"]), "Tracked exam sessions"),
            ("Alerts", str(metrics["total_alerts"]), "Suspicious activity records"),
        ]
    )
    render_visual_gallery(
        [
            (
                ANALYTICS_IMAGE_PATH,
                "Live Oversight",
                "A calmer, brighter control surface for session review and operational triage.",
            ),
            (
                SECURE_ROOM_IMAGE_PATH,
                "Protected Exam Space",
                "Candidate monitoring remains visible without surfacing personal email, passwords, or raw file paths.",
            ),
        ]
    )
    st.subheader("Recent Activity")
    friendly_dataframe(format_recent_activity_rows(recent_activity()))


def render_exams_page() -> None:
    section_banner("Test Creation Module", "Create exam templates and remove outdated ones.")
    with st.form("create-exam-form"):
        title = st.text_input("Test Title")
        subject = st.text_input("Subject")
        duration = st.number_input("Duration (minutes)", min_value=15, max_value=240, step=5, value=60)
        total_marks = st.number_input("Total Marks", min_value=10, max_value=500, step=5, value=100)
        instructions = st.text_area("Instructions", height=120)
        submitted = st.form_submit_button("Create Test", width="stretch")

    if submitted:
        if not all([title.strip(), subject.strip(), instructions.strip()]):
            st.error("Title, subject, and instructions are required.")
        else:
            create_exam(title, subject, int(duration), int(total_marks), instructions)
            st.success("Exam created successfully.")
            st.rerun()

    st.subheader("Existing Tests")
    for exam in list_exams():
        with st.container(border=True):
            col1, col2 = st.columns((0.82, 0.18))
            with col1:
                st.markdown(f"**{exam['title']}**")
                st.caption(
                    f"{exam['subject']} | {exam['duration_minutes']} minutes | {exam['total_marks']} marks"
                )
                st.write(exam["instructions"])
            with col2:
                if st.button("Delete Test", key=f"delete_exam_{exam['id']}", width="stretch"):
                    delete_exam(exam["id"])
                    st.success("Exam deleted.")
                    st.rerun()


def render_students_page() -> None:
    section_banner("Student Management", "Review registered students and enrollment status.")
    students = list_students()
    friendly_dataframe(format_student_rows(students))
    if students:
        student_ids = {
            f"{row['name']} ({mask_identifier(row['roll_number'], prefix=2, suffix=2)})": row["id"]
            for row in students
        }
        selected = st.selectbox("Remove Student", options=["None"] + list(student_ids.keys()))
        if selected != "None" and st.button("Delete Student Record", width="stretch"):
            delete_student(student_ids[selected])
            st.success("Student record deleted.")
            st.rerun()


def render_live_sessions_page() -> None:
    section_banner("Live Session Monitoring", "Track active sessions, suspicion scores, and alert history.")
    live_sessions_fragment()
    sessions = list_sessions(status="active")
    if not sessions:
        return
    options = {f"Session {row['id']} | {row['student_name']} | {row['exam_title']}": row["id"] for row in sessions}
    selected_label = st.selectbox("Inspect Active Session", options=list(options.keys()))
    session_id = options[selected_label]
    session = get_session_by_id(session_id)
    st.markdown(risk_badge(session["risk_level"]), unsafe_allow_html=True)
    st.subheader("Alert History")
    friendly_dataframe(format_alert_rows(list_alerts(session_id=session_id), include_student=True))


def render_altair_bar(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> None:
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(x, sort="-y"),
            y=alt.Y(y),
            color=alt.Color(color).legend(None) if color else alt.value("#60a5fa"),
            tooltip=list(df.columns),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def render_altair_pie(df: pd.DataFrame, theta: str, color: str) -> None:
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=70)
        .encode(theta=theta, color=alt.Color(color), tooltip=list(df.columns))
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def render_analytics_page() -> None:
    section_banner("Analytics Dashboard", "Review interactive session and alert trends across the platform.")
    frames = analytics_frames()
    sessions_df = pd.DataFrame(frames["sessions"])
    alerts_df = pd.DataFrame(frames["alerts"])

    if sessions_df.empty or alerts_df.empty:
        st.info("Analytics will appear after at least one monitored session generates activity.")
        return

    alert_counts = alerts_df.groupby("alert_type", as_index=False).size().rename(columns={"size": "count"})
    session_scores = sessions_df[["student_name", "exam_title", "suspicion_score", "risk_level"]].copy()
    session_scores["session_label"] = session_scores["student_name"] + " - " + session_scores["exam_title"]
    risk_counts = sessions_df.groupby("risk_level", as_index=False).size().rename(columns={"size": "count"})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alert Distribution")
        if HAS_PLOTLY:
            fig = px.bar(alert_counts, x="alert_type", y="count", color="alert_type", title="")
            st.plotly_chart(fig, width="stretch")
        else:
            render_altair_bar(alert_counts, "alert_type", "count", "alert_type")
    with col2:
        st.subheader("Alert Percentage")
        if HAS_PLOTLY:
            fig = px.pie(alert_counts, names="alert_type", values="count", hole=0.45)
            st.plotly_chart(fig, width="stretch")
        else:
            render_altair_pie(alert_counts, "count", "alert_type")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Session Suspicion Score")
        if HAS_PLOTLY:
            fig = px.bar(
                session_scores,
                x="session_label",
                y="suspicion_score",
                color="risk_level",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            render_altair_bar(session_scores, "session_label", "suspicion_score", "risk_level")
    with col4:
        st.subheader("Risk Level Distribution")
        if HAS_PLOTLY:
            fig = px.pie(risk_counts, names="risk_level", values="count", hole=0.4)
            st.plotly_chart(fig, width="stretch")
        else:
            render_altair_pie(risk_counts, "count", "risk_level")


def render_reports_page() -> None:
    section_banner("PDF Report Generation", "Generate downloadable session reports and review stored PDFs.")
    sessions = list_sessions()
    if not sessions:
        st.info("No sessions are available for report generation yet.")
        return

    options = {f"Session {row['id']} | {row['student_name']} | {row['exam_title']}": row["id"] for row in sessions}
    selected_label = st.selectbox("Select Session", options=list(options.keys()))
    selected_session_id = options[selected_label]

    if st.button("Generate PDF Report", width="stretch"):
        bundle = session_report_bundle(selected_session_id)
        if not bundle:
            st.error("Unable to prepare the selected session report.")
        else:
            report_bytes = build_session_report_pdf(bundle)
            report_path = save_report(selected_session_id, report_bytes)
            st.success(f"Report generated: {Path(report_path).name}")
            st.download_button(
                "Download Generated Report",
                data=report_bytes,
                file_name=Path(report_path).name,
                mime="application/pdf",
                width="stretch",
            )

    st.subheader("Saved Reports")
    friendly_dataframe(format_saved_report_rows(list_saved_reports()))


def unauthenticated_navigation() -> str:
    options = ["Home", "Student Registration", "Face Registration", "Student Login", "Admin Login"]
    current = st.session_state.current_page if st.session_state.current_page in options else "Home"
    selected, _ = render_top_ribbon(
        options,
        current,
        key_prefix="public",
        button_labels={
            "Home": "Home",
            "Student Registration": "Register",
            "Face Registration": "Face Setup",
            "Student Login": "Student Sign In",
            "Admin Login": "Admin Sign In",
        },
    )
    return selected


def student_navigation() -> str:
    options = ["Dashboard", "Face Registration", "Exam Console"]
    current = st.session_state.current_page if st.session_state.current_page in options else "Dashboard"
    selected, logout_clicked = render_top_ribbon(
        options,
        current,
        key_prefix="student",
        show_logout=True,
        identity_label=f"Student · {st.session_state.user['name']}",
        button_labels={
            "Dashboard": "Overview",
            "Face Registration": "Face Setup",
            "Exam Console": "Exam Space",
        },
    )
    if logout_clicked:
        clear_auth()
    return selected


def admin_navigation() -> str:
    options = [
        "Admin Overview",
        "Exams",
        "Students",
        "Live Sessions",
        "Analytics",
        "Reports",
    ]
    current = st.session_state.current_page if st.session_state.current_page in options else "Admin Overview"
    selected, logout_clicked = render_top_ribbon(
        options,
        current,
        key_prefix="admin",
        show_logout=True,
        identity_label=f"Admin · {st.session_state.user['name']}",
        button_labels={
            "Admin Overview": "Overview",
            "Exams": "Exams",
            "Students": "Students",
            "Live Sessions": "Live Sessions",
            "Analytics": "Analytics",
            "Reports": "Reports",
        },
    )
    if logout_clicked:
        clear_auth()
    return selected


render_masthead(
    APP_TITLE,
    APP_SUBTITLE,
    status_line=masthead_status_line(),
    image_path=masthead_image_for_page(st.session_state.current_page, st.session_state.role),
)
if st.session_state.role == "student" and st.session_state.user:
    selected_page = student_navigation()
elif st.session_state.role == "admin" and st.session_state.user:
    selected_page = admin_navigation()
else:
    selected_page = unauthenticated_navigation()
st.session_state.current_page = selected_page


page = st.session_state.current_page

if page == "Home":
    render_home_page()
elif page == "Student Registration":
    render_student_registration_page()
elif page == "Face Registration":
    render_face_registration_page()
elif page == "Student Login":
    render_student_login_page()
elif page == "Admin Login":
    render_admin_login_page()
elif page == "Dashboard" and st.session_state.role == "student":
    render_student_dashboard_page()
elif page == "Exam Console" and st.session_state.role == "student":
    render_exam_console_page()
elif page == "Admin Overview" and st.session_state.role == "admin":
    render_admin_overview_page()
elif page == "Exams" and st.session_state.role == "admin":
    render_exams_page()
elif page == "Students" and st.session_state.role == "admin":
    render_students_page()
elif page == "Live Sessions" and st.session_state.role == "admin":
    render_live_sessions_page()
elif page == "Analytics" and st.session_state.role == "admin":
    render_analytics_page()
elif page == "Reports" and st.session_state.role == "admin":
    render_reports_page()
else:
    st.warning("Please sign in with the correct role to access this section.")
