from __future__ import annotations

from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from kavach.config import REPORTS_DIR
from kavach.time_utils import display_now, format_display_timestamp


def _table(data: list[list[str]], column_widths: list[float]) -> Table:
    table = Table(data, colWidths=column_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build_session_report_pdf(bundle: dict) -> bytes:
    session = bundle["session"]
    alerts = bundle["alerts"]

    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Kavach Session Report", styles["Title"]),
        Spacer(1, 6),
        Paragraph(
            f"Generated at {format_display_timestamp(display_now(), include_timezone_label=True)}",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    summary_rows = [
        ["Field", "Value"],
        ["Student", session["student_name"]],
        ["Roll Number", session["roll_number"]],
        ["Email", session["email"]],
        ["Course", session["course"]],
        ["Exam", session["exam_title"]],
        ["Subject", session["subject"]],
        ["Status", session["status"].title()],
        ["Start Time", format_display_timestamp(session["start_time"])],
        ["End Time", format_display_timestamp(session["end_time"])],
        ["Suspicion Score", str(session["suspicion_score"])],
        ["Rule Based Risk", session["risk_level"]],
        ["ML Risk", f'{session["ml_risk_level"]} ({session["ml_confidence"]:.0%})'],
    ]
    story.append(_table(summary_rows, [48 * mm, 110 * mm]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Exam Instructions", styles["Heading2"]))
    story.append(Paragraph(session["instructions"], styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Alert Summary", styles["Heading2"]))
    if alerts:
        alert_rows = [["Time", "Alert", "Points", "Message"]]
        for alert in alerts:
            alert_rows.append(
                [
                    format_display_timestamp(alert["created_at"]),
                    alert["alert_type"].replace("_", " ").title(),
                    str(alert["points"]),
                    alert["message"],
                ]
            )
        story.append(_table(alert_rows, [34 * mm, 33 * mm, 18 * mm, 85 * mm]))
    else:
        story.append(Paragraph("No alerts were recorded for this session.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Candidate Response Notes", styles["Heading2"]))
    story.append(Paragraph(session["response_notes"] or "No answer notes were submitted.", styles["BodyText"]))

    document.build(story)
    return buffer.getvalue()


def save_report(session_id: int, report_bytes: bytes) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORTS_DIR / f"session_{session_id}_{display_now().strftime('%Y%m%d_%H%M%S')}.pdf"
    target.write_bytes(report_bytes)
    return str(target)
