from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DISPLAY_TIMEZONE = ZoneInfo("Asia/Kolkata")
DISPLAY_TIMEZONE_LABEL = "IST"


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        current = value
    else:
        cleaned = str(value).strip()
        if not cleaned:
            return None
        if cleaned.endswith("Z"):
            cleaned = f"{cleaned[:-1]}+00:00"
        current = datetime.fromisoformat(cleaned)

    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current


def to_display_timezone(value: str | datetime | None) -> datetime | None:
    current = _coerce_datetime(value)
    if current is None:
        return None
    return current.astimezone(DISPLAY_TIMEZONE)


def format_display_timestamp(
    value: str | datetime | None,
    *,
    default: str = "In Progress",
    include_timezone_label: bool = True,
) -> str:
    localized = to_display_timezone(value)
    if localized is None:
        return default

    formatted = localized.strftime("%Y-%m-%d %H:%M:%S")
    if include_timezone_label:
        return f"{formatted} {DISPLAY_TIMEZONE_LABEL}"
    return formatted


def display_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(DISPLAY_TIMEZONE)
