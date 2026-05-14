from __future__ import annotations

import json
import os
import time
from typing import Any

import streamlit as st

from kavach.config import RTC_CONFIGURATION
from kavach.runtime import import_optional_module

_RTC_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "configuration": RTC_CONFIGURATION,
    "status": "WebRTC transport: default Google STUN only. Add TURN credentials in deployment secrets for hosted live monitoring.",
}


def _secret_value(name: str) -> str | None:
    try:
        value = st.secrets[name]
        if isinstance(value, str):
            value = value.strip()
        return value or None
    except Exception:
        value = os.getenv(name)
        if isinstance(value, str):
            value = value.strip()
        return value or None


def _normalize_ice_servers(raw_value: Any) -> list[dict[str, Any]]:
    if isinstance(raw_value, str):
        raw_value = json.loads(raw_value)

    if isinstance(raw_value, dict):
        raw_value = [raw_value]

    if not isinstance(raw_value, list):
        raise ValueError("ICE server configuration must be a dict or list of dicts.")

    normalized: list[dict[str, Any]] = []
    for item in raw_value:
        if not isinstance(item, dict):
            raise ValueError("Each ICE server entry must be a dictionary.")
        urls = item.get("urls")
        if not urls:
            raise ValueError("Each ICE server entry must include urls.")
        normalized_item: dict[str, Any] = {"urls": urls}
        if item.get("username"):
            normalized_item["username"] = item["username"]
        if item.get("credential"):
            normalized_item["credential"] = item["credential"]
        normalized.append(normalized_item)

    if not normalized:
        raise ValueError("At least one ICE server entry is required.")
    return normalized


def _custom_ice_servers() -> tuple[list[dict[str, Any]] | None, str | None]:
    raw_value = _secret_value("RTC_ICE_SERVERS_JSON") or _secret_value("KAVACH_ICE_SERVERS_JSON")
    if not raw_value:
        return None, None

    ice_servers = _normalize_ice_servers(raw_value)
    return ice_servers, "WebRTC transport: custom ICE server list loaded from deployment secrets."


def _twilio_ice_servers() -> tuple[list[dict[str, Any]] | None, str | None]:
    account_sid = _secret_value("TWILIO_ACCOUNT_SID")
    auth_token = _secret_value("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        return None, None

    twilio_module = import_optional_module("twilio.rest")
    client = twilio_module.Client(account_sid, auth_token)
    token = client.tokens.create()
    ice_servers = _normalize_ice_servers(token.ice_servers)
    return ice_servers, "WebRTC transport: Twilio TURN/STUN loaded from deployment secrets."


def _cache_result(configuration: dict[str, Any], status: str, ttl_seconds: int) -> dict[str, Any]:
    _RTC_CACHE["configuration"] = configuration
    _RTC_CACHE["status"] = status
    _RTC_CACHE["expires_at"] = time.monotonic() + ttl_seconds
    return configuration


def resolved_rtc_configuration() -> dict[str, Any]:
    now = time.monotonic()
    if now < float(_RTC_CACHE["expires_at"]):
        return dict(_RTC_CACHE["configuration"])

    try:
        custom_servers, custom_status = _custom_ice_servers()
        if custom_servers:
            return _cache_result({"iceServers": custom_servers}, custom_status or "WebRTC transport: custom ICE servers loaded.", 900)

        twilio_servers, twilio_status = _twilio_ice_servers()
        if twilio_servers:
            return _cache_result({"iceServers": twilio_servers}, twilio_status or "WebRTC transport: Twilio TURN/STUN loaded.", 900)
    except Exception as exc:
        return _cache_result(
            RTC_CONFIGURATION,
            "WebRTC transport: TURN/ICE secret loading failed, so the app fell back to Google STUN only. "
            f"Detail: {exc}",
            120,
        )

    return _cache_result(
        RTC_CONFIGURATION,
        "WebRTC transport: default Google STUN only. Add TURN credentials in deployment secrets for hosted live monitoring.",
        900,
    )


def rtc_configuration_status() -> str:
    resolved_rtc_configuration()
    return str(_RTC_CACHE["status"])
