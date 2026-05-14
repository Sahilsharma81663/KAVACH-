import streamlit.components.v1 as components

from kavach.config import BROWSER_COMPONENT_DIR

_browser_security_component = components.declare_component(
    "browser_security",
    path=str(BROWSER_COMPONENT_DIR),
)


def browser_security(
    session_id: int,
    active: bool,
    key: str,
    *,
    auto_request: bool = True,
    prompt: str = "",
):
    return _browser_security_component(
        sessionId=str(session_id),
        active=active,
        autoRequest=auto_request,
        prompt=prompt,
        key=key,
        default=None,
    )
