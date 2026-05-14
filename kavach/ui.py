from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from kavach.config import STYLE_PATH


def inject_global_styles() -> None:
    if Path(STYLE_PATH).exists():
        st.markdown(f"<style>{Path(STYLE_PATH).read_text()}</style>", unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, tone) in zip(columns, items):
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-tone">{tone}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def risk_badge(text: str) -> str:
    normalized = text.lower().replace(" ", "-")
    return f'<span class="risk-badge risk-{normalized}">{text}</span>'


def section_banner(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="section-banner">
            <div class="section-title">{title}</div>
            <div class="section-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def friendly_dataframe(rows: list[dict], *, width: str = "stretch") -> None:
    if not rows:
        st.info("No records available yet.")
        return
    st.dataframe(pd.DataFrame(rows), width=width, hide_index=True)


def _image_data_uri(image_path: str | Path | None) -> str:
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().replace(".", "") or "png"
    return f"data:image/{suffix};base64,{encoded}"


def render_masthead(
    title: str,
    subtitle: str,
    *,
    status_line: str,
    image_path: str | Path | None = None,
) -> None:
    data_uri = _image_data_uri(image_path)
    background_style = (
        f"background-image: linear-gradient(110deg, rgba(255,255,255,0.94), rgba(255,255,255,0.78)), url('{data_uri}');"
        if data_uri
        else ""
    )
    st.markdown(
        f"""
        <section class="masthead-band" style="{background_style}">
            <div class="masthead-copy">
                <div class="masthead-kicker">AI exam integrity platform</div>
                <h1>{html.escape(title)}</h1>
                <p>{html.escape(subtitle)}</p>
            </div>
            <div class="masthead-status">{html.escape(status_line)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_top_ribbon(
    options: list[str],
    current: str,
    *,
    key_prefix: str,
    show_logout: bool = False,
    identity_label: str | None = None,
    button_labels: dict[str, str] | None = None,
) -> tuple[str, bool]:
    selected = current if current in options else options[0]
    logout_clicked = False
    labels = button_labels or {}

    st.markdown(
        "<div class='ribbon-note'>Workspace navigation</div>",
        unsafe_allow_html=True,
    )
    if identity_label:
        st.markdown(
            f"<div class='identity-chip'>{html.escape(identity_label)}</div>",
            unsafe_allow_html=True,
        )

    chunk_size = 4
    option_chunks = [options[index : index + chunk_size] for index in range(0, len(options), chunk_size)]
    for row_index, option_chunk in enumerate(option_chunks):
        column_widths = [1] * len(option_chunk)
        if show_logout and row_index == 0:
            column_widths.append(0.9)
        columns = st.columns(column_widths)
        for index, option in enumerate(option_chunk):
            button_type = "primary" if option == selected else "secondary"
            if columns[index].button(
                labels.get(option, option),
                key=f"{key_prefix}_{option.lower().replace(' ', '_')}",
                width="stretch",
                type=button_type,
            ):
                selected = option
        if show_logout and row_index == 0:
            logout_clicked = columns[-1].button(
                "Logout",
                key=f"{key_prefix}_logout",
                width="stretch",
            )
    return selected, logout_clicked


def render_visual_gallery(items: list[tuple[str | Path, str, str]]) -> None:
    if not items:
        return
    columns = st.columns(len(items))
    for column, (image_path, title, caption) in zip(columns, items):
        with column:
            st.image(str(image_path), width="stretch")
            st.markdown(
                f"""
                <div class="visual-caption">
                    <div class="visual-title">{html.escape(title)}</div>
                    <div class="visual-text">{html.escape(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_three_scene(*, height: int = 340, key: str = "hero-three-scene") -> None:
    scene_id = f"scene-{key}"
    components.html(
        f"""
        <section class="three-scene-shell">
            <canvas id="{scene_id}" style="width:100%; height:{height}px; display:block;"></canvas>
        </section>
        <script>
        (function() {{
            const canvas = document.getElementById("{scene_id}");
            if (!canvas) {{
                return;
            }}

            const ctx = canvas.getContext("2d");
            const sceneHeight = {height};
            const shieldPoints = [
                [0.0, 2.15, 0.0],
                [1.35, 1.22, 0.32],
                [1.75, -0.08, 0.24],
                [1.02, -1.72, 0.15],
                [0.0, -2.48, 0.0],
                [-1.02, -1.72, 0.15],
                [-1.75, -0.08, 0.24],
                [-1.35, 1.22, 0.32],
                [0.0, 0.15, -1.42]
            ];
            const shieldEdges = [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [4, 5], [5, 6], [6, 7], [7, 0],
                [0, 8], [1, 8], [2, 8], [3, 8],
                [4, 8], [5, 8], [6, 8], [7, 8]
            ];
            const shieldFaces = [
                [0, 1, 8], [1, 2, 8], [2, 3, 8], [3, 4, 8],
                [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 0, 8]
            ];
            const starField = Array.from({{ length: 120 }}, () => ({{
                x: (Math.random() - 0.5) * 16,
                y: (Math.random() - 0.5) * 9,
                z: Math.random() * 8 - 4,
                radius: Math.random() * 1.7 + 0.4,
                alpha: Math.random() * 0.55 + 0.2,
            }}));

            let frameHandle = null;
            let width = 0;
            let heightPx = sceneHeight;
            let dpr = 1;

            const resize = () => {{
                width = canvas.clientWidth || canvas.parentElement?.clientWidth || 760;
                heightPx = sceneHeight;
                dpr = Math.min(window.devicePixelRatio || 1, 2);
                canvas.width = width * dpr;
                canvas.height = heightPx * dpr;
                canvas.style.height = `${{heightPx}}px`;
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            }};

            const rotateX = (point, angle) => {{
                const [x, y, z] = point;
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                return [x, y * cos - z * sin, y * sin + z * cos];
            }};

            const rotateY = (point, angle) => {{
                const [x, y, z] = point;
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                return [x * cos + z * sin, y, -x * sin + z * cos];
            }};

            const rotateZ = (point, angle) => {{
                const [x, y, z] = point;
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                return [x * cos - y * sin, x * sin + y * cos, z];
            }};

            const transformPoint = (point, time) => {{
                let nextPoint = rotateY(point, time * 0.85);
                nextPoint = rotateX(nextPoint, 0.38 + Math.sin(time * 0.8) * 0.1);
                nextPoint = rotateZ(nextPoint, Math.sin(time * 0.45) * 0.08);
                nextPoint[1] += Math.sin(time * 1.2) * 0.06;
                return nextPoint;
            }};

            const project = (point) => {{
                const cameraDistance = 8.2;
                const scale = cameraDistance / (cameraDistance - point[2]);
                return {{
                    x: width / 2 + point[0] * scale * 72,
                    y: heightPx / 2 + point[1] * scale * 72,
                    scale,
                    z: point[2],
                }};
            }};

            const drawBackground = () => {{
                const gradient = ctx.createLinearGradient(0, 0, 0, heightPx);
                gradient.addColorStop(0, "rgba(239, 246, 255, 0.98)");
                gradient.addColorStop(0.48, "rgba(248, 250, 252, 0.95)");
                gradient.addColorStop(1, "rgba(224, 242, 254, 0.92)");
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, width, heightPx);

                const glow = ctx.createRadialGradient(width / 2, heightPx / 2 - 12, 10, width / 2, heightPx / 2 - 12, 180);
                glow.addColorStop(0, "rgba(147, 197, 253, 0.34)");
                glow.addColorStop(0.45, "rgba(96, 165, 250, 0.18)");
                glow.addColorStop(1, "rgba(255, 255, 255, 0)");
                ctx.fillStyle = glow;
                ctx.fillRect(0, 0, width, heightPx);
            }};

            const drawStars = (time) => {{
                for (const star of starField) {{
                    const animated = project([
                        star.x,
                        star.y + Math.sin(time + star.x) * 0.08,
                        star.z + Math.cos(time * 0.4 + star.y) * 0.18,
                    ]);
                    const radius = star.radius * animated.scale;
                    ctx.beginPath();
                    ctx.fillStyle = `rgba(37, 99, 235, ${{Math.min(star.alpha * animated.scale, 0.72)}})`;
                    ctx.arc(animated.x, animated.y, radius, 0, Math.PI * 2);
                    ctx.fill();
                }}
            }};

            const drawFloorGrid = (time) => {{
                ctx.save();
                ctx.strokeStyle = "rgba(59, 130, 246, 0.12)";
                ctx.lineWidth = 1;
                const horizon = heightPx * 0.67;
                for (let row = 0; row < 8; row += 1) {{
                    const depth = row / 7;
                    const y = horizon + depth * depth * 112;
                    ctx.beginPath();
                    ctx.moveTo(width * 0.12, y);
                    ctx.lineTo(width * 0.88, y);
                    ctx.stroke();
                }}
                for (let column = -5; column <= 5; column += 1) {{
                    const offset = column * 54 + Math.sin(time * 0.6 + column) * 2;
                    ctx.beginPath();
                    ctx.moveTo(width / 2 + offset * 2.4, horizon + 112);
                    ctx.lineTo(width / 2 + offset, horizon - 14);
                    ctx.stroke();
                }}
                ctx.restore();
            }};

            const drawRing = (time, radius, tiltX, tiltY, color, alpha, phase) => {{
                ctx.beginPath();
                let firstPoint = true;
                for (let step = 0; step <= 120; step += 1) {{
                    const angle = (step / 120) * Math.PI * 2 + time * 0.45 + phase;
                    let point = [Math.cos(angle) * radius, Math.sin(angle) * radius * 0.34, 0];
                    point = rotateX(point, tiltX);
                    point = rotateY(point, tiltY + time * 0.2);
                    const projected = project(point);
                    if (firstPoint) {{
                        ctx.moveTo(projected.x, projected.y);
                        firstPoint = false;
                    }} else {{
                        ctx.lineTo(projected.x, projected.y);
                    }}
                }}
                ctx.strokeStyle = `rgba(${{color.join(",")}}, ${{alpha}})`;
                ctx.lineWidth = 1.4;
                ctx.stroke();
            }};

            const drawShield = (time) => {{
                const transformed = shieldPoints.map((point) => transformPoint([...point], time));
                const projected = transformed.map(project);
                const outlineOrder = [0, 1, 2, 3, 4, 5, 6, 7];

                const fill = ctx.createLinearGradient(width / 2, heightPx / 2 - 160, width / 2, heightPx / 2 + 180);
                fill.addColorStop(0, "rgba(37, 99, 235, 0.92)");
                fill.addColorStop(0.58, "rgba(56, 189, 248, 0.78)");
                fill.addColorStop(1, "rgba(20, 184, 166, 0.7)");

                ctx.beginPath();
                outlineOrder.forEach((index, order) => {{
                    const point = projected[index];
                    if (order === 0) {{
                        ctx.moveTo(point.x, point.y);
                    }} else {{
                        ctx.lineTo(point.x, point.y);
                    }}
                }});
                ctx.closePath();
                ctx.fillStyle = fill;
                ctx.fill();

                shieldFaces
                    .map((face) => ({{
                        face,
                        depth: face.reduce((total, index) => total + transformed[index][2], 0) / face.length,
                    }}))
                    .sort((left, right) => left.depth - right.depth)
                    .forEach((item) => {{
                        const [a, b, c] = item.face.map((index) => projected[index]);
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y);
                        ctx.lineTo(b.x, b.y);
                        ctx.lineTo(c.x, c.y);
                        ctx.closePath();
                        ctx.fillStyle = `rgba(255, 255, 255, ${{0.04 + Math.max(item.depth + 1.6, 0) * 0.035}})`;
                        ctx.fill();
                    }});

                ctx.strokeStyle = "rgba(186, 230, 253, 0.92)";
                ctx.lineWidth = 1.6;
                for (const [startIndex, endIndex] of shieldEdges) {{
                    const start = projected[startIndex];
                    const end = projected[endIndex];
                    ctx.beginPath();
                    ctx.moveTo(start.x, start.y);
                    ctx.lineTo(end.x, end.y);
                    ctx.stroke();
                }}

                const scanLineY = heightPx / 2 + Math.sin(time * 1.9) * 92;
                ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
                ctx.lineWidth = 2.2;
                ctx.beginPath();
                ctx.moveTo(width / 2 - 108, scanLineY);
                ctx.lineTo(width / 2 + 108, scanLineY);
                ctx.stroke();
            }};

            const render = (tick) => {{
                const time = tick * 0.001;
                ctx.clearRect(0, 0, width, heightPx);
                drawBackground();
                drawStars(time);
                drawFloorGrid(time);
                drawRing(time, 3.2, Math.PI / 2.8, 0.1, [14, 165, 233], 0.24, 0);
                drawRing(time, 2.6, 0.24, Math.PI / 2.9, [20, 184, 166], 0.22, Math.PI / 3);
                drawShield(time);
                frameHandle = requestAnimationFrame(render);
            }};

            resize();
            window.addEventListener("resize", resize, {{ passive: true }});
            frameHandle = requestAnimationFrame(render);
        }})();
        </script>
        """,
        height=height,
        scrolling=False,
    )


def mask_email(email: str) -> str:
    if not email or "@" not in email:
        return "Hidden"
    username, domain = email.split("@", maxsplit=1)
    username_mask = (username[:2] + "*" * max(len(username) - 2, 3)) if len(username) > 2 else (username[:1] + "***")
    domain_name, _, suffix = domain.partition(".")
    domain_mask = (domain_name[:1] + "*" * max(len(domain_name) - 1, 3)) if domain_name else "***"
    return f"{username_mask}@{domain_mask}.{suffix or '***'}"


def mask_identifier(value: str, *, prefix: int = 2, suffix: int = 2) -> str:
    if not value:
        return "Hidden"
    if len(value) <= prefix + suffix:
        return value[:1] + "*" * max(len(value) - 1, 1)
    return f"{value[:prefix]}{'*' * max(len(value) - prefix - suffix, 3)}{value[-suffix:]}"
