"""Shared Streamlit rendering pieces.

Mirrors the component convention of the companion AI-Billing-Assistance
project so both apps read — and look — the same way.
"""

import streamlit as st


def render_section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Bank-service-desk style section banner.

    One shared component so every view opens with the same visual language —
    a branded gradient band with a customer-friendly title and a one-line
    explanation of what the section is for.
    """
    subtitle_html = (
        f'<div style="color:rgba(255,255,255,0.85); font-size:0.9rem; '
        f'margin-top:0.25rem;">{subtitle}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
<div style="background:linear-gradient(90deg,#123a6d 0%,#1f5fa8 55%,#2e86c1 100%);
            padding:0.9rem 1.25rem; border-radius:0.65rem; margin-bottom:1rem;
            box-shadow:0 1px 4px rgba(0,0,0,0.15);">
  <div style="color:#ffffff; font-size:1.3rem; font-weight:650; letter-spacing:0.2px;">
    {icon} {title}
  </div>
  {subtitle_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_label_css() -> None:
    """Label sizing: one point smaller than Streamlit's defaults, app-wide."""
    st.markdown(
        """
<style>
div[data-testid="stWidgetLabel"] p { font-size: 13px !important; }
.stButton button p { font-size: 15px !important; }
div[data-testid="stMetricLabel"] p { font-size: 13px !important; }
</style>
""",
        unsafe_allow_html=True,
    )
