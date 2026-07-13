import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_secrets() -> None:
    for key in ["ANTHROPIC_API_KEY", "ALPHA_VANTAGE_API_KEY"]:
        if os.getenv(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


_load_secrets()

from src.web_app.app import *  # noqa: F401,F403
