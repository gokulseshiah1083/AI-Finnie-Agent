"""Cloud entry point (Streamlit Community Cloud runs THIS file).

Design goals:
  1. Render *something* immediately so the front page is never blank.
  2. Log every boot milestone to stdout so the "Manage app" log panel on
     Streamlit Cloud tells you exactly how far boot got.
  3. Keep this file LIGHT — no heavy ML imports happen here. The expensive
     backend (LangGraph + sentence-transformers/torch + FAISS + market data)
     is loaded lazily, on demand, from inside src/web_app/app.py.
"""
import os
import sys
import time
import traceback
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_T0 = time.time()
print(f"[BOOT] app.py starting | cwd={os.getcwd()} | python={sys.version.split()[0]}", flush=True)

# set_page_config MUST be the first Streamlit command, so we call it here in the
# outer entry point and guard the inner one against a double-call.
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)
print("[BOOT] set_page_config done", flush=True)


def _load_secrets() -> None:
    """Copy Streamlit Cloud secrets into os.environ so the rest of the app,
    which only reads os.environ, works unchanged."""
    for key in ["ANTHROPIC_API_KEY", "ALPHA_VANTAGE_API_KEY"]:
        if os.getenv(key):
            print(f"[BOOT] secret {key}: found in environment", flush=True)
            continue
        try:
            value = st.secrets.get(key)
        except Exception as e:  # st.secrets raises if no secrets file at all
            print(f"[BOOT] secret {key}: st.secrets unavailable ({e})", flush=True)
            value = None
        if isinstance(value, str) and value.strip():
            os.environ[key] = value.strip()
            print(f"[BOOT] secret {key}: loaded from st.secrets", flush=True)
        else:
            print(f"[BOOT] secret {key}: NOT set (feature will be limited)", flush=True)


_load_secrets()

# ── Import the actual app (kept light on purpose) ─────────────────────────────
try:
    print("[BOOT] importing src.web_app.app ...", flush=True)
    from src.web_app.app import render  # noqa: E402

    print(f"[BOOT] src.web_app.app imported in {time.time() - _T0:.2f}s", flush=True)
    render()
    print(f"[BOOT] initial render complete in {time.time() - _T0:.2f}s", flush=True)
except Exception as exc:  # noqa: BLE001 - we want to surface *anything*
    print("[BOOT][FATAL] app failed to import/render:", flush=True)
    traceback.print_exc()
    st.error("The app hit an error while starting up. Details below:")
    st.exception(exc)
    raise
