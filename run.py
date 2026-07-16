"""Local launcher: `python run.py`.

This is NOT a Streamlit app itself — it's a convenience wrapper that shells out
to `streamlit run app.py`. On Streamlit Community Cloud you do NOT use this
file; the platform runs `streamlit run app.py` for you directly. Pointing the
cloud at run.py would start a *nested* Streamlit server on a port the platform
doesn't proxy, which is why cloud uses app.py and local can use run.py.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "app.py"
    print(f"[run.py] launching: streamlit run {app_path}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )
