import logging
import logging.config
import sys

from src.core.config import CONFIG

_CONFIGURED = False


def setup_logging(force: bool = False):
    """Configure root logging to stream to stdout.

    Streamlit Community Cloud captures stdout/stderr and shows it in the
    "Manage app" log panel, so everything we log here becomes visible there.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    cfg = CONFIG["logging"]

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(cfg["format"]))

    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg["level"], logging.INFO))
    # Avoid piling up duplicate handlers on Streamlit reruns.
    root.handlers = [handler]

    # Suppress noisy third-party loggers
    for lib in ["httpx", "httpcore", "sentence_transformers", "faiss", "urllib3", "yfinance"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Logging configured (level=%s)", cfg["level"])


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def boot(msg: str):
    """Log a boot-phase milestone to BOTH logging and raw stdout.

    Raw ``print`` guarantees the line shows up in hosted logs even before the
    logging handlers are attached, which is exactly the window where a blank
    screen originates.
    """
    print(f"[BOOT] {msg}", flush=True)
    logging.getLogger("boot").info(msg)
