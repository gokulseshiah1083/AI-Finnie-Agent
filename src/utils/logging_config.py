import logging
import logging.config
from src.core.config import CONFIG


def setup_logging():
    cfg = CONFIG["logging"]
    logging.basicConfig(
        level=getattr(logging, cfg["level"]),
        format=cfg["format"],
    )
    # Suppress noisy third-party loggers
    for lib in ["httpx", "httpcore", "sentence_transformers", "faiss"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
