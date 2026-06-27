import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent.parent


def load_config() -> dict:
    config_path = _ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def get_anthropic_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
    return key


def resolve_path(relative: str) -> Path:
    return _ROOT / relative
