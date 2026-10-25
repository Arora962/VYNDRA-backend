"""Central configuration for the VYNDRA Flask backend."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 128 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.59
NUDGE_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_PORT = int(os.getenv("VYNDRA_PORT", "5001"))
DEFAULT_HOST = os.getenv("VYNDRA_HOST", "0.0.0.0")


def cors_origins() -> list[str]:
    """Read allowed development origins from the environment."""
    raw = os.getenv("VYNDRA_CORS_ORIGINS", "http://localhost:8081,http://127.0.0.1:8081")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
