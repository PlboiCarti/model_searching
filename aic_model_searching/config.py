"""Configuration for the consumer-only CLIP/FAISS retrieval runtime."""
import os
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default.resolve()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


# Tự động nạp file .env từ thư mục gốc dự án
env_file = ROOT_DIR / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

ARTIFACT_DIR = _resolve_env_path("AIC_ARTIFACT_DIR", ROOT_DIR / "artifacts")
ARTIFACT_MANIFEST_PATH = ARTIFACT_DIR / "artifact_manifest.json"
FAISS_INDEX_PATH = ARTIFACT_DIR / "video.index"
FAISS_METADATA_PATH = ARTIFACT_DIR / "index_metadata.json"
SCENE_FAISS_INDEX_PATH = ARTIFACT_DIR / "scene.index"
SCENE_METADATA_PATH = ARTIFACT_DIR / "scene_metadata.json"
LORA_WEIGHTS_PATH = ARTIFACT_DIR / "lora_weights.pt"

CLIP_MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIMENSION = 512
ARTIFACT_SCHEMA_VERSION = 1

_use_lora = os.getenv("AIC_USE_LORA", "false").lower()
if _use_lora not in {"true", "false"}:
    raise ValueError("AIC_USE_LORA must be either true or false")
USE_LORA = _use_lora == "true"

def _get_default_device():
    env_dev = os.getenv("AIC_DEVICE")
    if env_dev:
        return env_dev
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _get_default_device()
