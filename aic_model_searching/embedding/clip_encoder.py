"""CLIP encoders shared by artifact construction and local retrieval."""

from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import torch

from aic_model_searching.config import CLIP_MODEL_NAME, DEVICE, KEYFRAME_POSITIONS, LORA_WEIGHTS_PATH, USE_LORA


@lru_cache(maxsize=1)
def _load_clip():
    """Load the configured CLIP model and, optionally, its text-only LoRA."""
    import clip

    model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    model.eval()

    should_use_lora = (USE_LORA == "auto" and LORA_WEIGHTS_PATH.exists()) or USE_LORA == "true"
    if USE_LORA == "true" and not LORA_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Configured LoRA checkpoint not found: {LORA_WEIGHTS_PATH}")
    if should_use_lora and LORA_WEIGHTS_PATH.is_file():
        from aic_model_searching.embedding.lora import load_lora_weights

        metadata = load_lora_weights(model, LORA_WEIGHTS_PATH)
        checkpoint_model = metadata["clip_model"].strip()
        if checkpoint_model != CLIP_MODEL_NAME:
            raise ValueError(
                "LoRA checkpoint/model mismatch: "
                f"checkpoint={checkpoint_model}, configured={CLIP_MODEL_NAME}"
            )
        model.eval()

    return model, preprocess


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    if vec.ndim == 1:
        return vec / max(np.linalg.norm(vec), 1e-8)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-8, None)


def extract_keyframe(clip_path: str) -> Optional[np.ndarray]:
    frames = extract_keyframes(clip_path, positions=(0.5,))
    return frames[0] if frames else None


def extract_keyframes(
    clip_path: str,
    positions: Iterable[float] = KEYFRAME_POSITIONS,
) -> list[np.ndarray]:
    """Extract representative RGB frames from a clip for offline feature work."""
    import cv2

    cap = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    frames: list[np.ndarray] = []
    for pos in positions:
        frame_idx = max(0, min(total - 1, int(total * float(pos))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def encode_image(frame_rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    model, preprocess = _load_clip()
    image = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(image).float()
    return _normalize(embedding.cpu().numpy())[0]


def encode_images(frames_rgb: list[np.ndarray]) -> Optional[np.ndarray]:
    if not frames_rgb:
        return None
    return _normalize(np.mean(np.stack([encode_image(frame) for frame in frames_rgb]), axis=0))


def encode_clip_visual(clip_path: str) -> Optional[np.ndarray]:
    return encode_images(extract_keyframes(clip_path))


def encode_text(text: str) -> np.ndarray:
    """Encode one final CLIP text query without rewriting or translation."""
    import clip

    model, _ = _load_clip()
    tokens = clip.tokenize([text.strip() or "empty video segment"], truncate=True).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(tokens).float()
    return _normalize(embedding.cpu().numpy())[0]
