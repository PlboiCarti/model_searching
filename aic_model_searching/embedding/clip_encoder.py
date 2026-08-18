"""Text encoding for the consumer-only CLIP retrieval runtime."""

from functools import lru_cache

import numpy as np
import torch

from aic_model_searching.config import CLIP_MODEL_NAME, DEVICE, LORA_WEIGHTS_PATH, USE_LORA


@lru_cache(maxsize=1)
def _load_clip():
    """Load the configured CLIP model and, optionally, its text-only LoRA."""
    import clip

    model, _preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    model.eval()

    if USE_LORA and not LORA_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Configured LoRA checkpoint not found: {LORA_WEIGHTS_PATH}")
    if USE_LORA:
        from aic_model_searching.embedding.lora import load_lora_weights

        metadata = load_lora_weights(model, LORA_WEIGHTS_PATH)
        checkpoint_model = metadata["clip_model"].strip()
        if checkpoint_model != CLIP_MODEL_NAME:
            raise ValueError(
                "LoRA checkpoint/model mismatch: "
                f"checkpoint={checkpoint_model}, configured={CLIP_MODEL_NAME}"
            )
        model.eval()

    return model


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    if vec.ndim == 1:
        return vec / max(np.linalg.norm(vec), 1e-8)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-8, None)


def encode_text(text: str) -> np.ndarray:
    """Encode one final CLIP text query without rewriting or translation."""
    import clip

    model = _load_clip()
    tokens = clip.tokenize([text.strip() or "empty video segment"], truncate=True).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_text(tokens).float()
    return _normalize(embedding.cpu().numpy())[0]
