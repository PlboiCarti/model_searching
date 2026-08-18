"""Pure local CLIP retrieval over a configured FAISS artifact bundle.

The caller owns query planning. Inputs to this module are final English CLIP
queries; it never rewrites or translates user text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from aic_model_searching.config import (
    ARTIFACT_MANIFEST_PATH,
    ARTIFACT_SCHEMA_VERSION,
    CLIP_EMBEDDING_DIMENSION,
    CLIP_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    SCENE_FAISS_INDEX_PATH,
    SCENE_METADATA_PATH,
    USE_LORA,
)
from aic_model_searching.embedding.clip_encoder import encode_text


class RetrievalBundleError(RuntimeError):
    """Raised when configured index artifacts are absent or inconsistent."""


@dataclass(frozen=True)
class SearchCandidate:
    video_id: str
    frame_id: str
    clip_id: str
    pts_time: float
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QueryRetrievalResult:
    query_index: int
    query_text: str
    candidates: list[SearchCandidate]


def _load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RetrievalBundleError(f"Missing {label}: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalBundleError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RetrievalBundleError(f"{label} must be a JSON list of objects: {path}")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RetrievalBundleError(f"Missing {label}: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalBundleError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RetrievalBundleError(f"{label} must be a JSON object: {path}")
    return value


def _validate_artifact_manifest(
    manifest: dict[str, Any],
    *,
    index: Any,
    metadata: list[dict[str, Any]],
    path: Path,
) -> None:
    required_values = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "clip_model": CLIP_MODEL_NAME,
        "image_embedding_space": "openai_clip_vit_b32",
        "embedding_dimension": CLIP_EMBEDDING_DIMENSION,
        "metric": "inner_product",
        "normalized": True,
    }
    for field, expected in required_values.items():
        if manifest.get(field) != expected:
            raise RetrievalBundleError(
                f"artifact manifest has incompatible {field!r}: "
                f"expected {expected!r}, got {manifest.get(field)!r}: {path}"
            )

    vector_count = manifest.get("vector_count")
    if not isinstance(vector_count, int) or isinstance(vector_count, bool):
        raise RetrievalBundleError(f"artifact manifest vector_count must be an integer: {path}")
    if vector_count != index.ntotal or vector_count != len(metadata):
        raise RetrievalBundleError(
            "artifact manifest/index/metadata mismatch: "
            f"manifest={vector_count}, index={index.ntotal}, metadata={len(metadata)}"
        )
    if index.d != CLIP_EMBEDDING_DIMENSION:
        raise RetrievalBundleError(
            "CLIP index dimension mismatch: "
            f"index={index.d}, expected={CLIP_EMBEDDING_DIMENSION}"
        )
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RetrievalBundleError(
            "CLIP index metric mismatch: expected FAISS inner product, "
            f"got metric_type={index.metric_type}"
        )
    adapter = manifest.get("text_encoder_adapter")
    if adapter not in {"none", "text_only_lora"}:
        raise RetrievalBundleError(
            "artifact manifest text_encoder_adapter must be 'none' or 'text_only_lora': "
            f"{path}"
        )


def _validate_keyframe_metadata(rows: list[dict[str, Any]], path: Path) -> None:
    """Reject keyframe artifacts that cannot produce valid AIC candidates."""
    for row_index, item in enumerate(rows):
        for field in ("video_id", "frame_id", "pts_time"):
            value = item.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise RetrievalBundleError(
                    f"keyframe metadata row {row_index} has missing {field!r}: {path}"
                )
        try:
            pts_time = float(item["pts_time"])
        except (TypeError, ValueError) as exc:
            raise RetrievalBundleError(
                f"keyframe metadata row {row_index} has invalid 'pts_time': {path}"
            ) from exc
        if not isfinite(pts_time) or pts_time < 0:
            raise RetrievalBundleError(
                f"keyframe metadata row {row_index} has invalid 'pts_time': {path}"
            )


class LocalClipRetriever:
    """Loads one compatible CLIP/FAISS artifact bundle and searches it."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = FAISS_METADATA_PATH,
        manifest_path: Path = ARTIFACT_MANIFEST_PATH,
        scene_index_path: Path = SCENE_FAISS_INDEX_PATH,
        scene_metadata_path: Path = SCENE_METADATA_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.manifest_path = Path(manifest_path)
        self.scene_index_path = Path(scene_index_path)
        self.scene_metadata_path = Path(scene_metadata_path)

        if not self.index_path.is_file():
            raise RetrievalBundleError(f"Missing keyframe FAISS index: {self.index_path}")
        try:
            self.keyframe_index = faiss.read_index(str(self.index_path))
        except Exception as exc:
            raise RetrievalBundleError(f"Cannot load keyframe FAISS index {self.index_path}: {exc}") from exc

        self.metadata = _load_json_list(self.metadata_path, "keyframe metadata")
        if self.keyframe_index.ntotal != len(self.metadata):
            raise RetrievalBundleError(
                "Keyframe index/metadata mismatch: "
                f"{self.keyframe_index.ntotal} vectors vs {len(self.metadata)} rows"
            )
        _validate_keyframe_metadata(self.metadata, self.metadata_path)
        self.manifest = _load_json_object(self.manifest_path, "artifact manifest")
        _validate_artifact_manifest(
            self.manifest,
            index=self.keyframe_index,
            metadata=self.metadata,
            path=self.manifest_path,
        )
        manifest_uses_lora = self.manifest["text_encoder_adapter"] == "text_only_lora"
        if manifest_uses_lora != USE_LORA:
            raise RetrievalBundleError(
                "LoRA configuration does not match artifact manifest: "
                f"manifest text_encoder_adapter={self.manifest['text_encoder_adapter']!r}, "
                f"AIC_USE_LORA={str(USE_LORA).lower()}"
            )

        has_scene_index = self.scene_index_path.is_file()
        has_scene_metadata = self.scene_metadata_path.is_file()
        if has_scene_index != has_scene_metadata:
            raise RetrievalBundleError(
                "Scene artifacts must be provided together: "
                f"index={self.scene_index_path}, metadata={self.scene_metadata_path}"
            )

        self.scene_index = None
        self.scene_metadata: list[dict[str, Any]] = []
        if has_scene_index:
            try:
                self.scene_index = faiss.read_index(str(self.scene_index_path))
            except Exception as exc:
                raise RetrievalBundleError(f"Cannot load scene FAISS index {self.scene_index_path}: {exc}") from exc
            self.scene_metadata = _load_json_list(self.scene_metadata_path, "scene metadata")
            if self.scene_index.ntotal != len(self.scene_metadata):
                raise RetrievalBundleError(
                    "Scene index/metadata mismatch: "
                    f"{self.scene_index.ntotal} vectors vs {len(self.scene_metadata)} rows"
                )
            if self.scene_index.d != self.keyframe_index.d:
                raise RetrievalBundleError(
                    "Scene and keyframe indexes have different embedding dimensions: "
                    f"{self.scene_index.d} vs {self.keyframe_index.d}"
                )
            if self.scene_index.metric_type != self.keyframe_index.metric_type:
                raise RetrievalBundleError(
                    "Scene and keyframe indexes have different distance metrics: "
                    f"{self.scene_index.metric_type} vs {self.keyframe_index.metric_type}"
                )

    @classmethod
    def from_config(cls) -> "LocalClipRetriever":
        return cls()

    def _scene_candidate_ids(self, query_vector: np.ndarray, scene_top_k: int) -> set[str]:
        if scene_top_k <= 0 or self.scene_index is None:
            return set()
        count = min(scene_top_k, self.scene_index.ntotal)
        if count <= 0:
            return set()
        _scores, indices = self.scene_index.search(query_vector, count)
        return {
            str(self.scene_metadata[int(index)].get("scene_id"))
            for index in indices[0]
            if 0 <= index < len(self.scene_metadata) and self.scene_metadata[int(index)].get("scene_id")
        }

    def search(self, clip_query: str, *, top_k: int = 20, scene_top_k: int = 0) -> list[SearchCandidate]:
        """Search one already-processed English CLIP query."""
        if not isinstance(clip_query, str) or not clip_query.strip():
            raise ValueError("clip_query must be a non-empty final CLIP query")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if scene_top_k < 0:
            raise ValueError("scene_top_k cannot be negative")

        query_vector = encode_text(clip_query).reshape(1, -1).astype("float32")
        if query_vector.shape[1] != self.keyframe_index.d:
            raise RetrievalBundleError(
                "CLIP query/index dimension mismatch: "
                f"query={query_vector.shape[1]}, index={self.keyframe_index.d}. "
                "Check CLIP model, LoRA checkpoint, and artifact bundle."
            )
        faiss.normalize_L2(query_vector)

        scene_ids = self._scene_candidate_ids(query_vector, scene_top_k)
        fetch_k = min(self.keyframe_index.ntotal, max(top_k * 20, top_k))
        scores, indices = self.keyframe_index.search(query_vector, fetch_k)

        candidates: list[SearchCandidate] = []
        seen_frames: set[tuple[str, str]] = set()
        for score, index in zip(scores[0], indices[0]):
            if not 0 <= index < len(self.metadata):
                continue
            item = self.metadata[int(index)]
            if scene_ids and str(item.get("scene_id", "")) not in scene_ids:
                continue

            video_id = str(item.get("video_id", ""))
            frame_id = str(item.get("frame_id", ""))
            frame_key = (video_id, frame_id)
            if frame_key in seen_frames:
                continue
            seen_frames.add(frame_key)

            candidates.append(
                SearchCandidate(
                    video_id=video_id,
                    frame_id=frame_id,
                    clip_id=str(item.get("clip_id", "")),
                    pts_time=float(item.get("pts_time", item.get("start", 0.0)) or 0.0),
                    score=float(score),
                    metadata=dict(item),
                )
            )
            if len(candidates) >= top_k:
                break
        return candidates

    def search_many(
        self,
        clip_queries: list[str],
        *,
        top_k: int = 20,
        scene_top_k: int = 0,
    ) -> list[QueryRetrievalResult]:
        return [
            QueryRetrievalResult(
                query_index=index,
                query_text=query,
                candidates=self.search(query, top_k=top_k, scene_top_k=scene_top_k),
            )
            for index, query in enumerate(clip_queries)
        ]


@lru_cache(maxsize=1)
def get_retriever() -> LocalClipRetriever:
    """Return the configured retrieval bundle, loading FAISS only once."""
    return LocalClipRetriever.from_config()


def search_clip_queries(
    clip_queries: list[str],
    *,
    top_k: int = 20,
    scene_top_k: int = 0,
) -> list[QueryRetrievalResult]:
    """Public function for the upstream query-planning repository."""
    return get_retriever().search_many(clip_queries, top_k=top_k, scene_top_k=scene_top_k)
