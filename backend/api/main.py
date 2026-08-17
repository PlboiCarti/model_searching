import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import (
    DEFAULT_TOP_K,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    INDEX_DIR,
    MAX_TOP_K,
    ROOT_DIR,
    SCENE_FAISS_INDEX_PATH,
    SCENE_METADATA_PATH,
)
from backend.embedding.clip_encoder import encode_text

app = FastAPI(title="AIC 2026 - Video Search Agent API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (ROOT_DIR / "data").exists():
    app.mount("/videos", StaticFiles(directory=str(ROOT_DIR / "data")), name="videos")

keyframe_index = None
scene_index = None
metadata_list: list[dict] = []
scene_metadata_list: list[dict] = []


class ResultItem(BaseModel):
    video_id: str
    video_title: str
    score: float
    start: float
    end: float
    frame_id: str
    clip_id: str
    text: str
    submission: str


class SearchResponse(BaseModel):
    results: List[ResultItem]
    needs_clarification: bool = False
    clarification: Optional[dict] = None


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def _load_indexes() -> None:
    global keyframe_index, scene_index, metadata_list, scene_metadata_list

    metadata_list = _load_json(FAISS_METADATA_PATH)
    scene_metadata_list = _load_json(SCENE_METADATA_PATH)

    if FAISS_INDEX_PATH.exists():
        keyframe_index = faiss.read_index(str(FAISS_INDEX_PATH))
    if SCENE_FAISS_INDEX_PATH.exists():
        scene_index = faiss.read_index(str(SCENE_FAISS_INDEX_PATH))

    if keyframe_index is not None and keyframe_index.ntotal != len(metadata_list):
        print(
            "WARNING: keyframe index/metadata length mismatch: "
            f"{keyframe_index.ntotal} vectors vs {len(metadata_list)} metadata rows"
        )


@app.on_event("startup")
def startup_event():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _load_indexes()
    loaded = keyframe_index.ntotal if keyframe_index is not None else 0
    print(f"Loaded local FAISS keyframe index with {loaded} vectors.")


@app.get("/")
def home():
    return {
        "status": "ok",
        "index_vectors": keyframe_index.ntotal if keyframe_index is not None else 0,
        "metadata_rows": len(metadata_list),
    }


def _scene_candidate_ids(query_vector: np.ndarray, scene_top_k: int) -> set[str]:
    if scene_index is None or not scene_metadata_list:
        return set()

    k = min(max(scene_top_k, 1), scene_index.ntotal)
    _scores, indices = scene_index.search(query_vector, k)
    scene_ids = set()
    for idx in indices[0]:
        if 0 <= idx < len(scene_metadata_list):
            scene_id = scene_metadata_list[idx].get("scene_id")
            if scene_id:
                scene_ids.add(str(scene_id))
    return scene_ids


def _format_submission(item: dict, task_type: str, answer: Optional[str]) -> str:
    video_id = str(item.get("video_id", ""))
    frame_id = str(item.get("frame_id", ""))
    if task_type.lower() == "qa" and answer:
        return f"{video_id}, {frame_id}, {answer}"
    return f"{video_id}, {frame_id}"


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., description="Mo ta su kien can tim"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="So luong ket qua"),
    task_type: str = Query("kis", description="Dang bai: kis, qa, trake"),
    answer: Optional[str] = Query(None, description="Cau tra loi cho dang Q&A"),
    scene_top_k: int = Query(8, ge=0, le=100, description="So scene de loc tho truoc khi xep hang keyframe"),
    clarification_answer: Optional[str] = None,
):
    if keyframe_index is None or keyframe_index.ntotal == 0:
        return SearchResponse(results=[], needs_clarification=False)

    query_vector = encode_text(query, translate=True).reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_vector)

    scene_ids = _scene_candidate_ids(query_vector, scene_top_k) if scene_top_k > 0 else set()
    fetch_k = min(keyframe_index.ntotal, max(top_k * 20, top_k))
    scores, indices = keyframe_index.search(query_vector, fetch_k)

    results: list[ResultItem] = []
    seen_frames: set[tuple[str, str]] = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata_list):
            continue
        item = metadata_list[int(idx)]
        if scene_ids and str(item.get("scene_id", "")) not in scene_ids:
            continue

        video_id = str(item.get("video_id", ""))
        frame_id = str(item.get("frame_id", ""))
        dedupe_key = (video_id, frame_id)
        if dedupe_key in seen_frames:
            continue
        seen_frames.add(dedupe_key)

        pts = float(item.get("pts_time", item.get("start", 0.0)) or 0.0)
        text_desc = item.get("caption") or item.get("text") or item.get("object_tags") or ""
        results.append(
            ResultItem(
                video_id=video_id,
                video_title=str(item.get("video_title", video_id)),
                score=float(score),
                start=max(0.0, pts - 2.0),
                end=pts + 3.0,
                frame_id=frame_id,
                clip_id=str(item.get("clip_id", "")),
                text=str(text_desc),
                submission=_format_submission(item, task_type, answer),
            )
        )
        if len(results) >= top_k:
            break

    return SearchResponse(results=results, needs_clarification=False)
