# AIC 2026 local retrieval backend

This repository is the **visual retrieval component**. It accepts final English
CLIP evidence queries from the separate query-planning repository and returns
ranked keyframe candidates. It does not rewrite Vietnamese, classify tasks,
answer Q&A, or submit CSV files.

## Runtime contract

```text
final English CLIP query
  -> CLIP ViT-B/32 text encoder (+ optional text-only LoRA)
  -> local FAISS video.index
  -> index_metadata.json
  -> SearchCandidate(video_id, original frame ID, timestamp, score)
```

There is no Qdrant service in this runtime.

## Artifact bundle

Point `AIC_ARTIFACT_DIR` at one complete bundle received from the teammate:

```text
artifact-bundle/
  video.index
  index_metadata.json
  lora_weights.pt              # optional
  scene.index                  # optional; must have its metadata too
  scene_metadata.json          # optional
```

`video.index` and `index_metadata.json` are mandatory. The metadata must map
each indexed keyframe to the real submission `frame_id`; keyframe filename or
ordinal is not a submission frame ID.

Copy `.env.example` to `.env` and configure, for example:

```env
AIC_ARTIFACT_DIR=D:/AIC/artifacts/clip-b32-raw-v1
AIC_CLIP_MODEL_NAME=ViT-B/32
AIC_USE_LORA=true
AIC_DEVICE=cuda
```

This codebase's LoRA checkpoint format adapts only CLIP's **text encoder**.
It is compatible with raw BTC `ViT-B/32` image features. A checkpoint that
also modified the visual encoder needs a matching custom loader and a
re-indexed `video.index`.

## Call from Python

```python
from backend.retrieval import search_clip_queries

results = search_clip_queries(
    ["a presenter standing on a stage", "a red car entering a stage"],
    top_k=50,
)

for result in results:
    print(result.query_index, result.query_text, result.candidates[:3])
```

Each `QueryRetrievalResult` keeps the query index and its own ranked results,
so the caller can apply its task-specific RRF, Q&A, or TRAKE logic safely.

## Optional offline tools

Install them only when building or enriching artifacts:

```bash
pip install -r requirements-offline.txt
python scripts/extract_btc_data.py
python scripts/import_btc_data.py --with-transcript  # optional ASR
python -m backend.preprocessing.generate_captions    # optional captions
python scripts/build_index_features.py
```

`import_btc_data.py` preserves the organizer's keyframe-to-original-frame
mapping. `build_index_features.py` builds the local FAISS keyframe and scene
indexes from BTC CLIP features. `extract_clip_features.py` is only for the
case where organizer features are unavailable and you deliberately create your
own raw CLIP ViT-B/32 features.

The repository deliberately contains no training loop. Training belongs to the
teammate's training repository; runtime only loads the resulting LoRA
checkpoint when configured.

The next capabilities proposed in the research PDF - OCR/ASR/caption indexes,
dense raw-video frame refinement, Q&A evidence answering, TRAKE DP, local
evaluation, and CSV export - are not implemented here yet. The remaining
offline metadata, transcription, and caption tools are retained because they
directly support those future additions.
