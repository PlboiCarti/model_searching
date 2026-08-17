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

`video.index` and `index_metadata.json` are mandatory. Every metadata row must
have `video_id`, the real submission `frame_id`, and non-negative `pts_time`.
Keyframe filename or ordinal is not a submission frame ID.

Copy `.env.example` to `.env` and configure, for example:

```env
AIC_ARTIFACT_DIR=D:/AIC/artifacts/clip-b32-raw-v1
AIC_CLIP_MODEL_NAME=ViT-B/32
AIC_USE_LORA=true
AIC_DEVICE=cuda
```

This codebase accepts only a LoRA checkpoint that adapts CLIP's **text
encoder**. Its `metadata` must declare `clip_model: "ViT-B/32"` and
`adapter_scope: "text_only"`; it is then compatible with raw BTC `ViT-B/32`
image features. A visual-side LoRA is rejected because it requires a matching
re-encoded `video.index`.

## Call from Python

```python
from aic_model_searching import search_clip_queries

results = search_clip_queries(
    ["a presenter standing on a stage", "a red car entering a stage"],
    top_k=50,
)

for result in results:
    print(result.query_index, result.query_text, result.candidates[:3])
```

Each `QueryRetrievalResult` keeps the query index and its own ranked results,
so the caller can apply its task-specific RRF, Q&A, or TRAKE logic safely.

## Optional offline index-building tools

Install them only when constructing local visual artifacts:

```bash
python -m pip install -e ".[offline]"
python scripts/extract_btc_data.py
python scripts/import_btc_data.py
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

This image-only component has no OCR, ASR, caption model, or text index. Dense
raw-video frame refinement is the next visual-only capability; Q&A answering,
TRAKE DP, local evaluation, and CSV export belong to the calling repository.

## Install

Install this repository into the virtual environment of the calling repository:

```powershell
python -m pip install -e D:\VideoQuery\model_searching
```

Use `-e ".[dev]"` while developing this repository, or `-e ".[offline]"`
when building visual artifacts locally. The calling repository imports only the
public API:

For IDEs that expect a `requirements.txt`, select this repository's virtual
environment and run `python -m pip install -r requirements.txt`. That file
intentionally forwards to the `dev` and `offline` extras in `pyproject.toml`,
so dependency versions remain defined in one place.

```python
from aic_model_searching import search_clip_queries
```
