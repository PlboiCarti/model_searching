# Handoff contract: CLIP keyframe retrieval

## Scope fixed for v0.2.0

This repository is the visual retrieval backend only:

```text
final English CLIP query
  -> OpenAI CLIP ViT-B/32 text encoder
  -> external FAISS keyframe index
  -> per-query ranked SearchCandidate records
```

It does not own Vietnamese query processing, query expansion, RRF, OCR, VLM,
QA answering, TRAKE dynamic programming, CSV output, training, or artifact
construction.

## Producer contract

The artifact producer supplies a versioned directory through
`AIC_ARTIFACT_DIR` containing:

```text
artifact_manifest.json
video.index
index_metadata.json
optional: lora_weights.pt, scene.index + scene_metadata.json
```

`artifact_manifest.json` uses `schema_version: 2`. `video.index` contains
normalized, 512-dimensional OpenAI CLIP ViT-B/32 image vectors indexed by
inner product. Its row order is immutable.

For vector row `i`, metadata row `i` must contain:

```json
{
  "video_id": "L01_V001",
  "keyframe_relpath": "L01_V001/0042.jpg",
  "frame_id": 6731,
  "pts_time": 269.24
}
```

`keyframe_relpath` identifies the original keyframe JPEG directly inside its
`video_id` directory, relative to a root owned by the task repository.
`frame_id` is the original video frame ID used for submission; it is never the
keyframe filename stem.

## Consumer contract

`search_clip_queries(clip_queries, top_k=N)` returns one
`QueryRetrievalResult` per input query. Each result retains its query index and
contains up to `N` `SearchCandidate` records. The caller resolves keyframe
evidence with:

```text
AIC_KEYFRAME_ROOT / candidate.keyframe_relpath
```

`AIC_KEYFRAME_ROOT` is deliberately not a setting of this package, because the
package does not open image files.

## Compatibility rules

- Adding `keyframe_relpath` to existing, aligned metadata does not change
  image vectors and therefore does not require a FAISS rebuild or LoRA retrain.
- Reordering, deleting, or inserting metadata rows without the identical
  vector operation invalidates the bundle.
- A visual encoder change, visual-side LoRA, projection head, or a different
  CLIP image space requires new image vectors and a new index.
- `text_only_lora` is valid only when trained for the same fixed ViT-B/32 image
  vector space and declared in both the manifest and `AIC_USE_LORA`.

## Verification before handoff

1. Create a Python 3.11 or 3.12 environment; do not use Python 3.14.
2. Install `requirements.txt` from this repository root.
3. Run `python -m pytest -q` and `python -m compileall aic_model_searching`.
4. Load the real bundle and make one English query.
5. Confirm a returned `keyframe_relpath` opens below the task repository's
   keyframe root and its `frame_id` is the corresponding original frame ID.
