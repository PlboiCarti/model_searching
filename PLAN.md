# Consumer-only retrieval boundary

## Purpose

This package consumes one pre-built OpenAI CLIP ViT-B/32 FAISS bundle and
returns per-query keyframe candidates. It is intentionally not an artifact
builder or a task-orchestration system.

```text
Query repository
  final English clip_queries
    -> aic_model_searching.search_clip_queries(...)
    -> list[QueryRetrievalResult]
    -> query-owned fusion, task logic, and submission

Artifact producer
  BTC keyframes/features/mapping
    -> versioned CLIP-B/32 artifact bundle
    -> supplied to this package through AIC_ARTIFACT_DIR
```

## In scope

- Load and validate the artifact manifest, keyframe FAISS index, metadata, and
  optional scene pair.
- Encode final English queries using OpenAI CLIP ViT-B/32.
- Optionally load a validated text-only LoRA checkpoint.
- Return provenance-preserving candidates with real `frame_id` and `pts_time`.

## Explicitly out of scope

- Downloading BTC data; extracting keyframes/features; generating metadata;
  scene cutting; or building FAISS indexes.
- OCR, ASR, captions, image re-encoding, PE/SigLIP2, model training, and
  visual-side LoRA.
- Vietnamese query processing, RRF, temporal regions, dense refinement, QA,
  TRAKE, evaluation, and CSV submission.

## Definition of a valid hand-off

The producer hands over one immutable directory described in `README.md`.
Before the query repository uses it, validate:

1. `artifact_manifest.json`, `video.index`, and `index_metadata.json` exist.
2. The manifest declares `ViT-B/32`, `openai_clip_vit_b32`, dimension `512`,
   normalized inner-product vectors, and the exact vector count.
3. FAISS total, metadata row count, and manifest vector count agree.
4. Every candidate row has an original submission `frame_id`, not a keyframe
   ordinal or filename.
5. If LoRA is enabled, its metadata declares the same model and `text_only`.
6. Run at least one real English query after the bundle is installed.

## Consumer release checklist

1. Keep the runtime dependency versions and the artifact manifest together in
   the hand-off record.
2. Run `python -m pytest -q` and `python -m compileall aic_model_searching`.
3. Tag the package release and the artifact bundle separately; do not mutate a
   bundle in place after it has been used for an experiment.
