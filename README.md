# AIC 2026 CLIP retrieval consumer

`aic-model-searching` is a consumer-only local retrieval package. It accepts
final English CLIP queries from the query-planning repository and searches a
pre-built, external FAISS artifact bundle. It never creates keyframes,
metadata, CLIP features, or indexes.

## Runtime contract

```text
final English CLIP query
  -> OpenAI CLIP ViT-B/32 text encoder (+ optional text-only LoRA)
  -> external FAISS video.index
  -> external index_metadata.json
  -> QueryRetrievalResult / SearchCandidate
```

The calling repository owns Vietnamese query planning, query fusion, temporal
grouping, QA, TRAKE, refinement, evaluation, and submission output.

## Artifact bundle supplied by the teammate

Set `AIC_ARTIFACT_DIR` to one immutable bundle with this layout:

```text
clip-b32-bundle/
  artifact_manifest.json       # required
  video.index                  # required FAISS inner-product index
  index_metadata.json          # required JSON array, one row per vector
  lora_weights.pt              # required only when the manifest uses text_only_lora
  scene.index                  # optional; only with scene_metadata.json
  scene_metadata.json          # optional; only with scene.index
```

`artifact_manifest.json` is required so the consumer can reject an
incompatible bundle before search. A valid minimal manifest is:

```json
{
  "schema_version": 1,
  "clip_model": "ViT-B/32",
  "image_embedding_space": "openai_clip_vit_b32",
  "embedding_dimension": 512,
  "metric": "inner_product",
  "normalized": true,
  "vector_count": 123456,
  "text_encoder_adapter": "none"
}
```

Use `"text_encoder_adapter": "text_only_lora"` only when the bundle also
contains the matching `lora_weights.pt` and the consumer is configured with
`AIC_USE_LORA=true`.

### Non-negotiable bundle constraints

- `video.index` must contain normalized **OpenAI CLIP ViT-B/32 image vectors**
  in a 512-dimensional inner-product space. A same-sized vector from another
  encoder, a projection head, or a visual-side LoRA is incompatible.
- `index_metadata.json` must be a JSON array in exactly the FAISS vector order.
  Its length must equal `vector_count` and `video.index.ntotal`.
- Every metadata row must contain non-empty `video_id`, the real original
  submission `frame_id` (not a keyframe filename/ordinal), and finite,
  non-negative `pts_time`. `clip_id` and `scene_id` are optional.
- If scene artifacts are provided, both scene files must be present, their row
  count must equal the scene index total, and their dimension must equal the
  keyframe index dimension.
- A LoRA checkpoint must contain `metadata.clip_model == "ViT-B/32"` and
  `metadata.adapter_scope == "text_only"`. A visual-side adapter requires a
  re-encoded image index and is rejected.

Ask the teammate to version the whole bundle together: changing any index,
metadata file, manifest, CLIP checkpoint family, or enabled LoRA produces a
new bundle version.

## Environment configuration

Copy `.env.example` to `.env`. Only these variables are supported:

```env
# Required: directory containing the complete bundle above.
AIC_ARTIFACT_DIR=D:/AIC/artifacts/clip-b32-btc-v1

# Optional: cpu, cuda, or cuda:N. Defaults to CUDA when available, else CPU.
AIC_DEVICE=cuda

# Optional: true only for a bundle declaring text_only_lora; otherwise false.
AIC_USE_LORA=false
```

`AIC_CLIP_MODEL_NAME`, per-file index paths, BTC data paths, and offline-build
settings are intentionally unsupported. The model is fixed to `ViT-B/32` and
bundle filenames are fixed to make compatibility inspectable.

## Install and call

Install into the virtual environment of the calling repository:

```powershell
python -m pip install -e D:\VideoQuery\model_searching
```

For package development, install test dependencies with:

```powershell
python -m pip install -e ".[dev]"
```

Then call the public API:

```python
from aic_model_searching import search_clip_queries

results = search_clip_queries(
    ["a presenter standing on a stage", "a red car entering a stage"],
    top_k=50,
)

for result in results:
    print(result.query_index, result.query_text, result.candidates[:3])
```

Each `QueryRetrievalResult` retains its original query index and ranked
candidates. The caller can therefore apply task-specific rank fusion without
losing provenance.
