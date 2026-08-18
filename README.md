# AIC 2026 CLIP keyframe retrieval

`aic-model-searching` is a **consumer-only** package. It receives final
English CLIP queries, searches one pre-built CLIP ViT-B/32 FAISS index, and
returns ranked keyframe references. It does not train models, create
keyframes, build indexes, open keyframe images, or implement KIS/QA/TRAKE
logic.

## Boundary

```text
Query/task repository                         This package
final English clip_queries  ───────────────►  CLIP ViT-B/32 text encoding
                                               + FAISS keyframe retrieval
                                               ↓
                                         QueryRetrievalResult per query
                                               ↓
task fusion, OCR/VLM, QA, TRAKE DP,          SearchCandidate metadata
and submission output  ◄───────────────────  (no task logic here)
```

A `SearchCandidate` is one coarse keyframe hit:

```text
video_id           L01_V001
keyframe_relpath   L01_V001/0042.jpg
frame_id           6731       # original frame ID for submission
pts_time           269.24
score              0.31       # comparable only within this query
```

`QueryRetrievalResult` is a wrapper for one input query and contains
`query_index`, `query_text`, and its ranked `candidates` list.

## Install

Use Python **3.11 or 3.12**. Python 3.14 is intentionally unsupported by the
package metadata because the current Torch/NumPy/FAISS stack is not a reliable
runtime combination on Windows.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` installs this repository in editable mode together with
the test dependency. For runtime-only installation in another repository:

```powershell
python -m pip install -e D:\VideoQuery\model_searching
```

## Runtime environment

Copy `.env.example` to `.env` and set only these package variables:

```env
AIC_ARTIFACT_DIR=E:/AIC2026/artifacts/clip-b32-btc-v1
AIC_DEVICE=cuda
AIC_USE_LORA=false
```

`AIC_KEYFRAME_ROOT` deliberately does **not** belong here: this package never
opens JPEGs. The task repository that performs OCR/VLM/TRAKE logic owns that
machine-local setting, for example `E:/AIC2026/keyframes`.

## Required artifact bundle

`AIC_ARTIFACT_DIR` must point to one immutable bundle:

```text
clip-b32-btc-v1/
├── artifact_manifest.json       # required
├── video.index                  # required FAISS inner-product index
├── index_metadata.json          # required JSON array, one row per vector
├── lora_weights.pt              # only for text_only_lora
├── scene.index                  # optional, paired with the file below
└── scene_metadata.json          # optional
```

Minimal `artifact_manifest.json`:

```json
{
  "schema_version": 2,
  "clip_model": "ViT-B/32",
  "image_embedding_space": "openai_clip_vit_b32",
  "embedding_dimension": 512,
  "metric": "inner_product",
  "normalized": true,
  "vector_count": 123456,
  "text_encoder_adapter": "none"
}
```

Each `index_metadata.json` row must remain in exactly the same order as its
FAISS vector and contain this minimum contract:

```json
{
  "video_id": "L01_V001",
  "keyframe_relpath": "L01_V001/0042.jpg",
  "frame_id": 6731,
  "pts_time": 269.24
}
```

Rules enforced at load time:

- Vectors are normalized OpenAI CLIP `ViT-B/32` image vectors, 512-dimension,
  with FAISS inner-product metric.
- `vector_count`, FAISS `ntotal`, and metadata row count must agree.
- `keyframe_relpath` is a relative `.jpg`/`.jpeg` POSIX path directly inside
  its `video_id` directory. Never put an absolute local path in the bundle.
- `frame_id` is the non-negative original video frame ID, never the keyframe
  filename/ordinal. `pts_time` is finite and non-negative.
- When `text_encoder_adapter` is `text_only_lora`, the bundle must include the
  matching checkpoint and `AIC_USE_LORA=true`. The checkpoint must declare
  `clip_model: ViT-B/32` and `adapter_scope: text_only`.

Adding `keyframe_relpath` to aligned metadata does **not** require re-training
LoRA or rebuilding FAISS. Rebuilding is required if the image vectors, their
order, the visual encoder, or the embedding space changes.

## Search API

```python
from aic_model_searching import search_clip_queries

results = search_clip_queries(
    ["a presenter on a stage", "a red car entering a stage"],
    top_k=50,
)

for result in results:
    for candidate in result.candidates:
        print(
            result.query_index,
            candidate.video_id,
            candidate.keyframe_relpath,
            candidate.frame_id,
            candidate.pts_time,
        )
```

`top_k` applies to **each** input query. The package preserves query
provenance; the caller must not compare raw CLIP `score` values across
different queries as if they were calibrated.

## Keyframe reference in the task repository

The task repository resolves evidence images with its own local root:

```python
from pathlib import Path

keyframe_root = Path("E:/AIC2026/keyframes")
image_path = keyframe_root / candidate.keyframe_relpath
```

Use `image_path` for OCR/VLM or visual task logic. Keep `candidate.frame_id`
unchanged for submission. `pts_time` remains useful for temporal grouping and
TRAKE ordering even in a keyframe-only baseline.

## Handoff checklist

1. Use Python 3.11 or 3.12 and install `requirements.txt`.
2. Provide a schema-2 bundle with the required manifest, index, and aligned
   metadata.
3. Ensure every metadata row resolves under the task repository's
   `AIC_KEYFRAME_ROOT`.
4. Verify one real English query and manually open at least one returned
   `keyframe_relpath`.
5. Run `python -m pytest -q` and `python -m compileall aic_model_searching`.
6. Version the source package and artifact bundle separately; never mutate a
   bundle in place after measuring an experiment.
