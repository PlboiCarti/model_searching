# Architecture decision: keyframe-only evidence

The current project uses the original supplied keyframes as evidence images;
it does not decode original videos during the first implementation.

```text
E:\AIC2026\
├── keyframes\
│   └── L01_V001\
│       └── 0042.jpg
└── artifacts\
    └── clip-b32-btc-v1\
        ├── artifact_manifest.json
        ├── video.index
        └── index_metadata.json
```

Each FAISS row maps to a portable keyframe reference and a distinct original
submission frame:

```text
keyframe_relpath = L01_V001/0042.jpg
frame_id = 6731
pts_time = 269.24
```

The task repository owns `AIC_KEYFRAME_ROOT=E:/AIC2026/keyframes` and opens
`AIC_KEYFRAME_ROOT / keyframe_relpath` for OCR/VLM/task logic. It must not
replace `frame_id` with the JPEG filename stem.

Keeping `pts_time` supports temporal grouping and TRAKE ordering now, and
leaves a clean path to optional original-video refinement later.
