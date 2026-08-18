import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from aic_model_searching.retrieval import LocalClipRetriever


@pytest.fixture(autouse=True)
def _disable_lora_for_bundle_tests(monkeypatch):
    monkeypatch.setattr("aic_model_searching.retrieval.USE_LORA", False)


def _write_bundle(tmp_path, manifest_updates=None, index_factory=faiss.IndexFlatIP):
    vectors = np.zeros((2, 512), dtype="float32")
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    index = index_factory(512)
    index.add(vectors)
    index_path = tmp_path / "video.index"
    faiss.write_index(index, str(index_path))

    metadata_path = tmp_path / "index_metadata.json"
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "L01_V001",
                    "keyframe_relpath": "L01_V001/0001.jpg",
                    "frame_id": 100,
                    "pts_time": 4.0,
                },
                {
                    "video_id": "L01_V002",
                    "keyframe_relpath": "L01_V002/0002.jpg",
                    "frame_id": 200,
                    "pts_time": 8.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "clip_model": "ViT-B/32",
        "image_embedding_space": "openai_clip_vit_b32",
        "embedding_dimension": 512,
        "metric": "inner_product",
        "normalized": True,
        "vector_count": 2,
        "text_encoder_adapter": "none",
    }
    manifest.update(manifest_updates or {})
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return index_path, metadata_path, manifest_path


def test_search_many_preserves_query_provenance(tmp_path, monkeypatch):
    index_path, metadata_path, manifest_path = _write_bundle(tmp_path)
    retriever = LocalClipRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
    )
    query_vector = np.zeros(512, dtype="float32")
    query_vector[0] = 1.0
    monkeypatch.setattr("aic_model_searching.retrieval.encode_text", lambda _query: query_vector)

    results = retriever.search_many(["a presenter", "a red car"], top_k=1)

    assert [result.query_index for result in results] == [0, 1]
    assert [result.query_text for result in results] == ["a presenter", "a red car"]
    assert [result.candidates[0].frame_id for result in results] == ["100", "100"]
    assert [result.candidates[0].keyframe_relpath for result in results] == [
        "L01_V001/0001.jpg",
        "L01_V001/0001.jpg",
    ]


def test_bundle_rejects_mismatched_metadata_length(tmp_path):
    index_path, metadata_path, manifest_path = _write_bundle(tmp_path)
    metadata_path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="index/metadata mismatch"):
        LocalClipRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_id", ""),
        ("keyframe_relpath", "../0001.jpg"),
        ("keyframe_relpath", "L01_V002/0001.jpg"),
        ("keyframe_relpath", "L01_V001\\\\0001.jpg"),
        ("keyframe_relpath", "/L01_V001/0001.jpg"),
        ("frame_id", None),
        ("pts_time", "not-a-timestamp"),
    ],
)
def test_bundle_rejects_invalid_candidate_metadata(tmp_path, field, value):
    index_path, metadata_path, manifest_path = _write_bundle(tmp_path)
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows[0][field] = value
    metadata_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(RuntimeError, match=field):
        LocalClipRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clip_model", "ViT-B/16"),
        ("embedding_dimension", 768),
        ("normalized", False),
        ("text_encoder_adapter", "visual_lora"),
    ],
)
def test_bundle_rejects_incompatible_manifest(tmp_path, field, value):
    index_path, metadata_path, manifest_path = _write_bundle(tmp_path, {field: value})

    with pytest.raises(RuntimeError, match="artifact manifest"):
        LocalClipRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )


def test_bundle_rejects_unpaired_scene_artifact(tmp_path):
    index_path, metadata_path, manifest_path = _write_bundle(tmp_path)
    (tmp_path / "scene.index").write_bytes(b"not-a-scene-index")

    with pytest.raises(RuntimeError, match="Scene artifacts must be provided together"):
        LocalClipRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            scene_index_path=tmp_path / "scene.index",
            scene_metadata_path=tmp_path / "scene_metadata.json",
        )


@pytest.mark.parametrize(
    ("manifest_adapter", "use_lora"),
    [("none", True), ("text_only_lora", False)],
)
def test_bundle_rejects_lora_configuration_mismatch(
    tmp_path,
    monkeypatch,
    manifest_adapter,
    use_lora,
):
    index_path, metadata_path, manifest_path = _write_bundle(
        tmp_path,
        {"text_encoder_adapter": manifest_adapter},
    )
    monkeypatch.setattr("aic_model_searching.retrieval.USE_LORA", use_lora)

    with pytest.raises(RuntimeError, match="LoRA configuration"):
        LocalClipRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )


def test_bundle_rejects_non_inner_product_index(tmp_path):
    index_path, metadata_path, manifest_path = _write_bundle(
        tmp_path,
        index_factory=faiss.IndexFlatL2,
    )

    with pytest.raises(RuntimeError, match="metric mismatch"):
        LocalClipRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )
