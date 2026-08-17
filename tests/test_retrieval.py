import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from aic_model_searching.retrieval import LocalClipRetriever


def _write_bundle(tmp_path):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    index_path = tmp_path / "video.index"
    faiss.write_index(index, str(index_path))

    metadata_path = tmp_path / "index_metadata.json"
    metadata_path.write_text(
        json.dumps(
            [
                {"video_id": "L01_V001", "frame_id": 100, "clip_id": "kf_0001", "pts_time": 4.0},
                {"video_id": "L01_V002", "frame_id": 200, "clip_id": "kf_0002", "pts_time": 8.0},
            ]
        ),
        encoding="utf-8",
    )
    return index_path, metadata_path


def test_search_many_preserves_query_provenance(tmp_path, monkeypatch):
    index_path, metadata_path = _write_bundle(tmp_path)
    retriever = LocalClipRetriever(index_path=index_path, metadata_path=metadata_path)
    monkeypatch.setattr("aic_model_searching.retrieval.encode_text", lambda _query: np.array([1.0, 0.0]))

    results = retriever.search_many(["a presenter", "a red car"], top_k=1)

    assert [result.query_index for result in results] == [0, 1]
    assert [result.query_text for result in results] == ["a presenter", "a red car"]
    assert [result.candidates[0].frame_id for result in results] == ["100", "100"]


def test_bundle_rejects_mismatched_metadata_length(tmp_path):
    index_path, metadata_path = _write_bundle(tmp_path)
    metadata_path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="index/metadata mismatch"):
        LocalClipRetriever(index_path=index_path, metadata_path=metadata_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("video_id", ""), ("frame_id", None), ("pts_time", "not-a-timestamp")],
)
def test_bundle_rejects_invalid_candidate_metadata(tmp_path, field, value):
    index_path, metadata_path = _write_bundle(tmp_path)
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows[0][field] = value
    metadata_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(RuntimeError, match=field):
        LocalClipRetriever(index_path=index_path, metadata_path=metadata_path)
