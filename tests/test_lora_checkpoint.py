import pytest

pytest.importorskip("torch")

from aic_model_searching.embedding.lora import _validate_checkpoint_metadata


def test_text_only_checkpoint_metadata_is_accepted(tmp_path):
    metadata = {"clip_model": "ViT-B/32", "adapter_scope": "text_only"}

    normalized = _validate_checkpoint_metadata(metadata, {}, tmp_path / "text-only.pt")

    assert normalized == metadata


@pytest.mark.parametrize(
    "state",
    [
        {"adapter_scope": "visual"},
        {},
    ],
)
def test_legacy_checkpoint_without_text_only_proof_is_rejected(tmp_path, state):
    with pytest.raises(ValueError, match="adapter_scope"):
        _validate_checkpoint_metadata({"clip_model": "ViT-B/32", **state}, {}, tmp_path / "unsafe.pt")
