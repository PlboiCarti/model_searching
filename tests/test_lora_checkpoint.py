import pytest

pytest.importorskip("torch")
pytest.importorskip("faiss")

from aic_model_searching.embedding.lora import _validate_checkpoint_metadata


def test_legacy_transformer_only_checkpoint_is_inferred_as_text_only(tmp_path):
    metadata = {"clip_model": "ViT-B/32"}
    state = {
        "transformer.resblocks.0.attn.out_proj.lora_A": object(),
        "transformer.resblocks.0.attn.out_proj.lora_B": object(),
    }

    normalized = _validate_checkpoint_metadata(metadata, state, tmp_path / "legacy.pt")

    assert normalized == {"clip_model": "ViT-B/32", "adapter_scope": "text_only"}
    assert "adapter_scope" not in metadata


@pytest.mark.parametrize(
    "state",
    [
        {"visual.transformer.resblocks.0.attn.out_proj.lora_A": object()},
        {"transformer.resblocks.0.attn.out_proj.weight": object()},
        {},
    ],
)
def test_legacy_checkpoint_without_text_only_proof_is_rejected(tmp_path, state):
    with pytest.raises(ValueError, match="not provably text-only"):
        _validate_checkpoint_metadata({"clip_model": "ViT-B/32"}, state, tmp_path / "unsafe.pt")
