"""Runtime loader for the text-only CLIP LoRA checkpoint format."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_TEXT_ONLY_SCOPE = "text_only"


class LoRALinear(nn.Module):
    """Frozen linear layer plus a low-rank adapter."""

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        device = original.weight.device
        dtype = original.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(original.in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        delta = (self.lora_A @ self.lora_B).T * self.scaling
        return self.original.weight + delta

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.original(value) + (value @ self.lora_A @ self.lora_B) * self.scaling


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
) -> int:
    """Inject adapters into CLIP's text encoder, never its ``visual`` encoder."""
    target_modules = target_modules or ["out_proj", "c_proj"]
    injected_count = 0

    for name, module in list(model.named_modules()):
        if "visual" in name or not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in target_modules:
            continue

        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))
        injected_count += 1

    if injected_count == 0:
        raise ValueError("No supported text-encoder linear modules found for LoRA injection")

    for param_name, parameter in model.named_parameters():
        if "lora_" not in param_name:
            parameter.requires_grad_(False)
    return injected_count


def _validate_checkpoint_metadata(metadata: object, lora_state: dict, path: Path) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError(f"LoRA checkpoint metadata must be an object: {path}")
    clip_model = metadata.get("clip_model")
    if not isinstance(clip_model, str) or not clip_model.strip():
        raise ValueError(f"LoRA checkpoint metadata.clip_model is required: {path}")

    adapter_scope = metadata.get("adapter_scope")
    if adapter_scope is None:
        raise ValueError(
            f"LoRA checkpoint metadata.adapter_scope is required: {path}"
        )
    if adapter_scope != _TEXT_ONLY_SCOPE:
        raise ValueError(
            "LoRA checkpoint metadata.adapter_scope must be 'text_only': "
            f"{path}"
        )
    return metadata


def load_lora_weights(model: nn.Module, path: Path) -> dict:
    """Load one compatible text-only LoRA checkpoint and return its metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid LoRA checkpoint: {path}")
    try:
        rank = checkpoint["rank"]
        alpha = checkpoint["alpha"]
        lora_state = checkpoint["lora_state_dict"]
    except KeyError as exc:
        raise ValueError(f"LoRA checkpoint is missing {exc.args[0]!r}: {path}") from exc
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError(f"LoRA checkpoint rank must be a positive integer: {path}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
        raise ValueError(f"LoRA checkpoint alpha must be positive: {path}")
    if not isinstance(lora_state, dict):
        raise ValueError(f"Invalid LoRA checkpoint state: {path}")
    metadata = _validate_checkpoint_metadata(checkpoint.get("metadata"), lora_state, path)

    if not any(isinstance(module, LoRALinear) for module in model.modules()):
        inject_lora(model, rank=rank, alpha=float(alpha))

    expected_keys = {
        key
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
        for key in (f"{name}.lora_A", f"{name}.lora_B")
    }
    actual_keys = set(lora_state)
    if expected_keys != actual_keys:
        raise ValueError(
            "LoRA checkpoint architecture mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(lora_state[f"{name}.lora_A"].to(module.lora_A))
            module.lora_B.data.copy_(lora_state[f"{name}.lora_B"].to(module.lora_B))

    return metadata
