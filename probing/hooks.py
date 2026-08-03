"""Forward hooks and activation interventions for VWorldModel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


class ActivationHookManager:
    """
    Register forward hooks on encoder / projector / predictor submodules.

    Supports knockout (zero) and mean-replacement interventions during forward.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.registered_names: list[str] = []
        self.activations: dict[str, torch.Tensor] = {}
        self.interventions: dict[str, tuple[str, torch.Tensor | None]] = {}
        self._capture_enabled = True

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.activations.clear()
        self.interventions.clear()

    def set_intervention(
        self,
        name: str,
        mode: str,
        value: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            name: registered hook name.
            mode: ``knockout`` (zero activations) or ``mean`` (replace with ``value``).
            value: required for ``mean``; tensor broadcastable to hooked output.
        """
        if mode not in {"knockout", "mean"}:
            raise ValueError(f"Unknown intervention mode: {mode}")
        if mode == "mean" and value is None:
            raise ValueError("mean intervention requires `value` tensor")
        self.interventions[name] = (mode, value)

    def clear_interventions(self) -> None:
        self.interventions.clear()

    def _make_hook(self, name: str) -> Callable:
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                tensor = output[0]
                rest = output[1:]
            else:
                tensor = output
                rest = None

            if name in self.interventions:
                mode, value = self.interventions[name]
                if mode == "knockout":
                    tensor = torch.zeros_like(tensor)
                else:
                    tensor = value.to(tensor.device, dtype=tensor.dtype).expand_as(
                        tensor
                    )

            if self._capture_enabled:
                self.activations[name] = tensor.detach().cpu()

            if rest is None:
                return tensor
            return (tensor, *rest)

        return hook

    def register_module(self, name: str, module: nn.Module) -> None:
        if name in self.registered_names:
            return
        handle = module.register_forward_hook(self._make_hook(name))
        self._handles.append(handle)
        self.registered_names.append(name)

    def register_umaze_defaults(self, *, dino_block_indices: list[int] | None = None) -> list[str]:
        """
        Register common probe sites for DINO-channel UMaze models.

        Returns:
            List of registered hook names.
        """
        encoder = self.model.encoder
        names: list[str] = []

        if hasattr(encoder, "base_model") and hasattr(encoder.base_model, "blocks"):
            blocks = encoder.base_model.blocks
            if dino_block_indices is None:
                dino_block_indices = [0, 5, 11]
            for idx in dino_block_indices:
                if 0 <= idx < len(blocks):
                    hook_name = f"dino.block.{idx}"
                    self.register_module(hook_name, blocks[idx])
                    names.append(hook_name)

        if hasattr(encoder, "projector") and encoder.projector is not None:
            self.register_module("projector", encoder.projector)
            names.append("projector")

        self.register_module("encoder", encoder)
        names.append("encoder")

        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.register_module("predictor", self.model.predictor)
            names.append("predictor")

        return names

    @torch.no_grad()
    def capture_encode_obs(
        self,
        obs: dict[str, torch.Tensor],
        *,
        readout: str = "post_projector",
    ) -> torch.Tensor:
        """
        Run ``encode_obs`` and return a flat feature vector per (batch, time) step.

        Readouts:
            ``post_projector``: mean-pooled visual tokens after encoder (default).
            ``agg_mlp``: MLP aggregation head (matches straightening losses).
            ``flatten``: flattened patch grid.
        """
        self.activations.clear()
        out = self.model.encode_obs(obs)
        visual = out["visual"]  # (B, T, P, D)

        if readout == "post_projector":
            return visual.mean(dim=2)
        if readout == "agg_mlp":
            b, t, p, d = visual.shape
            flat = visual.reshape(b * t, p, d)
            pooled = self.model.encoder.agg(flat)
            return pooled.reshape(b, t, -1)
        if readout == "flatten":
            b, t, p, d = visual.shape
            return visual.reshape(b, t, p * d)
        raise ValueError(f"Unknown readout: {readout}")


def flatten_activation(name: str, tensor: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    """
    Convert hooked activation to ``(N, F)`` feature matrix.

    ``N`` merges batch and time dimensions when present.
    """
    x = tensor
    if x.ndim == 4:
        # (B, T, P, D)
        if pool == "mean":
            x = x.mean(dim=2)
        elif pool == "flatten":
            b, t, p, d = x.shape
            x = x.reshape(b, t, p * d)
        else:
            raise ValueError(pool)
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim == 3:
        # (B, T, D) or (B, N, D)
        if pool == "mean" and x.shape[1] > 1 and x.shape[-1] <= 512:
            x = x.mean(dim=1)
        else:
            x = x.reshape(-1, x.shape[-1])
    elif x.ndim == 2:
        pass
    else:
        x = x.reshape(x.shape[0], -1)
    return x
