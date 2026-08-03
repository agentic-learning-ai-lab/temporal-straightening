"""Extract selected-layer UMaze features for region holdout.

Mirrors ``scripts/probe_umaze_layers.py`` collection for the selected layers:
``dino/6/pooled_patches`` and ``predictor/1/pooled_visual``.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Defaults from layer_probes/selected_layers.json on codex/umaze-physics-probes.
DEFAULT_DINO_LAYER = 6
DEFAULT_DINO_REP = "pooled_patches"
DEFAULT_PRED_LAYER = 1
DEFAULT_PRED_REP = "pooled_visual"


def resolve_checkpoint(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    ckpt_dir = path / "checkpoints"
    if ckpt_dir.is_dir():
        candidates = sorted(ckpt_dir.glob("model_*.pth"))
        if candidates:
            return candidates[-1]
    direct = path / "model.pth"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"No checkpoint under {path}")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    from models.dino import DinoV2Encoder

    _ = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    try:
        payload = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolve_checkpoint(path), map_location="cpu")
    modules = {}
    for name in ("encoder", "predictor", "proprio_encoder", "action_encoder"):
        if name not in payload:
            raise KeyError(f"checkpoint is missing {name!r}")
        modules[name] = payload[name].to(device).eval()
    return modules


def sample_windows(dataset, count: int, frameskip: int, nframes: int, seed: int):
    rng = random.Random(seed)
    choices = []
    for episode, length in enumerate(dataset.seq_lengths.tolist()):
        max_start = int(length) - 1 - frameskip * (nframes - 1)
        if max_start >= 0:
            choices.extend((episode, start) for start in range(max_start + 1))
    rng.shuffle(choices)
    return choices[: min(count, len(choices))]


def load_batch(dataset, choices, frameskip, nframes):
    visuals, proprios, actions, states = [], [], [], []
    for episode, start in choices:
        indices = [start + frameskip * offset for offset in range(nframes)]
        obs, act, state, _ = dataset.get_frames(episode, indices)
        visuals.append(obs["visual"])
        proprios.append(obs["proprio"])
        actions.append(act)
        states.append(state)
    return (
        torch.stack(visuals),
        torch.stack(proprios),
        torch.stack(actions),
        torch.stack(states),
    )


def _append(store, key, value):
    store.setdefault(key, []).append(value.detach().float().cpu())


def encode_stream(module, values, name):
    expected = int(module.patch_embed.in_channels)
    actual = int(values.shape[-1])
    if expected != actual:
        if actual > expected:
            raise ValueError(f"{name} has {actual} channels but checkpoint expects {expected}")
        print(
            f"WARNING: legacy {name} encoder expects {expected} channels but data has "
            f"{actual}; zero-padding for predictor activation probes",
            flush=True,
        )
        values = F.pad(values, (0, expected - actual))
    return module(values)


def dino_pooled_patches(encoder, x: torch.Tensor, layer: int) -> torch.Tensor:
    """Mean-pooled patch tokens at one DINO block via hub ``get_intermediate_layers``.

    Does not modify ``models/`` — same readout Daniel's ``forward_intermediates`` uses
    for ``pooled_patches``.
    """
    n_blocks = len(encoder.base_model.blocks)
    layer = int(layer)
    layer = layer + n_blocks if layer < 0 else layer
    if not 0 <= layer < n_blocks:
        raise ValueError(f"DINO layer {layer} is outside [0, {n_blocks - 1}]")
    patch, _cls = encoder.base_model.get_intermediate_layers(
        x,
        n=[layer],
        reshape=False,
        return_class_token=True,
        norm=True,
    )[0]
    return patch.mean(dim=1)


def predictor_layer_activations(predictor, x: torch.Tensor) -> list[torch.Tensor]:
    """Per-block normalized activations, matching Daniel's ``return_intermediates``.

    Walks the existing ``ViTPredictor`` modules in-place; does not change ``models/vit.py``.
    """
    b, n, _ = x.shape
    h = x + predictor.pos_embedding[:, :n]
    h = predictor.dropout(h)
    transformer = predictor.transformer
    intermediates = []
    for attn, ff in transformer.layers:
        h = attn(h) + h
        h = ff(h) + h
        intermediates.append(transformer.norm(h))
    return intermediates


@torch.inference_mode()
def collect_selected_activations(
    modules,
    dataset,
    choices,
    *,
    batch_size: int,
    frameskip: int,
    nframes: int,
    device: torch.device,
    dino_layer: int = DEFAULT_DINO_LAYER,
    predictor_layer: int = DEFAULT_PRED_LAYER,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return window tensors for Daniel's selected DINO + predictor sites."""
    encoder = modules["encoder"]
    predictor = modules["predictor"]
    representations: dict[str, list] = {}
    all_states, all_actions = [], []
    visual_dim = int(encoder.emb_dim)
    if hasattr(encoder, "agg_mlp"):
        token_count = int(encoder.agg_mlp[0].in_features // visual_dim)
        token_side = int(round(math.sqrt(token_count)))
        encoder_input_size = token_side * int(encoder.patch_size)
    else:
        encoder_input_size = 224

    for start in range(0, len(choices), batch_size):
        batch_choices = choices[start : start + batch_size]
        visual, proprio, action, state = load_batch(
            dataset, batch_choices, frameskip, nframes
        )
        b, t = visual.shape[:2]
        visual = visual.to(device)
        flat = visual.reshape(b * t, *visual.shape[2:])
        if flat.shape[-2:] != (encoder_input_size, encoder_input_size):
            flat = F.interpolate(
                flat,
                size=(encoder_input_size, encoder_input_size),
                mode="bilinear",
                align_corners=False,
            )

        pooled = dino_pooled_patches(encoder, flat, dino_layer)
        _append(
            representations,
            f"dino/{dino_layer}/pooled_patches",
            pooled.reshape(b, t, -1),
        )

        visual_tokens = encoder(flat).reshape(b, t, -1, visual_dim)
        prop_emb = encode_stream(modules["proprio_encoder"], proprio.to(device), "proprio")
        act_emb = encode_stream(modules["action_encoder"], action.to(device), "action")
        prop_tiled = prop_emb.unsqueeze(2).expand(-1, -1, visual_tokens.shape[2], -1)
        act_tiled = act_emb.unsqueeze(2).expand(-1, -1, visual_tokens.shape[2], -1)
        z = torch.cat([visual_tokens, prop_tiled, act_tiled], dim=-1)
        hist = min(int(predictor.pos_embedding.shape[1] // z.shape[2]), t - 1)
        pred_input = z[:, :hist].reshape(b, hist * z.shape[2], -1)
        pred_layers = predictor_layer_activations(predictor, pred_input)
        if not (0 <= predictor_layer < len(pred_layers)):
            raise IndexError(
                f"predictor layer {predictor_layer} out of range "
                f"[0, {len(pred_layers) - 1}]"
            )
        activation = pred_layers[predictor_layer].reshape(
            b, hist, z.shape[2], -1
        )[..., :visual_dim]
        _append(
            representations,
            f"predictor/{predictor_layer}/pooled_visual",
            activation.mean(dim=2),
        )

        all_states.append(state.float())
        all_actions.append(action.float())

    reps = {key: torch.cat(value).numpy() for key, value in representations.items()}
    return reps, torch.cat(all_states).numpy(), torch.cat(all_actions).numpy()


def flatten_window_features(rep: np.ndarray, *, transition: bool) -> np.ndarray:
    """Match Daniel's frame-level flatten (no patch rows)."""
    selected = rep
    if transition:
        selected = selected[:, 1:] - selected[:, :-1]
    if selected.ndim == 4:
        selected = selected.mean(axis=2)
    return selected.reshape(-1, selected.shape[-1])


def transition_labels_from_states(states: np.ndarray, usable_t: int) -> dict[str, np.ndarray]:
    """Displacement labels for consecutive window frames (Daniel layer-probe style)."""
    xy_delta = states[:, 1:usable_t, :2] - states[:, : usable_t - 1, :2]
    speed = np.linalg.norm(xy_delta, axis=-1)
    direction = xy_delta / np.maximum(speed[..., None], 1e-6)
    low = speed < 1e-4
    direction_cos = direction[..., 0].copy()
    direction_sin = direction[..., 1].copy()
    direction_cos[low] = np.nan
    direction_sin[low] = np.nan
    # Location for holdout: where the transition ends.
    pos = states[:, 1:usable_t, :2]
    return {
        "position_x": pos[..., 0].reshape(-1),
        "position_y": pos[..., 1].reshape(-1),
        "speed": speed.reshape(-1),
        "direction_cos": direction_cos.reshape(-1),
        "direction_sin": direction_sin.reshape(-1),
    }


def raw_frame_labels_from_states(states: np.ndarray, usable_t: int) -> dict[str, np.ndarray]:
    """Per-frame labels (qvel speed/heading) aligned to ``rep[:, :usable_t]``."""
    slice_states = states[:, :usable_t]
    pos = slice_states[..., :2]
    if slice_states.shape[-1] >= 4:
        vel = slice_states[..., 2:4]
    else:
        vel = np.zeros_like(pos)
    speed = np.linalg.norm(vel, axis=-1)
    direction_cos = vel[..., 0] / (speed + 1e-8)
    direction_sin = vel[..., 1] / (speed + 1e-8)
    low = speed < 1e-4
    direction_cos = direction_cos.copy()
    direction_sin = direction_sin.copy()
    direction_cos[low] = np.nan
    direction_sin[low] = np.nan
    return {
        "position_x": pos[..., 0].reshape(-1),
        "position_y": pos[..., 1].reshape(-1),
        "speed": speed.reshape(-1),
        "direction_cos": direction_cos.reshape(-1),
        "direction_sin": direction_sin.reshape(-1),
    }


def pack_probe_matrix(
    rep: np.ndarray,
    states: np.ndarray,
    *,
    transition: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Features + labels for one Daniel representation (window batch × time)."""
    features = flatten_window_features(rep, transition=transition)
    usable_t = rep.shape[1]
    if transition:
        labels = transition_labels_from_states(states, usable_t)
    else:
        labels = raw_frame_labels_from_states(states, usable_t)
    if features.shape[0] != labels["speed"].shape[0]:
        raise RuntimeError(
            f"feature/label length mismatch: {features.shape[0]} vs {labels['speed'].shape[0]}"
        )
    return features, labels
