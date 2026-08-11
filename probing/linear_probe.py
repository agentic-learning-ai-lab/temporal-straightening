"""Train and evaluate ridge linear probes on cached activations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


DEFAULT_TARGETS = [
    "position_x",
    "position_y",
    "speed",
    "direction_cos",
    "direction_sin",
]


def angular_error_deg(cos_pred: np.ndarray, sin_pred: np.ndarray, cos_true: np.ndarray, sin_true: np.ndarray) -> float:
    pred = np.arctan2(sin_pred, cos_pred)
    true = np.arctan2(sin_true, cos_true)
    diff = np.abs(np.arctan2(np.sin(pred - true), np.cos(pred - true)))
    return float(np.degrees(diff.mean()))


@dataclass
class ProbeResult:
    target: str
    train_r2: float
    val_r2: float
    train_rmse: float
    val_rmse: float
    n_train: int
    n_val: int
    extra: dict = field(default_factory=dict)


class LinearProbeSuite:
    """Independent ridge probes per target with feature standardization."""

    def __init__(self, *, alpha: float = 1.0, targets: list[str] | None = None):
        self.alpha = alpha
        self.targets = targets or list(DEFAULT_TARGETS)
        self.scalers: dict[str, StandardScaler] = {}
        self.models: dict[str, Ridge] = {}
        self.results: list[ProbeResult] = []

    def fit(self, x_train: np.ndarray, y_train: dict[str, np.ndarray], x_val: np.ndarray, y_val: dict[str, np.ndarray]) -> list[ProbeResult]:
        self.models.clear()
        self.scalers.clear()
        self.results.clear()

        x_scaler = StandardScaler()
        x_train_s = x_scaler.fit_transform(x_train)
        x_val_s = x_scaler.transform(x_val)
        self.scalers["features"] = x_scaler

        for target in self.targets:
            y_tr = y_train[target]
            y_va = y_val[target]
            valid_tr = np.isfinite(y_tr)
            valid_va = np.isfinite(y_va)
            if valid_tr.sum() < 10 or valid_va.sum() < 5:
                self.results.append(
                    ProbeResult(
                        target=target,
                        train_r2=float("nan"),
                        val_r2=float("nan"),
                        train_rmse=float("nan"),
                        val_rmse=float("nan"),
                        n_train=int(valid_tr.sum()),
                        n_val=int(valid_va.sum()),
                        extra={"skipped": "insufficient finite labels"},
                    )
                )
                continue

            model = Ridge(alpha=self.alpha)
            model.fit(x_train_s[valid_tr], y_tr[valid_tr])
            self.models[target] = model

            pred_tr = model.predict(x_train_s[valid_tr])
            pred_va = model.predict(x_val_s[valid_va])
            result = ProbeResult(
                target=target,
                train_r2=float(r2_score(y_tr[valid_tr], pred_tr)),
                val_r2=float(r2_score(y_va[valid_va], pred_va)),
                train_rmse=float(np.sqrt(mean_squared_error(y_tr[valid_tr], pred_tr))),
                val_rmse=float(np.sqrt(mean_squared_error(y_va[valid_va], pred_va))),
                n_train=int(valid_tr.sum()),
                n_val=int(valid_va.sum()),
            )
            self.results.append(result)

        if (
            "direction_cos" in self.models
            and "direction_sin" in self.models
            and np.isfinite(y_val["direction_cos"]).sum() >= 5
        ):
            valid = np.isfinite(y_val["direction_cos"]) & np.isfinite(y_val["direction_sin"])
            cos_pred = self.models["direction_cos"].predict(x_val_s[valid])
            sin_pred = self.models["direction_sin"].predict(x_val_s[valid])
            angle_err = angular_error_deg(
                cos_pred,
                sin_pred,
                y_val["direction_cos"][valid],
                y_val["direction_sin"][valid],
            )
            for result in self.results:
                if result.target == "direction_cos":
                    result.extra["direction_mae_deg"] = angle_err

        return self.results

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x_s = self.scalers["features"].transform(x)
        return {t: m.predict(x_s) for t, m in self.models.items()}

    def save_report(self, path: Path) -> None:
        payload = {
            "targets": self.targets,
            "alpha": self.alpha,
            "results": [
                {
                    "target": r.target,
                    "train_r2": r.train_r2,
                    "val_r2": r.val_r2,
                    "train_rmse": r.train_rmse,
                    "val_rmse": r.val_rmse,
                    "n_train": r.n_train,
                    "n_val": r.n_val,
                    **r.extra,
                }
                for r in self.results
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))


def episode_train_val_split(episode_ids: np.ndarray, val_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split frame indices by episode (no leakage across trajectories)."""
    rng = np.random.default_rng(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)
    n_val = max(1, int(len(unique_eps) * val_fraction))
    val_eps = set(unique_eps[:n_val])
    train_mask = np.array([ep not in val_eps for ep in episode_ids])
    val_mask = ~train_mask
    return train_mask, val_mask


def location_threshold(positions: np.ndarray, *, mode: str = "median") -> float:
    """Spatial cut for location holdout (median or midpoint)."""
    finite = positions[np.isfinite(positions)]
    if finite.size == 0:
        raise ValueError("No finite positions available for location holdout.")
    if mode == "median":
        return float(np.median(finite))
    if mode == "midpoint":
        return float(0.5 * (finite.min() + finite.max()))
    raise ValueError(f"Unknown threshold mode: {mode}")


def location_holdout_masks(
    pos_x: np.ndarray, pos_y: np.ndarray, *, axis: str = "x", threshold: float | None = None,
    threshold_mode: str = "median", train_side: str = "low",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split frames by maze half along one axis."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if train_side not in {"low", "high"}:
        raise ValueError("train_side must be 'low' or 'high'")
    coord = pos_x if axis == "x" else pos_y
    if threshold is None:
        threshold = location_threshold(coord, mode=threshold_mode)
    low = np.isfinite(coord) & (coord < threshold)
    high = np.isfinite(coord) & (coord >= threshold)
    if train_side == "low":
        return low, high, float(threshold)
    return high, low, float(threshold)


def run_location_holdout(
    features: np.ndarray, labels: dict[str, np.ndarray], *, axes: tuple[str, ...] = ("x", "y"),
    threshold_mode: str = "median", alpha: float = 1.0, targets: list[str] | None = None,
) -> dict:
    """Train probes in one maze half and evaluate in the complementary half."""
    report: dict = {
        "threshold_mode": threshold_mode,
        "splits": [],
        "summary": {},
    }
    target_names = targets or list(DEFAULT_TARGETS)
    pos_features = np.stack([labels["position_x"], labels["position_y"]], axis=1)

    for axis in axes:
        for train_side in ("low", "high"):
            train_mask, test_mask, thr = location_holdout_masks(
                labels["position_x"],
                labels["position_y"],
                axis=axis,
                threshold_mode=threshold_mode,
                train_side=train_side,
            )
            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())
            entry = {
                "axis": axis,
                "train_side": train_side,
                "threshold": thr,
                "n_train": n_train,
                "n_test": n_test,
                "results": [],
                "position_baseline": [],
            }
            if n_train < 20 or n_test < 10:
                entry["skipped"] = "insufficient frames in train or test region"
                report["splits"].append(entry)
                continue

            suite = LinearProbeSuite(alpha=alpha, targets=target_names)
            suite.fit(
                features[train_mask],
                {k: v[train_mask] for k, v in labels.items()},
                features[test_mask],
                {k: v[test_mask] for k, v in labels.items()},
            )
            entry["results"] = [
                {
                    "target": r.target,
                    "train_r2": r.train_r2,
                    "heldout_r2": r.val_r2,
                    "train_rmse": r.train_rmse,
                    "heldout_rmse": r.val_rmse,
                    "n_train": r.n_train,
                    "n_heldout": r.n_val,
                    **r.extra,
                }
                for r in suite.results
            ]

            baseline_targets = [t for t in target_names if t not in {"position_x", "position_y"}]
            if baseline_targets:
                baseline = LinearProbeSuite(alpha=alpha, targets=baseline_targets)
                baseline.fit(
                    pos_features[train_mask],
                    {k: v[train_mask] for k, v in labels.items()},
                    pos_features[test_mask],
                    {k: v[test_mask] for k, v in labels.items()},
                )
                entry["position_baseline"] = [
                    {
                        "target": r.target,
                        "heldout_r2": r.val_r2,
                        "heldout_rmse": r.val_rmse,
                        "n_heldout": r.n_val,
                    }
                    for r in baseline.results
                ]
            report["splits"].append(entry)

    # Mean held-out R² per target across completed splits.
    by_target: dict[str, list[float]] = {t: [] for t in target_names}
    baseline_by_target: dict[str, list[float]] = {}
    for split in report["splits"]:
        for row in split.get("results", []):
            if np.isfinite(row["heldout_r2"]):
                by_target[row["target"]].append(float(row["heldout_r2"]))
        for row in split.get("position_baseline", []):
            baseline_by_target.setdefault(row["target"], [])
            if np.isfinite(row["heldout_r2"]):
                baseline_by_target[row["target"]].append(float(row["heldout_r2"]))

    report["summary"] = {
        target: {
            "mean_heldout_r2": float(np.mean(vals)) if vals else float("nan"),
            "min_heldout_r2": float(np.min(vals)) if vals else float("nan"),
            "n_splits": len(vals),
            "position_baseline_mean_heldout_r2": (
                float(np.mean(baseline_by_target[target]))
                if baseline_by_target.get(target) else None
            ),
        }
        for target, vals in by_target.items()
    }
    return report


def tensors_to_numpy(
    features: torch.Tensor, labels: dict[str, torch.Tensor], episode_ids: torch.Tensor
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    x = features.reshape(features.shape[0], -1).cpu().numpy()
    y = {k: v.reshape(-1).cpu().numpy() for k, v in labels.items()}
    ep = episode_ids.reshape(-1).cpu().numpy()
    return x, y, ep
