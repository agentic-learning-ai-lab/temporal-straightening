from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = "temporal-straightening-medium"
REPO_DIR = Path("/workspace/temporal-straightening")
VOLUME_DIR = Path("/mnt/ts")

volume = modal.Volume.from_name("temporal-straightening-medium", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
        add_python=None,
    )
    .apt_install(
        "build-essential",
        "curl",
        "ffmpeg",
        "git",
        "libegl1",
        "libglew-dev",
        "libglib2.0-0",
        "libgl1",
        "libglx-mesa0",
        "libosmesa6-dev",
        "libsm6",
        "libxext6",
        "libxrender1",
        "patchelf",
        "unzip",
        "wget",
    )
    .pip_install("cython==0.29.37", "numpy==1.26.4", "setuptools<81", "wheel")
    .pip_install(
        "accelerate==0.26.1",
        "antlr4-python3-runtime==4.9.3",
        "cloudpickle==2.1.0",
        "d4rl==1.1",
        "decord==0.6.0",
        "einops==0.4.1",
        "glfw==2.7.0",
        "gym==0.23.1",
        "hydra-core==1.2.0",
        "hydra-submitit-launcher==1.2.0",
        "imageio==2.34.1",
        "imageio-ffmpeg==0.4.9",
        "labmaze==1.0.6",
        "matplotlib==3.5.3",
        "moviepy==1.0.3",
        "mujoco==3.2.7",
        "mujoco-py==2.1.2.14",
        "omegaconf==2.3.0",
        "opencv-python-headless==4.6.0.66",
        "pillow==10.3.0",
        "protobuf==3.20.3",
        "psutil==5.9.8",
        "requests==2.32.3",
        "scipy==1.13.1",
        "submitit==1.5.1",
        "tqdm==4.66.4",
        "wandb==0.13.1",
    )
    .run_commands(
        "mkdir -p /root/.mujoco && "
        "wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz && "
        "tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco && "
        "rm /tmp/mujoco210.tar.gz"
    )
    .env(
        {
            "D4RL_SUPPRESS_IMPORT_ERROR": "1",
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
            "LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
            "LIBGL_DRIVERS_PATH": "/usr/lib/x86_64-linux-gnu/dri",
            "MUJOCO_GL": "osmesa",
            "PYTHONUNBUFFERED": "1",
            "TORCH_HOME": str(VOLUME_DIR / "torch_home"),
            "WANDB_MODE": "disabled",
        }
    )
    .workdir(str(REPO_DIR))
    .add_local_dir(
        ".",
        str(REPO_DIR),
        ignore=[
            ".git",
            "__pycache__",
            ".pytest_cache",
            "checkpoints",
            "data",
            "outputs",
            "plan_outputs",
            "plan_outputs_gd",
            "wandb",
        ],
    )
)

app = modal.App(APP_NAME)

ENVIRONMENT_SPECS = {
    "medium": {
        "train_env": "point_maze_medium",
        "dataset_leaf": "point_maze_medium",
        "generator": "generate_point_maze_medium.py",
        "log_name": "generate_medium_dataset.log",
    },
    "wall": {
        "train_env": "wall",
        "dataset_leaf": "wall_single",
        "generator": "generate_wall_dataset.py",
        "log_name": "generate_wall_dataset.log",
    },
}


def _environment_spec(environment: str) -> dict[str, str]:
    key = environment.strip().lower()
    if key in {"point_maze_medium", "maze2d-medium", "maze2d_medium"}:
        key = "medium"
    if key not in ENVIRONMENT_SPECS:
        raise ValueError(
            f"Unknown environment '{environment}'. Expected one of: "
            f"{', '.join(sorted(ENVIRONMENT_SPECS))}"
        )
    return {"key": key, **ENVIRONMENT_SPECS[key]}


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(arg) for arg in args)
    print(f"\n$ {printable}", flush=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, args)


def _latest_hydra_run_dir(ckpt_base: Path) -> Path:
    candidates = sorted(
        ckpt_base.glob("test/*/hydra.yaml"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No hydra.yaml found under {ckpt_base}/test")
    return candidates[-1].parent


def _parse_epoch_losses(log_file: Path) -> dict[str, Any]:
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+Training loss:\s+(?P<train>[0-9.]+)\s+.*Validation loss:\s+(?P<val>[0-9.]+)"
    )
    last: dict[str, Any] = {}
    if not log_file.exists():
        return last
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            last = {
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("train")),
                "val_loss": float(match.group("val")),
            }
    return last


def _read_plan_logs(plan_dir: Path) -> dict[str, Any]:
    logs_path = plan_dir / "logs.json"
    if not logs_path.exists():
        raise FileNotFoundError(f"Missing planner log: {logs_path}")
    entries = [
        json.loads(line)
        for line in logs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"Planner log is empty: {logs_path}")
    return entries[-1]


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def _merge_variant_summary(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return summary

    existing = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(existing)
    for key, value in summary.items():
        if key != "results":
            merged[key] = value

    results_by_variant: dict[str, dict[str, Any]] = {}
    for result in existing.get("results", []):
        variant = result.get("variant")
        if variant:
            results_by_variant[variant] = result
    for result in summary.get("results", []):
        variant = result.get("variant")
        if variant:
            results_by_variant[variant] = result

    merged["results"] = list(results_by_variant.values())
    return merged


def _variant_specs(
    include_dino: bool,
    include_dino_cls: bool,
    include_speed_ablations: bool = False,
    include_adapter_ablations: bool = False,
) -> list[dict[str, str]]:
    variants = [
        {
            "name": "authors_dino_patch_straightened",
            "encoder": "dino",
            "straighten": "cos1e-1",
        }
    ]
    if include_dino:
        variants.append(
            {
                "name": "dino_patch_no_straightening",
                "encoder": "dino",
                "straighten": "False",
            }
        )
    if include_dino_cls:
        variants.append(
            {
                "name": "dino_cls_no_straightening",
                "encoder": "dino_cls",
                "straighten": "False",
            }
        )
    if include_speed_ablations:
        variants.extend(
            [
                {
                    "name": "dino_patch_speed_constancy",
                    "encoder": "dino",
                    "straighten": "aggspeed1e-1",
                },
                {
                    "name": "dino_patch_cos_plus_speed",
                    "encoder": "dino",
                    "straighten": "cos1e-1+aggspeed1e-1",
                },
                {
                    "name": "dino_cls_speed_constancy",
                    "encoder": "dino_cls",
                    "straighten": "speed1e-1",
                },
            ]
        )
    if include_adapter_ablations:
        variants.extend(
            [
                {
                    "name": "dino_channel_no_straightening",
                    "encoder": "dino_channel",
                    "straighten": "False",
                },
                {
                    "name": "dino_channel_cos_straightened",
                    "encoder": "dino_channel",
                    "straighten": "cos1e-1",
                },
                {
                    "name": "dino_channel_speed_constancy",
                    "encoder": "dino_channel",
                    "straighten": "speed1e-1",
                },
                {
                    "name": "dino_channel_cos_plus_speed",
                    "encoder": "dino_channel",
                    "straighten": "cos1e-1+speed1e-1",
                },
            ]
        )
    return variants


def _ensure_dataset(
    *,
    environment: str,
    data_parent: Path,
    n_episodes: int,
    episode_length: int,
    policy: str,
    env: dict[str, str],
    log_dir: Path,
) -> Path:
    spec = _environment_spec(environment)
    dataset_path = data_parent / spec["dataset_leaf"]
    expected = [
        dataset_path / "states.pth",
        dataset_path / "actions.pth",
        dataset_path / "obses" / f"episode_{n_episodes - 1:03d}.pth",
    ]
    if spec["key"] == "medium":
        expected.append(dataset_path / "seq_lengths.pth")
    if spec["key"] == "wall":
        expected.extend(
            [
                dataset_path / "door_locations.pth",
                dataset_path / "wall_locations.pth",
            ]
        )
    if all(path.exists() for path in expected):
        print(f"Reusing dataset at {dataset_path}", flush=True)
        return dataset_path

    args = [
        "python",
        spec["generator"],
        "--n_episodes",
        str(n_episodes),
        "--episode_length",
        str(episode_length),
        "--output_dir",
        str(dataset_path),
        "--checkpoint_every",
        "10",
    ]
    if spec["key"] == "medium":
        args.extend(["--policy", policy])
    _run_command(args, cwd=REPO_DIR, env=env, log_file=log_dir / spec["log_name"])
    volume.commit()
    return dataset_path


def _plan_variant(
    *,
    variant: dict[str, str],
    run_root: Path,
    epochs: int,
    plan_n_evals: int,
    plan_goal_h: int,
    env: dict[str, str],
    plan_root_name: str = "plans",
    plan_log_name: str = "plan_gd.log",
) -> dict[str, Any]:
    name = variant["name"]
    log_dir = run_root / "logs" / name
    ckpt_base = run_root / "checkpoints" / name
    plan_dir = run_root / plan_root_name / name
    plan_dir.mkdir(parents=True, exist_ok=True)

    model_dir = _latest_hydra_run_dir(ckpt_base)
    model_epoch: int | str = epochs
    if not (model_dir / "checkpoints" / f"model_{model_epoch}.pth").exists():
        model_epoch = "latest"

    plan_log = log_dir / plan_log_name
    _run_command(
        [
            "python",
            "plan.py",
            "--config-name",
            "plan_gd.yaml",
            f"ckpt_base_path={model_dir}",
            f"model_name={name}",
            f"model_epoch={model_epoch}",
            f"n_evals={plan_n_evals}",
            f"goal_H={plan_goal_h}",
            "goal_source=dset",
            "objective.alpha=0",
            "objective.mode=last",
            "planner.max_iter=1",
            f"planner.n_taken_actions={plan_goal_h}",
            f"planner.sub_planner.horizon={plan_goal_h}",
            "planner.sub_planner.lr=1",
            "planner.sub_planner.sample_type=randn",
            "planner.sub_planner.action_noise=0.003",
            "planner.sub_planner.opt_steps=1000",
            "planner.sub_planner.eval_every=-1",
            "decode_for_viz=false",
            "+wandb_logging=false",
            f"hydra.run.dir={plan_dir}",
            f"hydra.sweep.dir={plan_dir}",
            "hydra.job.chdir=True",
        ],
        cwd=REPO_DIR,
        env=env,
        log_file=plan_log,
    )
    volume.commit()

    return {
        "model_dir": str(model_dir),
        "model_epoch": model_epoch,
        "plan_dir": str(plan_dir),
        "plan": _read_plan_logs(plan_dir),
    }


def _train_and_plan_variant(
    *,
    variant: dict[str, str],
    run_root: Path,
    train_env: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    plan_n_evals: int,
    plan_goal_h: int,
    env: dict[str, str],
) -> dict[str, Any]:
    name = variant["name"]
    log_dir = run_root / "logs" / name
    ckpt_base = run_root / "checkpoints" / name
    plan_dir = run_root / "plans" / name
    ckpt_base.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)

    train_log = log_dir / "train.log"
    _run_command(
        [
            "python",
            "train.py",
            "--config-name",
            "train.yaml",
            f"env={train_env}",
            f"ckpt_base_path={ckpt_base}",
            f"encoder={variant['encoder']}",
            f"training.straighten={variant['straighten']}",
            f"training.epochs={epochs}",
            f"training.batch_size={batch_size}",
            f"env.num_workers={num_workers}",
            "training.reconstruct_every_x_batch=999999",
            "plan_settings.plan_cfg_path=null",
            "hydra.job.chdir=True",
        ],
        cwd=REPO_DIR,
        env=env,
        log_file=train_log,
    )
    volume.commit()

    plan_result = _plan_variant(
        variant=variant,
        run_root=run_root,
        epochs=epochs,
        plan_n_evals=plan_n_evals,
        plan_goal_h=plan_goal_h,
        env=env,
    )

    return {
        "variant": name,
        "encoder": variant["encoder"],
        "straighten": variant["straighten"],
        "train": _parse_epoch_losses(train_log),
        **plan_result,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_DIR): volume},
)
def smoke() -> dict[str, Any]:
    import torch

    env = os.environ.copy()
    env["DATASET_DIR"] = str(VOLUME_DIR / "smoke_data")
    log_dir = VOLUME_DIR / "smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "python",
            "-c",
            "import torch, gym, d4rl, mujoco_py; import env; "
            "print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); "
            "e=gym.make('point_maze_medium'); print('env', e.action_space.shape); e.close()",
        ],
        cwd=REPO_DIR,
        env=env,
        log_file=log_dir / "smoke.log",
    )
    return {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_DIR): volume},
)
def run_medium(
    run_id: str,
    environment: str,
    epochs: int,
    n_episodes: int,
    episode_length: int,
    policy: str,
    batch_size: int,
    num_workers: int,
    plan_n_evals: int,
    plan_goal_h: int,
    include_dino: bool,
    include_dino_cls: bool,
    include_speed_ablations: bool,
    include_adapter_ablations: bool,
    variant_name: str,
) -> dict[str, Any]:
    spec = _environment_spec(environment)
    run_root = VOLUME_DIR / "runs" / run_id
    data_parent = (
        VOLUME_DIR
        / "datasets"
        / f"{spec['key']}_{policy}_{n_episodes}x{episode_length}"
    )
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DATASET_DIR"] = str(data_parent)
    env["WANDB_MODE"] = "disabled"
    env["TORCH_HOME"] = str(VOLUME_DIR / "torch_home")

    dataset_path = _ensure_dataset(
        environment=spec["key"],
        data_parent=data_parent,
        n_episodes=n_episodes,
        episode_length=episode_length,
        policy=policy,
        env=env,
        log_dir=log_dir,
    )

    variants = _variant_specs(
        include_dino=include_dino,
        include_dino_cls=include_dino_cls,
        include_speed_ablations=include_speed_ablations,
        include_adapter_ablations=include_adapter_ablations,
    )
    if variant_name:
        variants = [variant for variant in variants if variant["name"] == variant_name]
        if not variants:
            raise ValueError(f"Unknown variant_name: {variant_name}")

    results = []
    for variant in variants:
        started = time.time()
        try:
            result = _train_and_plan_variant(
                variant=variant,
                run_root=run_root,
                train_env=spec["train_env"],
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                plan_n_evals=plan_n_evals,
                plan_goal_h=plan_goal_h,
                env=env,
            )
        except Exception as exc:
            result = {
                "variant": variant["name"],
                "encoder": variant["encoder"],
                "straighten": variant["straighten"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            (log_dir / f"{variant['name']}_error.txt").write_text(
                result["traceback"],
                encoding="utf-8",
            )
            volume.commit()
        result["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
        results.append(result)

    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "environment": spec["key"],
        "train_env": spec["train_env"],
        "dataset": str(dataset_path),
        "epochs": epochs,
        "n_episodes": n_episodes,
        "episode_length": episode_length,
        "policy": policy,
        "plan_n_evals": plan_n_evals,
        "plan_goal_h": plan_goal_h,
        "results": results,
    }
    summary_path = run_root / "summary.json"
    summary = _merge_variant_summary(summary_path, summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_DIR): volume},
)
def plan_existing_medium(
    run_id: str,
    epochs: int,
    plan_n_evals: int,
    plan_goal_h: int,
    include_dino: bool,
    include_dino_cls: bool,
    include_speed_ablations: bool,
    include_adapter_ablations: bool,
    variant_name: str,
) -> dict[str, Any]:
    run_root = VOLUME_DIR / "runs" / run_id
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["TORCH_HOME"] = str(VOLUME_DIR / "torch_home")

    existing_summary_path = run_root / "summary.json"
    if existing_summary_path.exists():
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        env["DATASET_DIR"] = str(Path(existing_summary["dataset"]).parent)
    else:
        env["DATASET_DIR"] = str(VOLUME_DIR / "datasets")

    variants = _variant_specs(
        include_dino,
        include_dino_cls,
        include_speed_ablations,
        include_adapter_ablations,
    )
    if variant_name:
        variants = [variant for variant in variants if variant["name"] == variant_name]
        if not variants:
            raise ValueError(f"Unknown variant_name: {variant_name}")

    results = []
    for variant in variants:
        started = time.time()
        result: dict[str, Any] = {
            "variant": variant["name"],
            "encoder": variant["encoder"],
            "straighten": variant["straighten"],
        }
        try:
            result.update(
                _plan_variant(
                    variant=variant,
                    run_root=run_root,
                    epochs=epochs,
                    plan_n_evals=plan_n_evals,
                    plan_goal_h=plan_goal_h,
                    env=env,
                    plan_root_name="plans_fixed",
                    plan_log_name="plan_gd_fixed.log",
                )
            )
            result["train"] = _parse_epoch_losses(
                run_root / "logs" / variant["name"] / "train.log"
            )
        except Exception as exc:
            result.update(
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            error_path = run_root / "logs" / f"{variant['name']}_plan_fixed_error.txt"
            error_path.write_text(result["traceback"], encoding="utf-8")
            volume.commit()
        result["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
        results.append(result)

    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "epochs": epochs,
        "plan_n_evals": plan_n_evals,
        "plan_goal_h": plan_goal_h,
        "results": results,
    }
    summary_path = run_root / "summary_plans_fixed.json"
    summary = _merge_variant_summary(summary_path, summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    return summary


def _latent_metrics_from_z(z: Any, eps: float = 1e-6) -> dict[str, list[float]]:
    import torch
    import torch.nn.functional as F

    z = z.float()
    velocity = z[:, 1:] - z[:, :-1]
    speed = velocity.norm(dim=-1)
    result: dict[str, list[float]] = {
        "cosine_curvature": [],
        "cosine_alignment": [],
        "latent_speed": [],
        "latent_speed_cv": [],
        "relative_speed_jump": [],
        "path_endpoint_ratio": [],
    }

    speed_values = speed[speed > eps].detach().cpu().flatten().tolist()
    result["latent_speed"].extend(float(value) for value in speed_values)

    if speed.shape[1] > 0:
        speed_mean_t = speed.mean(dim=1)
        speed_std_t = speed.std(dim=1, unbiased=False)
        speed_cv = speed_std_t / (speed_mean_t + eps)
        result["latent_speed_cv"].extend(
            float(value) for value in speed_cv.detach().cpu().flatten().tolist()
        )

    if speed.shape[1] > 1:
        speed_mean_t = speed.mean(dim=1, keepdim=True)
        rel_jump = (speed[:, 1:] - speed[:, :-1]).abs() / (speed_mean_t + eps)
        result["relative_speed_jump"].extend(
            float(value) for value in rel_jump.detach().cpu().flatten().tolist()
        )

    if velocity.shape[1] > 1:
        v1 = velocity[:, :-1]
        v2 = velocity[:, 1:]
        step1 = v1.norm(dim=-1)
        step2 = v2.norm(dim=-1)
        mask = (step1 > eps) & (step2 > eps)
        if mask.any():
            cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)[mask]
            curvature = 1.0 - cos
            result["cosine_alignment"].extend(
                float(value) for value in cos.detach().cpu().flatten().tolist()
            )
            result["cosine_curvature"].extend(
                float(value) for value in curvature.detach().cpu().flatten().tolist()
            )

    endpoint = (z[:, -1] - z[:, 0]).norm(dim=-1)
    path_len = speed.sum(dim=1)
    endpoint_mask = endpoint > eps
    if endpoint_mask.any():
        ratio = path_len[endpoint_mask] / endpoint[endpoint_mask]
        result["path_endpoint_ratio"].extend(
            float(value) for value in ratio.detach().cpu().flatten().tolist()
        )
    return result


def _summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "std": (
            sum((value - _mean(values)) ** 2 for value in values) / len(values)
        )
        ** 0.5
        if values
        else float("nan"),
        "p05": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _analyze_variant_latents(
    *,
    variant: dict[str, str],
    run_root: Path,
    epochs: int,
    env: dict[str, str],
    max_rollouts: int,
) -> dict[str, Any]:
    import hydra
    import torch
    from omegaconf import OmegaConf

    from plan import load_model

    name = variant["name"]
    model_dir = _latest_hydra_run_dir(run_root / "checkpoints" / name)
    model_epoch: int | str = epochs
    if not (model_dir / "checkpoints" / f"model_{model_epoch}.pth").exists():
        model_epoch = "latest"
    if model_epoch == "latest":
        checkpoint_candidates = sorted(
            (model_dir / "checkpoints").glob("model_*.pth"),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoint_candidates:
            raise FileNotFoundError(f"No model checkpoints found in {model_dir}")
        model_ckpt = checkpoint_candidates[-1]
        match = re.search(r"model_(\d+)\.pth$", model_ckpt.name)
        if match:
            model_epoch = int(match.group(1))
    else:
        model_ckpt = model_dir / "checkpoints" / f"model_{model_epoch}.pth"

    with (model_dir / "hydra.yaml").open("r", encoding="utf-8") as handle:
        model_cfg = OmegaConf.load(handle)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    model.eval()

    _, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    base_dset = traj_dset["train"].dataset
    frameskip = int(model_cfg.frameskip)
    n_rollouts = min(max_rollouts, len(base_dset))

    collected: dict[str, list[float]] = {
        "cosine_curvature": [],
        "cosine_alignment": [],
        "latent_speed": [],
        "latent_speed_cv": [],
        "relative_speed_jump": [],
        "path_endpoint_ratio": [],
        "state_position_speed_cv": [],
    }
    frame_counts: list[int] = []

    torch.set_grad_enabled(False)
    try:
        for rollout_idx in range(n_rollouts):
            seq_len = int(base_dset.get_seq_length(rollout_idx))
            frame_idx = list(range(0, seq_len, frameskip))
            if len(frame_idx) < 3:
                continue

            visual = base_dset.load_visual_frames(rollout_idx, frame_idx)
            proprio = base_dset.proprios[rollout_idx, frame_idx]
            obs = {
                "visual": visual.unsqueeze(0).to(device),
                "proprio": proprio.unsqueeze(0).to(device),
            }
            z_visual = model.encode_obs(obs)["visual"]
            metrics = _latent_metrics_from_z(z_visual)
            for key, values in metrics.items():
                collected[key].extend(values)

            state = base_dset.states[rollout_idx, frame_idx]
            if state.shape[0] > 1 and state.shape[-1] >= 2:
                state_speed = (state[1:, :2] - state[:-1, :2]).norm(dim=-1)
                mean_speed = state_speed.mean()
                if float(mean_speed) > 1e-6:
                    cv = state_speed.std(unbiased=False) / (mean_speed + 1e-6)
                    collected["state_position_speed_cv"].append(float(cv))
            frame_counts.append(len(frame_idx))
    finally:
        torch.set_grad_enabled(True)

    latent_speed = collected["latent_speed"]
    speed_p05 = _percentile(latent_speed, 0.05)
    speed_p95 = _percentile(latent_speed, 0.95)
    metrics_summary = {
        "cosine_curvature_mean": _mean(collected["cosine_curvature"]),
        "cosine_alignment_mean": _mean(collected["cosine_alignment"]),
        "path_endpoint_ratio_mean": _mean(collected["path_endpoint_ratio"]),
        "latent_speed_mean": _mean(latent_speed),
        "latent_speed_std": _summarize_values(latent_speed)["std"],
        "latent_speed_cv_time_mean": _mean(collected["latent_speed_cv"]),
        "latent_speed_p95_p05_ratio": speed_p95 / (speed_p05 + 1e-6)
        if latent_speed
        else float("nan"),
        "relative_speed_jump_mean": _mean(collected["relative_speed_jump"]),
        "state_position_speed_cv_mean": _mean(collected["state_position_speed_cv"]),
    }

    return {
        "variant": name,
        "encoder": variant["encoder"],
        "straighten": variant["straighten"],
        "model_dir": str(model_dir),
        "model_epoch": model_epoch,
        "rollouts_analyzed": len(frame_counts),
        "frames_per_rollout_mean": _mean([float(count) for count in frame_counts]),
        "metrics": metrics_summary,
        "distributions": {
            "cosine_curvature": _summarize_values(collected["cosine_curvature"]),
            "latent_speed": _summarize_values(latent_speed),
            "latent_speed_cv": _summarize_values(collected["latent_speed_cv"]),
            "relative_speed_jump": _summarize_values(
                collected["relative_speed_jump"]
            ),
            "path_endpoint_ratio": _summarize_values(
                collected["path_endpoint_ratio"]
            ),
            "state_position_speed_cv": _summarize_values(
                collected["state_position_speed_cv"]
            ),
        },
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_DIR): volume},
)
def analyze_latents_medium(
    run_id: str,
    epochs: int,
    environment: str,
    include_dino: bool,
    include_dino_cls: bool,
    include_speed_ablations: bool,
    include_adapter_ablations: bool,
    variant_name: str,
    max_rollouts: int,
) -> dict[str, Any]:
    run_root = VOLUME_DIR / "runs" / run_id
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["TORCH_HOME"] = str(VOLUME_DIR / "torch_home")

    existing_summary_path = run_root / "summary.json"
    if existing_summary_path.exists():
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        env["DATASET_DIR"] = str(Path(existing_summary["dataset"]).parent)
    else:
        env["DATASET_DIR"] = str(VOLUME_DIR / "datasets")

    variants = _variant_specs(
        include_dino,
        include_dino_cls,
        include_speed_ablations,
        include_adapter_ablations,
    )
    if variant_name:
        variants = [variant for variant in variants if variant["name"] == variant_name]
        if not variants:
            raise ValueError(f"Unknown variant_name: {variant_name}")

    results = []
    for variant in variants:
        started = time.time()
        try:
            result = _analyze_variant_latents(
                variant=variant,
                run_root=run_root,
                epochs=epochs,
                env=env,
                max_rollouts=max_rollouts,
            )
        except Exception as exc:
            result = {
                "variant": variant["name"],
                "encoder": variant["encoder"],
                "straighten": variant["straighten"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            error_path = run_root / "logs" / f"{variant['name']}_latent_error.txt"
            error_path.write_text(result["traceback"], encoding="utf-8")
            volume.commit()
        result["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
        results.append(result)

    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "environment": _environment_spec(environment)["train_env"],
        "analysis": "latent straightness and speed constancy diagnostics",
        "max_rollouts": max_rollouts,
        "metric_definitions": {
            "cosine_curvature_mean": "mean(1 - cosine(v_t, v_t+1)); lower is directionally straighter",
            "path_endpoint_ratio_mean": "latent path length divided by endpoint distance; 1 is globally straight",
            "latent_speed_cv_time_mean": "mean over token trajectories of std(speed_t)/mean(speed_t); lower is more constant speed",
            "latent_speed_p95_p05_ratio": "global p95 latent step speed divided by p05 latent step speed; lower means fewer speed lurches",
            "relative_speed_jump_mean": "mean abs(speed_t+1 - speed_t) normalized by trajectory mean speed",
            "state_position_speed_cv_mean": "same CV diagnostic on physical x/y state, for context only",
        },
        "results": results,
    }
    summary_path = run_root / "latent_analysis.json"
    summary = _merge_variant_summary(summary_path, summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    return summary


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    ephemeral_disk=524288,
    timeout=60 * 60 * 6,
    volumes={str(VOLUME_DIR): volume},
)
def package_medium_artifacts(
    run_id: str,
    include_epoch_checkpoints: bool,
) -> dict[str, Any]:
    run_root = VOLUME_DIR / "runs" / run_id
    if not run_root.exists():
        raise FileNotFoundError(f"Missing run root: {run_root}")

    export_dir = VOLUME_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    suffix = "full" if include_epoch_checkpoints else "media-evidence"
    tar_path = export_dir / f"{run_id}-{suffix}.tar"
    manifest_path = export_dir / f"{run_id}-{suffix}-manifest.json"

    files_included = 0
    skipped: list[str] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_root)
        is_epoch_checkpoint = (
            "checkpoints" in rel.parts
            and path.suffix == ".pth"
            and path.name.startswith("model_")
        )
        if is_epoch_checkpoint and not include_epoch_checkpoints:
            skipped.append(str(rel))
            continue
        files_included += 1

    if tar_path.exists():
        tar_path.unlink()
    tar_cmd = ["tar"]
    if not include_epoch_checkpoints:
        tar_cmd.extend(
            [
                "--exclude",
                f"runs/{run_id}/checkpoints/*/test/*/checkpoints/model_*.pth",
            ]
        )
    tar_cmd.extend(["-cf", str(tar_path), "-C", str(VOLUME_DIR), f"runs/{run_id}"])
    subprocess.run(tar_cmd, check=True)

    manifest = {
        "run_id": run_id,
        "run_root": str(run_root),
        "tar_path": str(tar_path),
        "include_epoch_checkpoints": include_epoch_checkpoints,
        "files_included": files_included,
        "files_skipped": len(skipped),
        "skipped_epoch_checkpoints": skipped,
        "tar_size_bytes": tar_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    action: str = "run",
    run_id: str = "",
    environment: str = "medium",
    epochs: int = 20,
    n_episodes: int = 50,
    episode_length: int = 100,
    policy: str = "random",
    batch_size: int = 32,
    num_workers: int = 8,
    plan_n_evals: int = 50,
    plan_goal_h: int = 25,
    include_dino: bool = True,
    include_dino_cls: bool = False,
    include_speed_ablations: bool = False,
    include_adapter_ablations: bool = False,
    variant_name: str = "",
    max_rollouts: int = 50,
    include_epoch_checkpoints: bool = False,
) -> None:
    if action == "smoke":
        print(json.dumps(smoke.remote(), indent=2))
        return
    if action == "plan-existing":
        if not run_id:
            raise ValueError("run_id is required for plan-existing")
        summary = plan_existing_medium.remote(
            run_id=run_id,
            epochs=epochs,
            plan_n_evals=plan_n_evals,
            plan_goal_h=plan_goal_h,
            include_dino=include_dino,
            include_dino_cls=include_dino_cls,
            include_speed_ablations=include_speed_ablations,
            include_adapter_ablations=include_adapter_ablations,
            variant_name=variant_name,
        )
        print(json.dumps(summary, indent=2))
        return
    if action == "analyze-latents":
        if not run_id:
            raise ValueError("run_id is required for analyze-latents")
        summary = analyze_latents_medium.remote(
            run_id=run_id,
            epochs=epochs,
            environment=environment,
            include_dino=include_dino,
            include_dino_cls=include_dino_cls,
            include_speed_ablations=include_speed_ablations,
            include_adapter_ablations=include_adapter_ablations,
            variant_name=variant_name,
            max_rollouts=max_rollouts,
        )
        print(json.dumps(summary, indent=2))
        return
    if action == "package-artifacts":
        if not run_id:
            raise ValueError("run_id is required for package-artifacts")
        manifest = package_medium_artifacts.remote(
            run_id=run_id,
            include_epoch_checkpoints=include_epoch_checkpoints,
        )
        print(json.dumps(manifest, indent=2))
        return
    if action != "run":
        raise ValueError(
            "action must be 'smoke', 'run', 'plan-existing', 'analyze-latents', or 'package-artifacts'"
        )

    if not run_id:
        run_id = f"{_environment_spec(environment)['key']}-" + datetime.utcnow().strftime(
            "%Y%m%d-%H%M%S"
        )
    summary = run_medium.remote(
        run_id=run_id,
        environment=environment,
        epochs=epochs,
        n_episodes=n_episodes,
        episode_length=episode_length,
        policy=policy,
        batch_size=batch_size,
        num_workers=num_workers,
        plan_n_evals=plan_n_evals,
        plan_goal_h=plan_goal_h,
        include_dino=include_dino,
        include_dino_cls=include_dino_cls,
        include_speed_ablations=include_speed_ablations,
        include_adapter_ablations=include_adapter_ablations,
        variant_name=variant_name,
    )
    print(json.dumps(summary, indent=2))
