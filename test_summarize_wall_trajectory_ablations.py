import json
from pathlib import Path

from summarize_wall_trajectory_ablations import CONDITIONS, render_report


def comparison_fixture() -> dict:
    conditions = {}
    for index, name in enumerate(CONDITIONS):
        values = {"100": 0.7 + 0.05 * index, "200": 0.8, "300": 0.9}
        mean = sum(values.values()) / len(values)
        conditions[name] = {
            "seeds": [100, 200, 300],
            "per_seed": {
                seed: {"final_eval/success_rate": value}
                for seed, value in values.items()
            },
            "aggregate": {
                "final_eval/success_rate": {
                    "mean": mean,
                    "population_std": 0.05,
                }
            },
        }
    return {
        "status": "complete",
        "conditions": conditions,
        "paired_deltas": {
            name: {
                "metrics": {"final_eval/success_rate": {"mean": 0.05 * index}}
            }
            for index, name in enumerate(("r1", "r2", "r3"), start=1)
        },
    }


def test_render_report_includes_conditions_deltas_and_losses(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    for name in ("r1", "r2", "r3"):
        condition_dir = checkpoint_root / CONDITIONS[name]["checkpoint_dir"]
        condition_dir.mkdir(parents=True)
        (condition_dir / "train.log").write_text(
            "Epoch 20 Training loss: 0.1234 Validation loss: 0.2345\n"
        )

    r0_train_log = tmp_path / "r0_train.log"
    r0_train_log.write_text(
        "Epoch 20 Training loss: 0.0531 Validation loss: 0.0508\n"
    )
    report = render_report(comparison_fixture(), checkpoint_root, r0_train_log)
    assert "R0 direction-only baseline" in report
    assert "R1 speed-only" in report
    assert "R2 full penalty" in report
    assert "R3 direction + speed" in report
    assert "+5.00 pp" in report
    assert "0.0531 | 0.0508" in report
    assert "0.1234 | 0.2345" in report
    assert "150 evaluations per condition" in report


def test_report_cli_writes_output(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(comparison_fixture()))
    output = tmp_path / "README.md"

    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_wall_trajectory_ablations.py")),
            "--comparison",
            str(comparison),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.read_text().startswith("# Wall trajectory-penalty ablation")
