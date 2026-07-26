import json
import math
import subprocess
import sys
from pathlib import Path


def write_seed(root: Path, seed: int, success: float, state: float) -> None:
    path = root / f"seed_{seed}" / "aggregate.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "seed": seed,
                "n_evals": 50,
                "metrics": {
                    "final_eval": {
                        "final_eval/success_rate": success,
                        "final_eval/mean_state_dist": state,
                    }
                },
            }
        )
        + "\n"
    )


def test_aggregate_condition_seeds(tmp_path: Path) -> None:
    off, on = tmp_path / "off", tmp_path / "on"
    for seed, off_success, on_success in ((100, 0.4, 0.5), (200, 0.6, 0.9)):
        write_seed(off, seed, off_success, 4.0)
        write_seed(on, seed, on_success, 3.0)

    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("aggregate_condition_seeds.py")),
            "--condition",
            f"off={off}",
            "--condition",
            f"on={on}",
            "--baseline",
            "off",
            "--treatment",
            "on",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    success = result["conditions"]["on"]["aggregate"]["final_eval/success_rate"]
    delta = result["paired_delta"]["metrics"]["final_eval/success_rate"]
    assert success == {"mean": 0.7, "population_std": 0.2}
    assert math.isclose(delta["mean"], 0.2)
    assert math.isclose(delta["population_std"], 0.1)


def test_aggregate_multiple_treatments_against_baseline(tmp_path: Path) -> None:
    r0, r1, r2 = tmp_path / "r0", tmp_path / "r1", tmp_path / "r2"
    for seed, baseline, speed, full in (
        (100, 0.4, 0.5, 0.7),
        (200, 0.6, 0.8, 0.9),
    ):
        write_seed(r0, seed, baseline, 4.0)
        write_seed(r1, seed, speed, 3.0)
        write_seed(r2, seed, full, 2.0)

    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("aggregate_condition_seeds.py")),
            "--condition",
            f"r0={r0}",
            "--condition",
            f"r1={r1}",
            "--condition",
            f"r2={r2}",
            "--baseline",
            "r0",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert set(result["paired_deltas"]) == {"r1", "r2"}
    assert math.isclose(
        result["paired_deltas"]["r1"]["metrics"][
            "final_eval/success_rate"
        ]["mean"],
        0.15,
    )
    assert math.isclose(
        result["paired_deltas"]["r2"]["metrics"][
            "final_eval/success_rate"
        ]["mean"],
        0.3,
    )
