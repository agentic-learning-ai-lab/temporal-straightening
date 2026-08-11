import json
import math
import subprocess
import sys
from pathlib import Path


def test_aggregate_plan_chunks(tmp_path: Path) -> None:
    chunks = []
    for offset, success, state, visual_norm in (
        (0, 0.1, 2.0, 3.0),
        (10, 0.3, 4.0, 4.0),
    ):
        path = tmp_path / f"chunk_{offset}.json"
        records = [
            {
                "final_eval/success_rate": success,
                "final_eval/mean_state_dist": state,
                "final_eval/mean_visual_dist": state + 1,
                "final_eval/mean_proprio_dist": state + 2,
                "final_eval/mean_div_visual_emb": visual_norm,
                "final_eval/mean_div_proprio_emb": visual_norm + 1,
            }
        ]
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        chunks.extend(["--chunk", f"{offset}:10:{path}"])

    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("aggregate_plan_chunks.py")),
            *chunks,
            "--expected-evals",
            "20",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    metrics = result["metrics"]["final_eval"]
    assert metrics["final_eval/success_rate"] == 0.2
    assert metrics["final_eval/mean_state_dist"] == 3.0
    assert metrics["final_eval/mean_div_visual_emb"] == 5.0
    assert math.isclose(
        metrics["final_eval/mean_div_proprio_emb"], math.sqrt(4.0**2 + 5.0**2)
    )
    assert result["eval_seeds"] == [100 * n + 1 for n in range(20)]
