from pathlib import Path

import pandas as pd
import pytest

from scripts.benchmark.repeat_split_sensitivity import describe
from scripts.benchmark.summarize_crossattack_temporal_baselines import heldout_metrics


def test_describe_reports_distribution_bounds() -> None:
    summary = describe([0.6, 0.7, 0.7, 0.7])

    assert summary["median"] == pytest.approx(0.7)
    assert summary["min"] == pytest.approx(0.6)
    assert summary["max"] == pytest.approx(0.7)
    assert summary["q1"] <= summary["median"] <= summary["q3"]


def test_heldout_metrics_filters_complete_clips(tmp_path: Path) -> None:
    rows = []
    for clip_id in (0, 1, 2):
        for local_frame in range(24):
            attacked = int(local_frame in (4, 5, 6))
            rows.append(
                {
                    "clip_id": clip_id,
                    "adversarial_label": attacked,
                    "score": 0.9 if attacked else 0.1,
                }
            )
    path = tmp_path / "frames.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    metrics = heldout_metrics(path, "score", 0.7, [0, 2])

    assert metrics["frames"] == 48
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)
