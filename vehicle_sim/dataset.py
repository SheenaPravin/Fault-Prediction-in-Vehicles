"""Dataset persistence for the simulated sensor stream."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import HealthFeatures
from .vehicle import VehicleSimulator


def write_dataset(sim: VehicleSimulator, out_path: str,
                  add_health: bool = True, health_window: int = 100) -> Path:
    """Run the simulation and write a labelled CSV dataset.

    Columns are the sensor readings, physics-derived health features,
    fused health index, RUL estimate, anomaly score and one-hot fault
    labels.
    """
    records = sim.run()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(sim.total_columns())
    hf = HealthFeatures(records)

    health = hf.fused_health_index() if add_health else [float("nan")] * len(records)
    rul = hf.rul(health_window) if add_health else [float("nan")] * len(records)
    anomaly = hf.anomaly_score() if add_health else [float("nan")] * len(records)

    for i, r in enumerate(records):
        r["health_index"] = health[i]
        r["rul_steps"] = rul[i]
        r["anomaly_score"] = anomaly[i]

    extra_cols = ["health_index", "rul_steps", "anomaly_score"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols + extra_cols)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return path


__all__ = ["write_dataset"]