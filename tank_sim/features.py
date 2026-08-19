"""Physics-derived health features and sensor-fusion RUL estimation.

Implements section 15/16 of the physics document:

    x(t) = [a, T, P, Q, tau, h, epsilon, p_acoustic, Np]
    x(t) -> Health State
    x(t) -> P(Fault_i)
    RUL = f(x_{1:t})

The fused health index maps normalised deviations of the physics-based
features onto a 0-100 score; RUL is estimated by fitting the health
index trajectory and extrapolating to a failure threshold.
"""

from __future__ import annotations

import numpy as np

from .physics.vibration import kurtosis, rms

FAILURE_THRESHOLD = 20.0  # health index at which failure is declared


class HealthFeatures:
    """Extracts health features from a window of raw sensor records."""

    def __init__(self, records: list[dict[str, float]] | None = None):
        self.records = records or []

    def add(self, record: dict[str, float]) -> None:
        self.records.append(record)

    def trends(self, column: str, window: int = 100) -> list[float]:
        """Slope of a sensor column over a trailing window (linear fit)."""
        n = len(self.records)
        out: list[float] = []
        for i in range(n):
            start = max(0, i - window + 1)
            seg = self.records[start:i + 1]
            if len(seg) < 3:
                out.append(0.0)
                continue
            x = np.arange(len(seg))
            y = np.array([r[column] for r in seg], dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            out.append(float(slope))
        return out

    def fused_health_index(self) -> list[float]:
        """Map normalised multi-sensor deviation onto a 0-100 health score.

        A higher deviation of damage-indicative features lowers the score.
        """
        cfg_ref = {
            "vib_rms": 1.2, "vib_kurtosis": 4.0, "oil_pressure": 4.2e5,
            "oil_temp": 95.0, "coolant_temp": 95.0, "debris_rate": 2.0,
            "debris_cumulative": 200.0, "ae_event_rate": 3.0, "ae_energy": 2.0,
            "susp_stress_MPa": 120.0, "torsion_shear_MPa": 300.0,
            "lambda": 1.0, "hyd_pressure": 1.8e7,
        }
        out: list[float] = []
        for r in self.records:
            dev = 0.0
            n = 0
            for col, ref in cfg_ref.items():
                if col not in r:
                    continue
                val = abs(r[col] - ref)
                # Lambda deviation is symmetric; most others damage up.
                if col == "lambda":
                    dev += val / 0.2
                elif col in ("oil_pressure", "hyd_pressure"):
                    dev += max(ref - r[col], 0.0) / (ref * 0.3)
                else:
                    dev += val / (ref * 1.0)
                n += 1
            score = 100.0 * (1.0 - min(dev / max(n, 1) / 2.5, 1.0))
            out.append(float(np.clip(score, 0.0, 100.0)))
        return out

    def rul(self, window: int = 500) -> list[float]:
        """Estimate remaining useful life by extrapolating the fused
        health-index trajectory to the failure threshold."""
        health = self.fused_health_index()
        n = len(health)
        out: list[float] = [float("nan")] * n
        for i in range(n):
            start = max(0, i - window + 1)
            seg = health[start:i + 1]
            if len(seg) < 10:
                out[i] = float(n - i)
                continue
            x = np.arange(len(seg))
            slope = np.polyfit(x, seg, 1)[0]
            if slope >= 0:
                out[i] = float(n - i)
                continue
            remaining_steps = (FAILURE_THRESHOLD - seg[-1]) / slope
            out[i] = float(max(remaining_steps, 0.0))
        return out

    def anomaly_score(self) -> list[float]:
        """Simple z-score anomaly score from the fused health deviation."""
        health = np.array(self.fused_health_index())
        mean = np.mean(health)
        std = np.std(health) + 1e-9
        return [float((mean - h) / std) for h in health]


__all__ = ["HealthFeatures", "rms", "kurtosis", "FAILURE_THRESHOLD"]