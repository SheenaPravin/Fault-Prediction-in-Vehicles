"""Suspension load, torsion-bar and shock-load sensors.

Physics
-------
Stress and strain:

    sigma = F / A
    sigma = E epsilon  ->  epsilon = sigma / E

Strain-gauge transduction:

    dR / R = GF * epsilon
    epsilon = dR / (GF * R)

Torsion-bar twist and torque:

    theta = T L / (J G)
    T = J G theta / L

Shock loading (Newton's second law):

    F = m a

Shock-severity indicator:

    a_RMS = sqrt((1/T) integral_0^T a^2 dt)

Shock histories accumulate to characterise terrain severity and
component loading / fatigue exposure.
"""

from __future__ import annotations

import numpy as np

from ..config import VehicleConfig


def torsion_stiffness(cfg: VehicleConfig) -> float:
    """Torsional stiffness K = J G / L in N*m/rad."""
    j = np.pi * cfg.torsion_r**4 / 2.0
    return j * cfg.torsion_G / cfg.torsion_L


class StrainSensor:
    """Strain-gauge load sensor on a suspension arm / torsion bar."""

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def read(self, vertical_accel_g: float, stiffness_mult: float = 1.0,
             fatigue_factor: float = 1.0) -> dict[str, float]:
        cfg = self.cfg
        force = cfg.vehicle_mass * 9.81 * vertical_accel_g / 6.0  # per road-wheel
        sigma = force / cfg.suspension_area * fatigue_factor
        eps = sigma / (cfg.suspension_E * stiffness_mult)
        dR = cfg.strain_gauge_GF * eps * cfg.strain_gauge_R
        force_kN = force / 1000.0 + self.rng.normal(0.0, cfg.suspension_noise)
        return {
            "susp_load_kN": float(force_kN),
            "susp_stress_MPa": float(sigma / 1e6),
            "susp_strain_ue": float(eps * 1e6),
            "susp_dR_ohm": float(dR),
        }


class TorsionBar:
    """Torsion-bar sensor: torque and twist history, fatigue accumulation."""

    def __init__(self, cfg: VehicleConfig):
        self.cfg = cfg
        self.cumulative_twist_rad = 0.0

    def read(self, torque_nm: float, stiffness_mult: float = 1.0) -> dict[str, float]:
        k = torsion_stiffness(self.cfg) * stiffness_mult
        theta = torque_nm / max(k, 1e-6)
        self.cumulative_twist_rad += abs(theta)
        j = np.pi * self.cfg.torsion_r**4 / 2.0
        shear = torque_nm * self.cfg.torsion_r / j
        return {
            "torsion_torque": float(torque_nm),
            "torsion_twist_deg": float(np.degrees(theta)),
            "torsion_shear_MPa": float(shear / 1e6),
            "torsion_cumulative_twist": float(self.cumulative_twist_rad),
        }


class ShockSensor:
    """Accelerometer measuring terrain shock loads on the hull."""

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def time_series(self, terrain: float, n_samples: int | None = None) -> np.ndarray:
        n = n_samples or self.cfg.window_samples
        t = np.arange(n) / self.cfg.sample_rate
        base = 1.0 + terrain * 4.0 * np.sin(2 * np.pi * 2.0 * t)
        impulses = np.where(np.abs(np.sin(2 * np.pi * 0.8 * t)) < 0.02,
                            terrain * 9.0, 0.0)
        return base + impulses + self.rng.normal(0.0, 0.25, n)

    def features(self, terrain: float) -> dict[str, float]:
        a = self.time_series(terrain)
        rms_a = float(np.sqrt(np.mean(a**2)))
        return {
            "shock_a_rms_g": rms_a,
            "shock_peak_g": float(np.max(np.abs(a))),
            "shock_energy": float(np.sum(a**2)),
        }