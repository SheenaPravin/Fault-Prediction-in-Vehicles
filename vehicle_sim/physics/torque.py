"""Torque sensors for engine and drivetrain mechanical load.

Physics
-------
Torque from tangential force and shaft radius:

    tau = r F

Mechanical power:

    P_mech = tau * omega,    omega = 2 pi N / 60

Circular-shaft shear stress and strain:

    tau_shear = T r / J
    gamma      = T r / (J G)

Drivetrain efficiency degradation reduces the delivered torque for the
same engine power and appears as an increased torque sensor value for a
fixed wheel load.
"""

from __future__ import annotations

import numpy as np

from ..config import VehicleConfig


class TorqueSensor:
    """Simulates a strain-based shaft torque sensor on the drivetrain."""

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def omega(self, rpm: float) -> float:
        return 2.0 * np.pi * rpm / 60.0

    def read(self, rpm: float, engine_power_w: float,
             efficiency: float = 1.0) -> dict[str, float]:
        cfg = self.cfg
        omega = self.omega(rpm)
        tau_wheel = engine_power_w / max(omega, 1e-3) if omega > 0 else 0.0
        tau_shaft = tau_wheel / cfg.gear_ratio / max(efficiency, 0.05)
        tau_shaft += self.rng.normal(0.0, cfg.torque_noise)

        r = cfg.shaft_radius
        shear = tau_shaft * r / cfg.shaft_J
        gamma = tau_shaft * r / (cfg.shaft_J * cfg.shaft_shear_modulus)
        return {
            "shaft_torque": float(tau_shaft),
            "shaft_shear_stress": float(shear),
            "shaft_shear_strain": float(gamma),
            "mech_power": float(engine_power_w),
            "shaft_omega": float(omega),
        }