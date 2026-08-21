"""Capacitive fuel, oil and coolant level sensors.

Physics
-------
Capacitance of a parallel-plate / coaxial sensor:

    C = epsilon A / d,   epsilon = epsilon_0 epsilon_r

As the liquid height changes the effective dielectric in the sensor
changes, so capacitance becomes a function of height:

    C = C(h)

Cylindrical vehicle volume:

    V = pi r^2 h

With vehicle-specific calibration, capacitance is converted into liquid
height and volume for fuel, oil and coolant.
"""

from __future__ import annotations

import numpy as np

from ..config import VehicleConfig

VEHICLE_GEOMETRY = {
    "fuel": {"radius": None, "permittivity": None},
    "oil": {"radius": None, "permittivity": None},
    "coolant": {"radius": None, "permittivity": None},
}


def capacitance_level(cfg: VehicleConfig, height_frac: float,
                      permittivity_r: float, area: float, gap: float) -> float:
    """Capacitance (F) of a level sensor partially immersed in liquid."""
    eps = cfg.EPSILON_0 * permittivity_r
    return eps * area * height_frac / gap


class LevelSensor:
    """Simulates capacitive fluid-level gauges for fuel, oil and coolant.

    A leak fault reduces the level faster than nominal consumption.
    """

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)
        self.fuel_h = 0.82
        self.oil_h = 0.32
        self.coolant_h = 0.38
        # Effective sensor area and gap used for calibration.
        self.area = 2.0e-3
        self.gap = 1.0e-3

    def read(self, load: float, dt: float,
             fuel_leak: float = 0.0, oil_leak: float = 0.0,
             coolant_leak: float = 0.0) -> dict[str, float]:
        cfg = self.cfg
        burn = cfg.fuel_burn_rate * load
        self.fuel_h = max(self.fuel_h - (burn + fuel_leak) * dt / (np.pi * cfg.fuel_vehicle_r**2), 0.0)
        self.oil_h = max(self.oil_h - oil_leak * dt / 0.08, 0.05)
        self.coolant_h = max(self.coolant_h - coolant_leak * dt / 0.06, 0.05)

        fuel_frac = self.fuel_h / cfg.fuel_vehicle_h
        oil_frac = self.oil_h / cfg.oil_sump_h
        coolant_frac = self.coolant_h / cfg.coolant_h

        return {
            "fuel_level": float(fuel_frac),
            "oil_level": float(oil_frac),
            "coolant_level": float(coolant_frac),
            "fuel_capacitance_pf": float(capacitance_level(
                cfg, fuel_frac, cfg.fuel_permittivity, self.area, self.gap) * 1e12),
            "fuel_volume": float(np.pi * cfg.fuel_vehicle_r**2 * self.fuel_h),
        }