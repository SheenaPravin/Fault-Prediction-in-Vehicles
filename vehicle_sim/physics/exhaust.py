"""Exhaust-gas sensors for combustion health.

Physics
-------
Ideal gas:

    P V = n R T  ->  P = rho R_s T

Gas mass flow:

    m_dot = rho A v

Hydrocarbon combustion:

    C_x H_y + (x + y/4) O2 -> x CO2 + (y/2) H2O

Air-fuel equivalence ratio:

    lambda = (A/F)_actual / (A/F)_stoich

Exhaust temperature, pressure, oxygen concentration and flow jointly
reveal combustion abnormalities, incomplete combustion, air-fuel
imbalance and exhaust restrictions.
"""

from __future__ import annotations

import numpy as np

from ..config import VehicleConfig


class ExhaustSensor:
    """Simulates EGT, exhaust pressure, mass flow and lambda."""

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def read(self, thermal: ThermalSystem, rpm: float, load: float,
             fuel_mult: float = 1.0, air_mult: float = 1.0,
             restriction: float = 1.0) -> dict[str, float]:
        cfg = self.cfg
        t_k = thermal.T_exhaust + 273.15
        rho = cfg.exhaust_pressure_base / (cfg.exhaust_Rs * t_k)
        v = 8.0 * load * (0.4 + 0.6 * rpm / cfg.max_speed_rpm) * air_mult
        m_dot = rho * cfg.exhaust_area * v
        p_restricted = cfg.exhaust_pressure_base * (1.0 + 0.4 * restriction)

        afr_actual = (air_mult * 14.7) / max(fuel_mult * 1.0, 1e-6)
        lam = afr_actual / cfg.stoich_afr + self.rng.normal(0.0, cfg.lambda_noise)
        o2_pct = 20.9 * (lam - 1.0) / lam if lam > 1.0 else 0.0
        return {
            "exhaust_pressure": float(p_restricted),
            "exhaust_mass_flow": float(m_dot),
            "lambda": float(lam),
            "exhaust_o2_pct": float(o2_pct),
        }