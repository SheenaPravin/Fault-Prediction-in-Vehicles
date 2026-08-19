"""Hydraulic and brake-fluid pressure sensors.

Physics
-------
Pascal's law:

    P = F / A,   F_out = P A_out

Fluid flow:

    Q = A v

Hydraulic power:

    P_hyd = P Q

Pressure, flow and temperature trends identify leakage, pump
degradation, valve faults and seal deterioration in steering, braking
and turret-stabilisation systems.
"""

from __future__ import annotations

import numpy as np

from ..config import TankConfig


class HydraulicSensor:
    """Simulates the turret-stabiliser / steering hydraulic circuit."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def read(self, cmd: float, pump_eff: float = 1.0,
             valve_fault: float = 0.0, seal_leak: float = 0.0) -> dict[str, float]:
        cfg = self.cfg
        p = cfg.hyd_pump_pressure * pump_eff * (1.0 - 0.3 * valve_fault)
        p += self.rng.normal(0.0, 1e5)
        v = cfg.hyd_valve_area * cmd
        q = v + self.rng.normal(0.0, cfg.hyd_flow_noise)
        leak_q = cfg.hyd_leak_area * np.sqrt(max(p, 0.0) / 1.0e7) * (1.0 + seal_leak)
        q_net = max(q - leak_q, 0.0)
        f_out = p * cfg.hyd_valve_area
        p_hyd = p * q_net
        return {
            "hyd_pressure": float(p),
            "hyd_flow": float(q_net),
            "hyd_force": float(f_out),
            "hyd_power": float(p_hyd),
            "hyd_leak_flow": float(leak_q),
        }