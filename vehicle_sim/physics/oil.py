"""Oil pressure and oil-debris sensors.

Physics
-------
Pressure definition:

    P = F / A

Hagen-Poiseuille laminar flow through a cylindrical passage:

    Q = (pi r^4 / (8 mu L)) dP
    dP = 8 mu L Q / (pi r^4)

So oil pressure must be interpreted with engine speed, oil temperature,
flow and load instead of a fixed threshold.

Arrhenius-type temperature-viscosity model (fluid degradation):

    mu(T) = A exp[E / (R T)]
    dP proportional mu Q        (laminar)

Oil-debris sensing relies on the electromagnetic induction of metallic
particles changing coil inductance/impedance:

    B = mu n I,  Z = R + j w L,  L ~ mu N^2 A / l

Particle-generation rate:

    Np_dot = dNp/dt

A rapidly increasing particle-generation rate indicates accelerating
mechanical wear.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import VehicleConfig


def oil_viscosity(cfg: VehicleConfig, temperature_c: float) -> float:
    """Arrhenius-type viscosity in Pa*s at a given oil temperature (C)."""
    t_k = temperature_c + 273.15
    return cfg.oil_A * math.exp(cfg.oil_E / (cfg.R_UNIVERSAL * t_k))


def pressure_drop(cfg: VehicleConfig, viscosity: float, flow: float,
                  radius: float, length: float) -> float:
    """Hagen-Poiseuille pressure drop across a passage (Pa)."""
    return 8.0 * viscosity * length * flow / (math.pi * radius**4)


class OilPressureSensor:
    """Simulates the main-gallery oil pressure gauge.

    Pump head is reduced under a pump fault; bearing/journal wear opens
    clearances (larger effective gallery radius) which lowers the
    pressure drop through the filter and gallery.
    """

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def read(self, thermal: ThermalSystem, load: float,
             pump_eff: float = 1.0, gallery_r_mult: float = 1.0,
             flow_mult: float = 1.0) -> dict[str, float]:
        cfg = self.cfg
        mu = oil_viscosity(cfg, thermal.T_oil)
        # Flow scales with engine speed and load.
        q = (0.0006 + 0.0012 * load) * flow_mult
        dP_filter = pressure_drop(cfg, mu, q, cfg.filter_r, cfg.filter_L)
        dP_gallery = pressure_drop(cfg, mu, q,
                                   cfg.main_gallery_r * gallery_r_mult,
                                   cfg.main_gallery_L)
        p = cfg.pump_discharge_pressure * pump_eff - dP_filter - dP_gallery
        p += self.rng.normal(0.0, cfg.oil_pressure_noise)
        return {
            "oil_pressure": float(max(p, 0.0)),
            "oil_temp": thermal.T_oil,
            "oil_viscosity": mu,
            "oil_flow": q,
        }


class OilDebrisSensor:
    """Counts metallic wear particles via inductance/impedance change.

    Inductance of the sensing coil: L ~ mu N^2 A / l.  Each particle
    perturbs L, producing an event whose magnitude encodes particle
    size and material.  The generation rate rises with wear severity.
    """

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)
        self.coil_N = 200
        self.coil_A = 2.0e-4
        self.coil_l = 0.02
        self.count = 0.0
        self.last_count = 0.0

    def base_inductance(self, mu: float) -> float:
        return mu * self.coil_N**2 * self.coil_A / self.coil_l

    def read(self, severity: float, dt: float) -> dict[str, float]:
        cfg = self.cfg
        rate = cfg.debris_gain * (1.0 + 30.0 * severity)
        self.last_count = self.count
        self.count += rate * dt
        count_in_window = max(self.count - self.last_count, 0.0)
        rate_noise = self.rng.poisson(max(count_in_window, 0.0))
        particles_this_step = rate_noise * (1.0 + cfg.debris_noise * severity)
        cumulative = self.count * (1.0 + 0.5 * severity)
        return {
            "debris_particles": float(particles_this_step),
            "debris_cumulative": float(cumulative),
            "debris_rate": float(rate * (1.0 + severity)),
        }