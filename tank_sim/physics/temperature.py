"""Engine, transmission and exhaust temperature sensors.

Physics
-------
Thermocouple (Seebeck effect):

    V = integral[Tc..Th] S(T) dT  ~  S (Th - Tc)

RTD resistance:

    R(T) = R0 [1 + alpha (T - T0)]

Simplified engine thermal balance:

    m c_p dT/dt = Q_gen - Q_cool - Q_exhaust

Thermal deviations indicate cooling-system degradation, excessive
engine load, lubrication problems, combustion abnormalities or
transmission friction.
"""

from __future__ import annotations

import numpy as np

from ..config import TankConfig


class ThermalSystem:
    """Time-integrates the engine thermal balance.

    The heat-generation term scales with throttle/load; cooling removes
    heat at a rate proportional to the temperature difference to the
    ambient (its efficiency degrades under a cooling fault).  A fraction
    of the generated heat is rejected through the exhaust.
    """

    def __init__(self, cfg: TankConfig, ambient: float = 25.0):
        self.cfg = cfg
        self.ambient = ambient
        self.T_engine = 45.0 + 0.1 * cfg.noise_seed  # initial warm-up state
        self.T_oil = self.T_engine - 8.0
        self.T_exhaust = 120.0

    def step(self, load: float, rpm: float,
             cooling_eff: float = 1.0, load_multiplier: float = 1.0) -> None:
        cfg = self.cfg
        q_gen = cfg.max_fuel_energy_rate * load * load_multiplier
        # Radiator + fan + oil cooler effectiveness rises with duty cycle,
        # otherwise the linear balance never reaches an equilibrium.
        k_eff = cfg.coolant_power * (1.0 + 3.0 * load)
        q_cool = k_eff * max(self.T_engine - self.ambient, 0.0) * cooling_eff
        q_exhaust = 0.30 * q_gen
        dT = (q_gen - q_cool - q_exhaust) / (cfg.engine_mass_thermal * cfg.c_p_coolant)
        self.T_engine += dT * cfg.dt
        self.T_oil += (self.T_engine - self.T_oil) * cfg.dt * 0.8 + 0.06 * dT
        self.T_exhaust += (self.T_engine * 0.85 + 90.0 - self.T_exhaust) * cfg.dt * 2.0


class EngineTemperatureSensor:
    """RTD reading of engine coolant temperature."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)
        self.alpha = 0.00385  # 1/K for platinum RTD
        self.R0 = 100.0
        self.T0 = 0.0

    def resistance(self, temperature_c: float) -> float:
        return self.R0 * (1.0 + self.alpha * (temperature_c - self.T0))

    def read(self, thermal: ThermalSystem) -> dict[str, float]:
        t = thermal.T_engine + self.rng.normal(0.0, 0.4)
        return {"coolant_temp": t, "coolant_rtd_ohm": self.resistance(t)}


class CoolantSensor:
    """Exhaust-gas temperature (thermocouple) reading."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)
        self.seebeck = 41.0e-6  # V/K (type-K approximation)
        self.T_cold = 25.0

    def read(self, thermal: ThermalSystem) -> dict[str, float]:
        t = thermal.T_exhaust + self.rng.normal(0.0, 1.2)
        emf = self.seebeck * (t - self.T_cold)
        return {"exhaust_temp": t, "exhaust_thermocouple_v": emf}