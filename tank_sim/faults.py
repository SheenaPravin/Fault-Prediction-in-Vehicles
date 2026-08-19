"""Fault-injection manager.

Each fault profile maps a time-varying severity (0 = healthy, 1+ = severe)
to physical-parameter multipliers that are consumed by the sensor
modules.  This keeps the simulation physically consistent: a bearing
fault raises vibration amplitude *and* kurtosis *and* debris rate *and*
slightly heats the oil, matching the sensor-fusion narrative of
section 15 of the physics document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaultProfile:
    name: str
    start_step: int
    ramp_steps: int = 2000
    max_severity: float = 1.0
    noise: float = 0.02

    def severity(self, step: int, rng: np.random.Generator) -> float:
        """Sigmoid ramp from 0 to max severity after the fault start."""
        if step < self.start_step:
            return 0.0
        progress = (step - self.start_step) / max(self.ramp_steps, 1)
        base = self.max_severity * (1.0 - math.exp(-3.0 * progress))
        return max(0.0, base + rng.normal(0.0, self.noise))


class FaultManager:
    """Computes the aggregate physics-parameter multipliers for a set of
    active fault profiles at a given simulation step."""

    # Defines which physical parameters each fault perturbs and how.
    FAULT_MAP: dict[str, dict] = {
        "bearing_wear": {
            "vib_severity": lambda s: s,
            "vib_defect": lambda s: "bearing_outer" if s > 0 else "none",
            "debris_severity": lambda s: s,
            "oil_temp_bias": lambda s: 0.6 * s,
        },
        "gear_wear": {
            "vib_severity": lambda s: s,
            "vib_defect": lambda s: "gear_wear" if s > 0 else "none",
            "debris_severity": lambda s: 0.5 * s,
        },
        "cooling_failure": {
            "cooling_eff": lambda s: max(1.0 - 0.75 * s, 0.05),
            "coolant_leak": lambda s: 1.0e-4 * s,
        },
        "oil_pump_degradation": {
            "pump_eff": lambda s: max(1.0 - 0.55 * s, 0.2),
            "debris_severity": lambda s: 0.3 * s,
        },
        "bearing_clearance_wear": {
            "gallery_r_mult": lambda s: 1.0 + 0.25 * s,
            "flow_mult": lambda s: 1.0 + 0.2 * s,
            "debris_severity": lambda s: 0.8 * s,
            "oil_temp_bias": lambda s: 0.4 * s,
        },
        "seal_leakage": {
            "oil_leak": lambda s: 2.0e-5 * s,
            "hyd_seal_leak": lambda s: s,
            "coolant_leak": lambda s: 5.0e-5 * s,
        },
        "fuel_injector_fault": {
            "fuel_mult": lambda s: 1.0 + 0.35 * math.sin(s) + 0.05 * s,
            "air_mult": lambda s: max(1.0 - 0.3 * s, 0.6),
            "load_multiplier": lambda s: 1.0 + 0.05 * s,
        },
        "exhaust_restriction": {
            "restriction": lambda s: s,
            "cooling_eff": lambda s: max(1.0 - 0.2 * s, 0.4),
        },
        "torsion_fatigue": {
            "stiffness_mult": lambda s: max(1.0 - 0.35 * s, 0.5),
            "fatigue_factor": lambda s: 1.0 + 0.8 * s,
            "ae_severity": lambda s: s,
        },
        "hydraulic_valve_fault": {
            "valve_fault": lambda s: s,
            "pump_eff": lambda s: max(1.0 - 0.25 * s, 0.4),
        },
        "structural_crack": {
            "ae_severity": lambda s: s,
            "fatigue_factor": lambda s: 1.0 + 0.5 * s,
        },
        "drivetrain_efficiency_loss": {
            "efficiency": lambda s: max(1.0 - 0.18 * s, 0.7),
            "debris_severity": lambda s: 0.6 * s,
        },
    }

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(7)
        self.profiles: list[FaultProfile] = []

    def add(self, name: str, start_step: int, ramp_steps: int = 2000,
            max_severity: float = 1.0, noise: float = 0.02) -> "FaultManager":
        self.profiles.append(FaultProfile(name, start_step, ramp_steps,
                                          max_severity, noise))
        return self

    def parameters(self, step: int) -> dict[str, float]:
        """Aggregate parameter multipliers for a simulation step.

        Multiplicative parameters default to 1.0; additive biases and
        severities default to 0.0.
        """
        params: dict[str, float] = {
            "vib_severity": 0.0, "vib_defect": "none", "debris_severity": 0.0,
            "oil_temp_bias": 0.0, "cooling_eff": 1.0, "pump_eff": 1.0,
            "gallery_r_mult": 1.0, "flow_mult": 1.0, "oil_leak": 0.0,
            "hyd_seal_leak": 0.0, "coolant_leak": 0.0, "fuel_mult": 1.0,
            "air_mult": 1.0, "restriction": 0.0, "stiffness_mult": 1.0,
            "fatigue_factor": 1.0, "ae_severity": 0.0, "valve_fault": 0.0,
            "efficiency": 1.0, "load_multiplier": 1.0,
        }
        additive = {"vib_severity", "debris_severity", "ae_severity",
                    "oil_temp_bias", "restriction", "valve_fault",
                    "oil_leak", "hyd_seal_leak", "coolant_leak"}
        for profile in self.profiles:
            s = profile.severity(step, self.rng)
            if s <= 0.0:
                continue
            for key, fn in self.FAULT_MAP[profile.name].items():
                val = fn(s)
                if key in ("vib_defect",):
                    if s > 0 and params.get("vib_defect", "none") == "none":
                        params[key] = val
                elif key in additive:
                    params[key] += val
                else:
                    params[key] *= val
        # Normalise additive severities so multiple faults combine sensibly.
        params["vib_severity"] = min(params["vib_severity"], 3.0)
        params["debris_severity"] = min(params["debris_severity"], 3.0)
        params["ae_severity"] = min(params["ae_severity"], 3.0)
        return params

    def labels(self) -> list[str]:
        return [p.name for p in self.profiles]

    def active_faults(self, step: int) -> list[str]:
        return [p.name for p in self.profiles if step >= p.start_step]