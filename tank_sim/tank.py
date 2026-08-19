"""Battle-tank digital-twin orchestrator.

Couples the physics modules through a shared evolving state, applies
fault-induced parameter perturbations, samples every sensor each time
step and assembles a labelled record (feature vector + fault labels)
ready for the AI layer.

    Sensors -> Physics Features -> Digital Twin -> AI
                      (anomaly / fault / RUL)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TankConfig
from .faults import FaultManager
from .physics import (
    AcousticEmissionSensor,
    AcousticSensor,
    CoolantSensor,
    EngineTemperatureSensor,
    ExhaustSensor,
    HydraulicSensor,
    LevelSensor,
    OilDebrisSensor,
    OilPressureSensor,
    ShockSensor,
    StrainSensor,
    TorqueSensor,
    TorsionBar,
    VibrationSensor,
    ThermalSystem,
)

SENSOR_COLUMNS = [
    "time", "step", "rpm", "load", "terrain",
    "coolant_temp", "coolant_rtd_ohm", "exhaust_temp", "exhaust_thermocouple_v",
    "exhaust_pressure", "exhaust_mass_flow",
    "lambda", "exhaust_o2_pct", "oil_pressure", "oil_temp", "oil_viscosity",
    "oil_flow", "debris_cumulative", "debris_rate", "debris_particles",
    "shaft_torque", "shaft_shear_stress", "shaft_shear_strain",
    "mech_power", "shaft_omega", "fuel_level", "fuel_volume",
    "oil_level", "coolant_level", "fuel_capacitance_pf", "hyd_pressure",
    "hyd_flow", "hyd_force", "hyd_power", "hyd_leak_flow",
    "susp_load_kN", "susp_stress_MPa",
    "susp_strain_ue", "susp_dR_ohm", "torsion_torque", "torsion_twist_deg",
    "torsion_shear_MPa",
    "torsion_cumulative_twist", "shock_a_rms_g", "shock_peak_g", "shock_energy",
    "spl_db",
    "acoustic_dom_freq", "acoustic_energy", "ae_event_rate", "ae_events",
    "ae_energy",
    "ae_amp_dB", "ae_duration_s", "vib_rms", "vib_kurtosis", "vib_dom_freq",
    "vib_dom_amp",
    "vib_energy",
]

FAULT_COLUMN_PREFIX = "fault_"


@dataclass
class MissionStep:
    """One entry of the mission/usage profile."""
    duration_s: float
    rpm: float
    load: float
    terrain: float


def default_mission(cfg: TankConfig) -> list[MissionStep]:
    """A representative tank mission: idle -> road cruise -> sprint ->
    rough-terrain traverse -> idle (hot soak)."""
    return [
        MissionStep(60.0, cfg.idle_speed_rpm, 0.10, 0.1),
        MissionStep(240.0, 1500.0, 0.45, 0.2),
        MissionStep(120.0, 2200.0, 0.80, 0.3),
        MissionStep(180.0, 1800.0, 0.60, 0.9),
        MissionStep(120.0, 2600.0, 0.95, 0.6),
        MissionStep(180.0, 900.0, 0.15, 0.2),
    ]


class TankSimulator:
    """Runs the coupled physics simulation and yields labelled samples."""

    def __init__(self, cfg: TankConfig | None = None,
                 faults: FaultManager | None = None,
                 mission: list[MissionStep] | None = None,
                 seed: int | None = None):
        self.cfg = cfg or TankConfig()
        seed = seed if seed is not None else self.cfg.noise_seed
        self.rng = np.random.default_rng(seed)
        self.faults = faults or FaultManager(self.rng)
        self.mission = mission or default_mission(self.cfg)

        self.thermal = ThermalSystem(self.cfg)
        self.vib = VibrationSensor(self.cfg, self.rng)
        self.eng_temp = EngineTemperatureSensor(self.cfg, self.rng)
        self.coolant = CoolantSensor(self.cfg, self.rng)
        self.oil_p = OilPressureSensor(self.cfg, self.rng)
        self.debris = OilDebrisSensor(self.cfg, self.rng)
        self.torque = TorqueSensor(self.cfg, self.rng)
        self.exhaust = ExhaustSensor(self.cfg, self.rng)
        self.level = LevelSensor(self.cfg, self.rng)
        self.hyd = HydraulicSensor(self.cfg, self.rng)
        self.strain = StrainSensor(self.cfg, self.rng)
        self.torsion = TorsionBar(self.cfg)
        self.shock = ShockSensor(self.cfg, self.rng)
        self.acoustic = AcousticSensor(self.cfg, self.rng)
        self.ae = AcousticEmissionSensor(self.cfg, self.rng)

    def fault_columns(self) -> list[str]:
        return [FAULT_COLUMN_PREFIX + name for name in self.faults.labels()]

    def total_columns(self) -> list[str]:
        return SENSOR_COLUMNS + self.fault_columns()

    def run(self) -> list[dict[str, float]]:
        """Simulate the full mission and return one record per time step."""
        cfg = self.cfg
        records: list[dict[str, float]] = []
        t = 0.0
        step = 0
        for mission in self.mission:
            n_steps = int(round(mission.duration_s / cfg.dt))
            for _ in range(n_steps):
                params = self.faults.parameters(step)
                self._advance(mission, params)
                record = self._sample(mission, params, t, step)
                records.append(record)
                t += cfg.dt
                step += 1
        return records

    def _advance(self, mission: MissionStep, params: dict) -> None:
        cfg = self.cfg
        rpm_frac = mission.rpm / cfg.max_speed_rpm
        self.thermal.step(mission.load * rpm_frac, mission.rpm,
                          cooling_eff=params["cooling_eff"],
                          load_multiplier=params["load_multiplier"])
        if params["oil_temp_bias"]:
            self.thermal.T_oil += params["oil_temp_bias"] * cfg.dt

    def _sample(self, mission: MissionStep, params: dict,
                t: float, step: int) -> dict[str, float]:
        cfg = self.cfg
        rpm, load, terrain = mission.rpm, mission.load, mission.terrain

        eng = self.eng_temp.read(self.thermal)
        cool = self.coolant.read(self.thermal)
        oil = self.oil_p.read(self.thermal, load,
                              pump_eff=params["pump_eff"],
                              gallery_r_mult=params["gallery_r_mult"],
                              flow_mult=params["flow_mult"])
        debris = self.debris.read(params["debris_severity"], cfg.dt)

        rpm_frac = rpm / cfg.max_speed_rpm
        power = cfg.max_fuel_energy_rate * load * rpm_frac / 1000.0  # kW
        tor = self.torque.read(rpm, power * 1000.0,
                               efficiency=params["efficiency"])
        exh = self.exhaust.read(self.thermal, rpm, load,
                                fuel_mult=params["fuel_mult"],
                                air_mult=params["air_mult"],
                                restriction=params["restriction"])
        lvl = self.level.read(load, cfg.dt,
                              fuel_leak=params["oil_leak"] * 0.0,
                              oil_leak=params["oil_leak"],
                              coolant_leak=params["coolant_leak"])
        hyd = self.hyd.read(load, pump_eff=params["pump_eff"],
                            valve_fault=params["valve_fault"],
                            seal_leak=params["hyd_seal_leak"])
        strain = self.strain.read(1.0 + terrain,
                                  stiffness_mult=params["stiffness_mult"],
                                  fatigue_factor=params["fatigue_factor"])
        torbar = self.torsion.read(abs(tor["shaft_torque"]),
                                   stiffness_mult=params["stiffness_mult"])
        shock = self.shock.features(terrain)
        acous = self.acoustic.features(rpm)
        ae = self.ae.read(params["ae_severity"], cfg.dt)
        vib = self.vib.features(rpm, params["vib_severity"],
                                params.get("vib_defect", "none"))

        rec = {
            "time": t, "step": float(step), "rpm": float(rpm),
            "load": float(load), "terrain": float(terrain),
        }
        rec.update(eng); rec.update(cool); rec.update(oil); rec.update(debris)
        rec.update(tor); rec.update(exh); rec.update(lvl); rec.update(hyd)
        rec.update(strain); rec.update(torbar); rec.update(shock)
        rec.update(acous); rec.update(ae); rec.update(vib)
        for name in self.faults.labels():
            rec[FAULT_COLUMN_PREFIX + name] = 1.0 if name in self.faults.active_faults(step) else 0.0
        return rec