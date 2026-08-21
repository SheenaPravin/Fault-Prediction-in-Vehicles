"""Physics-based sensor simulation modules.

Each module implements the equations documented in
``Physics_Based_Sensor_Equations_Military_Tank_Preventive_Maintenance.docx``
and exposes a ``Sensor`` class that converts the evolving shared
state into a measured signal, plus the physics-derived features that
feed the AI layer.
"""

from .vibration import VibrationSensor, characteristic_frequencies
from .temperature import ThermalSystem, CoolantSensor, EngineTemperatureSensor
from .oil import OilPressureSensor, OilDebrisSensor, oil_viscosity, pressure_drop
from .torque import TorqueSensor
from .exhaust import ExhaustSensor
from .level import LevelSensor, capacitance_level
from .hydraulics import HydraulicSensor
from .suspension import StrainSensor, TorsionBar, ShockSensor
from .acoustics import AcousticSensor, AcousticEmissionSensor

__all__ = [
    "VibrationSensor",
    "characteristic_frequencies",
    "ThermalSystem",
    "CoolantSensor",
    "EngineTemperatureSensor",
    "OilPressureSensor",
    "OilDebrisSensor",
    "oil_viscosity",
    "pressure_drop",
    "TorqueSensor",
    "ExhaustSensor",
    "LevelSensor",
    "capacitance_level",
    "HydraulicSensor",
    "StrainSensor",
    "TorsionBar",
    "ShockSensor",
    "AcousticSensor",
    "AcousticEmissionSensor",
]