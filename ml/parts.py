"""Subsystem (part) definitions, parameter thresholds and LSTM input
feature schema shared between the Python pipeline and the web dashboard.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-part parameters: used for gauges, digital displays, thresholds and
# per-part health scoring.  Each parameter carries display metadata and
# safety thresholds (warn_lo/warn_hi = yellow zone, crit_lo/crit_hi = red
# zone / SOS).
# ---------------------------------------------------------------------------

PARTS = {
    "engine": {
        "label": "Engine",
        "params": [
            {"key": "coolant_temp", "label": "Coolant Temp", "unit": "°C",
             "min": 0, "max": 150, "warn_hi": 105, "crit_hi": 120,
             "healthy": 80},
            {"key": "oil_temp", "label": "Oil Temp", "unit": "°C",
             "min": 0, "max": 160, "warn_hi": 110, "crit_hi": 130,
             "healthy": 75},
            {"key": "exhaust_temp", "label": "Exhaust Temp", "unit": "°C",
             "min": 0, "max": 500, "warn_hi": 240, "crit_hi": 300,
             "healthy": 170},
            {"key": "lambda", "label": "Air-Fuel Ratio λ", "unit": "",
             "min": 0.6, "max": 1.6, "warn_lo": 0.9, "warn_hi": 1.2,
             "crit_lo": 0.8, "crit_hi": 1.4, "healthy": 1.0},
        ],
        "gauge": "coolant_temp",
        "alarm_key": "coolant_temp",
    },
    "powertrain": {
        "label": "Powertrain",
        "params": [
            {"key": "shaft_torque", "label": "Shaft Torque", "unit": "kN·m",
             "min": 0, "max": 1.2, "warn_hi": 0.75, "crit_hi": 0.85,
             "healthy": 0.45, "scale": 0.001, "decimals": 2},
            {"key": "vib_rms", "label": "Vibration RMS", "unit": "m/s²",
             "min": 0, "max": 2.5, "warn_hi": 0.75, "crit_hi": 1.2,
             "healthy": 0.36},
            {"key": "vib_kurtosis", "label": "Vibration Kurtosis", "unit": "",
             "min": 0, "max": 30, "warn_hi": 4.5, "crit_hi": 8,
             "healthy": 2.0},
            {"key": "vib_dom_amp", "label": "Dominant FFT Amp", "unit": "",
             "min": 0, "max": 400, "warn_hi": 160, "crit_hi": 220,
             "healthy": 105},
        ],
        "gauge": "vib_rms",
        "alarm_key": "vib_kurtosis",
    },
    "lubrication": {
        "label": "Lubrication",
        "params": [
            {"key": "oil_pressure", "label": "Oil Pressure", "unit": "bar",
             "min": 0, "max": 7, "warn_lo": 3.2, "crit_lo": 2.0,
             "healthy": 5.0, "scale": 1e-5},
            {"key": "oil_temp", "label": "Oil Temp", "unit": "°C",
             "min": 0, "max": 160, "warn_hi": 110, "crit_hi": 130,
             "healthy": 75},
            {"key": "debris_rate", "label": "Debris Rate", "unit": "/s",
             "min": 0, "max": 40, "warn_hi": 8, "crit_hi": 15,
             "healthy": 1.0},
            {"key": "debris_cumulative", "label": "Total Debris", "unit": "pts",
             "min": 0, "max": 4000, "warn_hi": 1200, "crit_hi": 2000,
             "healthy": 200},
        ],
        "gauge": "oil_pressure",
        "alarm_key": "oil_pressure",
    },
    "cooling": {
        "label": "Cooling System",
        "params": [
            {"key": "coolant_temp", "label": "Coolant Temp", "unit": "°C",
             "min": 0, "max": 150, "warn_hi": 105, "crit_hi": 120,
             "healthy": 90},
            {"key": "coolant_level", "label": "Coolant Level", "unit": "%",
             "min": 0, "max": 100, "warn_lo": 55, "crit_lo": 35,
             "healthy": 95, "scale": 100},
            {"key": "exhaust_pressure", "label": "Exhaust Press.", "unit": "bar",
             "min": 0, "max": 3, "warn_hi": 2.2, "crit_hi": 2.6,
             "healthy": 1.25, "scale": 1e-5},
        ],
        "gauge": "coolant_temp",
        "alarm_key": "coolant_temp",
    },
    "hydraulics": {
        "label": "Hydraulics",
        "params": [
            {"key": "hyd_pressure", "label": "Circuit Pressure", "unit": "bar",
             "min": 0, "max": 300, "warn_lo": 140, "crit_lo": 90,
             "healthy": 210, "scale": 1e-5},
            {"key": "hyd_force", "label": "Actuator Force", "unit": "kN",
             "min": 0, "max": 6, "warn_lo": 2.9, "crit_lo": 2.0,
             "healthy": 4.2, "scale": 1e-3},
            {"key": "hyd_leak_flow", "label": "Seal Leak Flow", "unit": "L/s",
             "min": 0, "max": 0.08, "warn_hi": 0.02, "crit_hi": 0.04,
             "healthy": 0.0002, "scale": 1000, "decimals": 4},
        ],
        "gauge": "hyd_pressure",
        "alarm_key": "hyd_pressure",
    },
    "suspension": {
        "label": "Suspension",
        "params": [
            {"key": "susp_load_kN", "label": "Road-Wheel Load", "unit": "kN",
             "min": 0, "max": 220, "warn_hi": 155, "crit_hi": 175,
             "healthy": 115},
            {"key": "susp_strain_ue", "label": "Strain", "unit": "με",
             "min": 0, "max": 600, "warn_hi": 260, "crit_hi": 320,
             "healthy": 140},
            {"key": "shock_a_rms_g", "label": "Shock RMS", "unit": "g",
             "min": 0, "max": 8, "warn_hi": 3.2, "crit_hi": 4.5,
             "healthy": 1.8},
        ],
        "gauge": "susp_load_kN",
        "alarm_key": "susp_strain_ue",
    },
    "structure": {
        "label": "Structure / Torsion",
        "params": [
            {"key": "torsion_twist_deg", "label": "Torsion Twist", "unit": "°",
             "min": 0, "max": 1.5, "warn_hi": 0.65, "crit_hi": 0.75,
             "healthy": 0.4, "decimals": 2},
            {"key": "torsion_cumulative_twist", "label": "Cum. Twist", "unit": "rad",
             "min": 0, "max": 4000, "warn_hi": 1200, "crit_hi": 2000,
             "healthy": 300},
            {"key": "ae_event_rate", "label": "AE Event Rate", "unit": "/s",
             "min": 0, "max": 40, "warn_hi": 7, "crit_hi": 12,
             "healthy": 2.0},
            {"key": "ae_energy", "label": "AE Energy", "unit": "",
             "min": 0, "max": 40, "warn_hi": 8, "crit_hi": 15,
             "healthy": 0.5},
        ],
        "gauge": "torsion_twist_deg",
        "alarm_key": "ae_event_rate",
    },
    "overall": {
        "label": "Overall Tank",
        "params": [
            {"key": "health_index", "label": "Fused Health Index", "unit": "",
             "min": 0, "max": 100, "warn_lo": 40, "crit_lo": 25,
             "healthy": 90},
            {"key": "spl_db", "label": "Noise SPL", "unit": "dB",
             "min": 60, "max": 150, "warn_hi": 128, "crit_hi": 135,
             "healthy": 110},
            {"key": "fuel_level", "label": "Fuel Level", "unit": "%",
             "min": 0, "max": 100, "warn_lo": 20, "crit_lo": 10,
             "healthy": 90, "scale": 100},
        ],
        "gauge": "health_index",
        "alarm_key": "health_index",
    },
}

PART_ORDER = ["engine", "powertrain", "lubrication", "cooling",
              "hydraulics", "suspension", "structure", "overall"]

# LSTM input features (normalised 0-1) spanning all subsystems.
INPUT_FEATURES = [
    "coolant_temp", "oil_temp", "exhaust_temp", "oil_pressure",
    "oil_viscosity", "debris_rate", "debris_cumulative", "shaft_torque",
    "lambda", "hyd_pressure", "hyd_leak_flow", "susp_load_kN",
    "susp_strain_ue", "torsion_twist_deg", "torsion_cumulative_twist",
    "shock_a_rms_g", "spl_db", "ae_event_rate", "ae_energy", "vib_rms",
    "vib_kurtosis", "vib_dom_amp",
]

# Failure threshold for per-part health index (RUL = time to cross it).
FAIL_HEALTH = 25.0
# RUL cap (steps) for piecewise-linear RUL targets.
RUL_CAP_STEPS = 2400


def part_features(part: str) -> list[str]:
    return [p["key"] for p in PARTS[part]["params"]]


def healthy_reference(part: str) -> dict[str, float]:
    return {p["key"]: p["healthy"] for p in PARTS[part]["params"]}


def threshold_span(part: str) -> dict[str, float]:
    """Normalisation span = distance from healthy value to the nearest
    critical threshold (used for per-part health deviation scoring)."""
    span: dict[str, float] = {}
    for p in PARTS[part]["params"]:
        healthy = p["healthy"]
        crit = 0.0
        if "crit_hi" in p:
            crit = max(crit, p["crit_hi"] - healthy)
        if "crit_lo" in p:
            crit = max(crit, healthy - p["crit_lo"])
        span[p["key"]] = max(crit, 1e-9)
    return span


def part_health_index(part: str, values: dict[str, float]) -> float:
    """0-100 health score for one part from its parameter values.

    Raw sensor values are converted to display units via each param's
    ``scale`` before comparing against the (display-unit) healthy
    reference and critical thresholds.
    """
    ref = healthy_reference(part)
    span = threshold_span(part)
    devs = []
    for p in PARTS[part]["params"]:
        key = p["key"]
        v = values.get(key)
        if v is None:
            continue
        v_disp = v * p.get("scale", 1.0)
        dev = abs(v_disp - ref[key]) / span[key]
        devs.append(min(dev, 3.0))
    if not devs:
        return 100.0
    return float(np_clip(100.0 * (1.0 - sum(devs) / len(devs) / 1.0), 0.0, 100.0))


def np_clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x