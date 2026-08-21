"""Scenario generation: create labelled training data from the physics
simulator, with per-part health indices, piecewise-linear RUL targets
and fault-class labels, plus the demo live stream for the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from tank_sim.config import TankConfig
from tank_sim.faults import FaultManager
from tank_sim.tank import TankSimulator, MissionStep, default_mission

from .parts import (
    FAIL_HEALTH,
    INPUT_FEATURES,
    PART_ORDER,
    RUL_CAP_STEPS,
    part_health_index,
)

ALL_FAULTS = sorted(FaultManager.FAULT_MAP.keys())


@dataclass
class Scenario:
    name: str
    faults: list[tuple[str, float]]  # (fault name, start fraction)
    seed: int
    steps: int = 2400


def scenario_suite() -> list[Scenario]:
    """Healthy scenarios, per-fault scenarios at several onset times and
    severities, and a few multi-fault combinations."""
    suite: list[Scenario] = []
    for i in range(5):
        suite.append(Scenario(f"healthy_{i}", [], seed=100 + i))
    for j, fault in enumerate(ALL_FAULTS):
        for k, frac in enumerate((0.30, 0.40, 0.50, 0.60, 0.70)):
            suite.append(Scenario(f"single_{fault}_{k}",
                                  [(fault, frac)], seed=200 + 10 * j + k))
    suite.append(Scenario("combo_wear_overheat", [
        ("bearing_wear", 0.38), ("cooling_failure", 0.55)], seed=401))
    suite.append(Scenario("combo_wear_hyd", [
        ("bearing_wear", 0.40), ("hydraulic_valve_fault", 0.58)], seed=402))
    suite.append(Scenario("combo_lube_fatigue", [
        ("oil_pump_degradation", 0.42), ("torsion_fatigue", 0.60)], seed=403))
    suite.append(Scenario("combo_lube_seal", [
        ("seal_leakage", 0.45), ("oil_pump_degradation", 0.60)], seed=404))
    return suite


def _mission_for_steps(steps: int, dt: float) -> list[MissionStep]:
    base = default_mission(TankConfig())
    total_duration = steps * dt
    base_duration = sum(m.duration_s for m in base)
    factor = total_duration / base_duration
    return [MissionStep(m.duration_s * factor, m.rpm, m.load, m.terrain)
            for m in base]


def run_scenario(scenario: Scenario, window_samples: int = 512,
                 sample_rate: float = 500.0) -> tuple[list[dict], np.ndarray]:
    """Simulate one scenario; return (records, fault_label_per_step)."""
    cfg = TankConfig()
    cfg.window_samples = window_samples
    cfg.sample_rate = sample_rate
    rng = np.random.default_rng(scenario.seed)
    fm = FaultManager(rng)
    for name, start_frac in scenario.faults:
        fm.add(name, start_step=int(scenario.steps * start_frac),
               ramp_steps=max(int(scenario.steps * 0.22), 50))
    sim = TankSimulator(cfg, faults=fm,
                        mission=_mission_for_steps(scenario.steps, cfg.dt),
                        seed=scenario.seed)
    records = sim.run()[: scenario.steps]
    labels = [sim.faults.active_faults(i)[0] if sim.faults.active_faults(i)
              else "healthy" for i in range(len(records))]
    return records, np.array(labels)


def part_health_series(records: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for part in PART_ORDER:
        out[part] = np.array([part_health_index(part, r) for r in records])
    return out


def rul_labels(health_series: dict[str, np.ndarray]) -> np.ndarray:
    """Piecewise-linear RUL (steps) per part, capped, normalised to [0,1].

    A part that never crosses the failure threshold keeps full RUL (1.0);
    otherwise RUL ramps down to 0 as the failure point approaches.
    """
    n = len(health_series[PART_ORDER[0]])
    out = np.zeros((n, len(PART_ORDER)))
    for j, part in enumerate(PART_ORDER):
        h = health_series[part]
        fail_idx = np.where(h < FAIL_HEALTH)[0]
        if fail_idx.size == 0:
            out[:, j] = 1.0
            continue
        first_fail = int(fail_idx[0])
        rul = np.clip(first_fail - np.arange(n), 0, RUL_CAP_STEPS) / RUL_CAP_STEPS
        out[:, j] = rul
    return out


def feature_matrix(records: list[dict]) -> np.ndarray:
    return np.array([[r[k] for k in INPUT_FEATURES] for r in records], dtype=float)


def build_scaler(features_all: np.ndarray) -> dict:
    scaler: dict[str, dict] = {}
    lo = np.min(features_all, axis=0)
    hi = np.max(features_all, axis=0)
    for k, mn, mx in zip(INPUT_FEATURES, lo, hi):
        scaler[k] = {"min": float(mn), "max": float(mx)}
    return scaler


def normalise(features: np.ndarray, scaler: dict) -> np.ndarray:
    out = np.zeros_like(features)
    for j, k in enumerate(INPUT_FEATURES):
        mn, mx = scaler[k]["min"], scaler[k]["max"]
        out[:, j] = np.clip((features[:, j] - mn) / max(mx - mn, 1e-9), 0, 1)
    return out


def make_windows(features: np.ndarray, rul: np.ndarray, labels: np.ndarray,
                 window: int, stride: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, L = [], [], []
    for i in range(0, len(features) - window + 1, stride):
        X.append(features[i:i + window])
        Y.append(rul[i + window - 1])
        L.append(labels[i + window - 1])
    return np.array(X), np.array(Y), np.array(L)


def build_dataset(window_samples: int = 512, sample_rate: float = 500.0,
                  window: int = 40, stride: int = 6,
                  demo_steps: int = 1800, demo_faults: list[tuple] | None = None,
                  demo_seed: int = 7) -> dict:
    """Generate everything: training arrays, scaler, demo stream."""
    suite = scenario_suite()
    all_feats, all_rul, all_labels, all_records = [], [], [], []
    scaler = None

    for sc in suite:
        records, labels = run_scenario(sc, window_samples, sample_rate)
        feats = feature_matrix(records)
        health = part_health_series(records)
        rul = rul_labels(health)
        if scaler is None:
            scaler = build_scaler(feats)
        feats_n = normalise(feats, scaler)
        X, Y, L = make_windows(feats_n, rul, labels, window, stride)
        all_feats.append(X)
        all_rul.append(Y)
        all_labels.append(L)
        all_records.append(records)

    X = np.concatenate(all_feats)
    Y = np.concatenate(all_rul)
    class_names = ["healthy"] + ALL_FAULTS
    class_index = {name: i for i, name in enumerate(class_names)}
    L = np.array([class_index[lbl] for lbl in np.concatenate(all_labels)])

    per_scenario = [
        {"name": sc.name, "X": x, "Y": y,
         "L": np.array([class_index[lbl] for lbl in l])}
        for sc, x, y, l in zip(suite, all_feats, all_rul, all_labels)
    ]

    # Demo live stream: healthy -> bearing wear -> cooling failure -> fatigue.
    if demo_faults is None:
        demo_faults = [("bearing_wear", 0.30), ("cooling_failure", 0.52),
                       ("torsion_fatigue", 0.74)]
    demo = Scenario("demo_live", demo_faults, seed=demo_seed, steps=demo_steps)
    demo_records, _ = run_scenario(demo, window_samples, sample_rate)
    demo_health = part_health_series(demo_records)

    stream = {
        "records": demo_records,
        "health": {p: demo_health[p].tolist() for p in PART_ORDER},
        "meta": {"name": demo.name, "steps": len(demo_records),
                 "dt": TankConfig().dt,
                 "faults": [f for f, _ in demo_faults]},
    }
    return {
        "X": X, "Y": Y, "labels": L, "scaler": scaler,
        "part_order": PART_ORDER, "stream": stream,
        "class_names": class_names,
        "input_features": INPUT_FEATURES,
        "window": window,
        "per_scenario": per_scenario,
    }


def save_demo(stream: dict, path: str) -> None:
    """Trim the demo records to the fields the dashboard displays."""
    needed = {
        "rpm", "load", "terrain", "step", "time",
        "coolant_temp", "oil_temp", "exhaust_temp", "lambda", "shaft_torque",
        "vib_rms", "vib_kurtosis", "vib_dom_amp", "oil_pressure",
        "oil_viscosity", "debris_rate", "debris_cumulative", "coolant_level",
        "exhaust_pressure", "hyd_pressure", "hyd_leak_flow", "hyd_flow",
        "susp_load_kN", "susp_strain_ue", "shock_a_rms_g",
        "torsion_twist_deg", "torsion_cumulative_twist", "ae_event_rate",
        "ae_energy", "health_index", "spl_db", "fuel_level",
    }
    trimmed = [{k: round(r[k], 4) if isinstance(r[k], float) else r[k]
                for k in needed if k in r} for r in stream["records"]]
    health = {p: [round(v, 2) for v in arr] for p, arr in stream["health"].items()}
    out = {**stream, "records": trimmed, "health": health}
    with open(path, "w") as fh:
        json.dump(out, fh)


def split_indices(n_seq: int, val_frac: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_seq)
    n_val = int(n_seq * val_frac)
    return idx[n_val:], idx[:n_val]