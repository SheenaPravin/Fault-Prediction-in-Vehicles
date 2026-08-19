#!/usr/bin/env python3
"""Generate physics-simulated sensor datasets for a battle-tank
predictive-maintenance digital twin.

Example
-------
    python main.py --steps 12000 --faults bearing_wear,cooling_failure \
        --out data/sim_dataset.csv --plot

The physics equations for every sensor are documented in
``Physics_Based_Sensor_Equations_Military_Tank_Preventive_Maintenance.docx``
"""

from __future__ import annotations

import argparse

import numpy as np

from tank_sim.config import TankConfig
from tank_sim.dataset import write_dataset
from tank_sim.faults import FaultManager
from tank_sim.tank import TankSimulator, default_mission

KNOWN_FAULTS = sorted(FaultManager.FAULT_MAP.keys())


def build_mission(total_steps: int, mission: str, dt: float) -> list:
    """Scale a mission profile to the requested number of simulation steps."""
    base = default_mission(TankConfig())
    total_duration = total_steps * dt
    base_duration = sum(m.duration_s for m in base)
    factor = total_duration / base_duration
    return [type(m)(m.duration_s * factor, m.rpm, m.load, m.terrain) for m in base]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics-informed sensor simulator for tank preventive maintenance")
    parser.add_argument("--steps", type=int, default=6000,
                        help="number of simulation time steps (dt=%.2f s)" % TankConfig().dt)
    parser.add_argument("--faults", type=str, default="bearing_wear",
                        help="comma-separated fault names to inject: %s" % ", ".join(KNOWN_FAULTS))
    parser.add_argument("--out", type=str, default="data/sim_dataset.csv",
                        help="output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no-health", action="store_true",
                        help="omit fused health index / RUL / anomaly columns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TankConfig()
    faults = args.faults.split(",") if args.faults else []
    for f in faults:
        if f not in KNOWN_FAULTS:
            raise SystemExit("Unknown fault '%s'. Known: %s" % (f, ", ".join(KNOWN_FAULTS)))

    fm = FaultManager(np.random.default_rng(args.seed))
    for i, f in enumerate(faults):
        start = int(args.steps * (0.25 + 0.2 * i))
        fm.add(f, start_step=start, ramp_steps=max(int(args.steps * 0.2), 10))

    sim = TankSimulator(cfg, faults=fm,
                        mission=build_mission(args.steps, "default", cfg.dt),
                        seed=args.seed)
    path = write_dataset(sim, args.out, add_health=not args.no_health)
    n = args.steps
    print("Wrote %d samples -> %s" % (n, path))
    print("Faults injected: %s" % (", ".join(faults) if faults else "none"))


if __name__ == "__main__":
    main()