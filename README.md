# Fault-Prediction-in-Vehicles

Physics-informed digital-twin simulator for **predictive maintenance of
military battle tanks**. Every sensor is simulated from the physics-based
equations in
`Physics_Based_Sensor_Equations_Military_Tank_Preventive_Maintenance.docx`
and coupled through an evolving digital-twin state, so that injected
faults degrade the readings in a physically consistent way.  The
resulting labelled datasets feed AI models for anomaly detection, fault
diagnosis and remaining-useful-life (RUL) prediction.

```
Sensors -> Physics-Based Features -> Digital Twin -> AI -> Anomaly / Fault / RUL
```

## Sensor families and governing physics

| # | Sensor | Physics | Key features | Associated fault |
|---|--------|---------|--------------|------------------|
| 1 | Vibration (accelerometer) | `a = d²x/dt²`, `x = A·sin(2πft+φ)` | RMS, kurtosis, FFT peaks, BPFO/BPFI | Bearing / gear wear |
| 2 | Engine/exhaust temperature | `m·cₚ·dT/dt = Q̇_gen − Q̇_cool − Q̇_exh`, RTD `R = R₀(1+αΔT)` | Temperature trend | Overheating / cooling failure |
| 3 | Oil pressure | Hagen–Poiseuille `ΔP = 8μLQ/(πr⁴)`, `μ(T)=A·e^{E/RT}` | Pressure deviation vs. speed/load/temp | Pump / leakage / blockage |
| 4 | Oil debris | Inductive `L ≈ μN²A/l`, `ṅₚ = dNₚ/dt` | Particle count & rate | Accelerated wear |
| 5 | Torque | `P = τω`, `τ = Tr/J`, `ω = 2πN/60` | Torque / power / shear | Load / efficiency degradation |
| 6 | Exhaust | `PV = nRT`, `ṁ = ρAv`, `λ` | EGT, pressure, λ, O₂ | Combustion abnormality |
| 7 | Fluid levels | `C = εA/d`, `V = πr²h` | Level / volume / capacitance | Leakage / depletion |
| 8 | Hydraulics | Pascal `P = F/A`, `P_hyd = PQ` | Pressure–flow relation | Pump / valve / seal fault |
| 9 | Suspension strain | `σ = Eε`, `ΔR/R = GF·ε` | Strain / load | Fatigue / overload |
| 10 | Torsion bar | `θ = TL/JG`, `T = JGθ/L` | Twist / torque history | Torsion-bar fatigue |
| 11 | Shock | `F = ma`, `a_RMS` | RMS / peak g | Terrain / component loading |
| 12 | Acoustic | `p(t)=P₀+A·sin(2πft)`, `SPL = 20·log₁₀(p/p_ref)` | SPL, FFT signatures | Gear / bearing noise |
| 13 | Acoustic emission | Wave equation, event features | Event rate / energy | Crack / impact / friction |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate a labelled dataset with injected faults:

```bash
python main.py --steps 12000 \
    --faults bearing_wear,cooling_failure,oil_pump_degradation \
    --out data/sim_dataset.csv
```

The CSV contains raw sensor readings, physics-derived features
(`vib_rms`, `vib_kurtosis`, `spl_db`, `debris_rate`, ...), a fused
`health_index` (0–100), an extrapolated `rul_steps` estimate, an
`anomaly_score`, and one-hot `fault_*` label columns.

### Available fault profiles

```
bearing_wear gear_wear cooling_failure oil_pump_degradation
bearing_clearance_wear seal_leakage fuel_injector_fault
exhaust_restriction torsion_fatigue hydraulic_valve_fault
structural_crack drivetrain_efficiency_loss
```

### Programmatic use

```python
import numpy as np
from tank_sim.config import TankConfig
from tank_sim.faults import FaultManager
from tank_sim.tank import TankSimulator
from tank_sim.dataset import write_dataset

fm = FaultManager(np.random.default_rng(0))
fm.add("bearing_wear", start_step=3000, ramp_steps=1500)
fm.add("cooling_failure", start_step=6000, ramp_steps=1500)

sim = TankSimulator(TankConfig(), faults=fm, seed=0)
records = sim.run()                 # list of dict records
write_dataset(sim, "data/run.csv")  # CSV with health features + labels
```

## Digital-twin health fusion

Following section 15 of the physics document, the fused health index
combines vibration RMS/kurtosis, oil temperature/pressure, debris rate,
AE activity, structural stress and lambda into a single 0–100 score;
`rul_steps` is obtained by linear extrapolation of the health-index
trajectory to the failure threshold (section 16).

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

Educational / research use for the IDP project.