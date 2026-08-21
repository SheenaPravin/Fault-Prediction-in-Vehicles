import math
import unittest

import numpy as np

from vehicle_sim.config import VehicleConfig
from vehicle_sim.faults import FaultManager
from vehicle_sim.physics import (
    AcousticEmissionSensor,
    ExhaustSensor,
    HydraulicSensor,
    LevelSensor,
    OilDebrisSensor,
    OilPressureSensor,
    ThermalSystem,
    TorsionBar,
    TorqueSensor,
    VibrationSensor,
    characteristic_frequencies,
    oil_viscosity,
    pressure_drop,
)
from vehicle_sim.physics.vibration import kurtosis, rms
from vehicle_sim.vehicle import VehicleSimulator


class TestCharacteristicFrequencies(unittest.TestCase):
    def test_shaft_frequency(self):
        cfg = VehicleConfig()
        f = characteristic_frequencies(cfg, 1200.0)
        self.assertAlmostEqual(f["f_r"], 20.0, places=6)
        self.assertAlmostEqual(f["f_gmf"], 360.0, places=6)

    def test_bpfo_less_than_bpfi(self):
        cfg = VehicleConfig()
        f = characteristic_frequencies(cfg, 1500.0)
        self.assertLess(f["BPFO"], f["BPFI"])


class TestVibration(unittest.TestCase):
    def test_sinusoid_rms(self):
        sig = 2.0 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, 10000))
        self.assertAlmostEqual(rms(sig), 2.0 / math.sqrt(2), places=1)

    def test_kurtosis_gaussian_near_three(self):
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(kurtosis(rng.normal(0, 1, 100000)), 3.0, places=1)

    def test_kurtosis_impulsive_high(self):
        rng = np.random.default_rng(0)
        smooth = rng.normal(0, 1, 10000)
        impulsive = np.copy(smooth)
        impulsive[:200] = 20.0
        self.assertGreater(kurtosis(impulsive), kurtosis(smooth))

    def test_bearing_fault_raises_rms_and_kurtosis(self):
        cfg = VehicleConfig()
        sensor = VibrationSensor(cfg, np.random.default_rng(1))
        healthy = sensor.features(1800.0, 0.0, "none")
        faulty = sensor.features(1800.0, 1.0, "bearing_outer")
        self.assertGreater(faulty["vib_rms"], healthy["vib_rms"])
        self.assertGreater(faulty["vib_kurtosis"], healthy["vib_kurtosis"])


class TestThermal(unittest.TestCase):
    def test_load_raises_temperature(self):
        cfg = VehicleConfig()
        low = ThermalSystem(cfg)
        high = ThermalSystem(cfg)
        for _ in range(200):
            low.step(0.1, 1000.0)
            high.step(0.95, 2600.0)
        self.assertGreater(high.T_engine, low.T_engine)

    def test_cooling_failure_heats_up(self):
        cfg = VehicleConfig()
        ok = ThermalSystem(cfg)
        bad = ThermalSystem(cfg)
        for _ in range(300):
            ok.step(0.8, 2000.0)
            bad.step(0.8, 2000.0, cooling_eff=0.3)
        self.assertGreater(bad.T_engine, ok.T_engine)


class TestOil(unittest.TestCase):
    def test_viscosity_drops_with_temperature(self):
        cfg = VehicleConfig()
        cold = oil_viscosity(cfg, 40.0)
        hot = oil_viscosity(cfg, 95.0)
        self.assertGreater(cold, hot)

    def test_pressure_drop_increases_with_flow(self):
        cfg = VehicleConfig()
        mu = oil_viscosity(cfg, 90.0)
        self.assertGreater(
            pressure_drop(cfg, mu, 0.002, cfg.filter_r, cfg.filter_L),
            pressure_drop(cfg, mu, 0.0005, cfg.filter_r, cfg.filter_L))

    def test_pump_fault_lowers_pressure(self):
        cfg = VehicleConfig()
        thermal = ThermalSystem(cfg)
        sensor = OilPressureSensor(cfg, np.random.default_rng(0))
        healthy = sensor.read(thermal, 0.5, pump_eff=1.0)
        degraded = sensor.read(thermal, 0.5, pump_eff=0.5)
        self.assertGreater(healthy["oil_pressure"], degraded["oil_pressure"])

    def test_debris_rate_grows_with_severity(self):
        cfg = VehicleConfig()
        sensor = OilDebrisSensor(cfg, np.random.default_rng(0))
        low = sensor.read(0.0, 1.0)
        high = sensor.read(1.0, 1.0)
        self.assertGreater(high["debris_rate"], low["debris_rate"])


class TestOtherSensors(unittest.TestCase):
    def test_torque_shear_stress_scales(self):
        cfg = VehicleConfig()
        sensor = TorqueSensor(cfg, np.random.default_rng(0))
        a = sensor.read(1500.0, 200000.0)
        b = sensor.read(1500.0, 500000.0)
        self.assertGreater(b["shaft_shear_stress"], a["shaft_shear_stress"])

    def test_lambda_lean_high_o2(self):
        cfg = VehicleConfig()
        thermal = ThermalSystem(cfg)
        sensor = ExhaustSensor(cfg, np.random.default_rng(0))
        lean = sensor.read(thermal, 2000.0, 0.5, fuel_mult=0.7)
        self.assertGreater(lean["lambda"], 1.0)
        self.assertGreater(lean["exhaust_o2_pct"], 0.0)

    def test_fuel_level_decreases(self):
        cfg = VehicleConfig()
        sensor = LevelSensor(cfg, np.random.default_rng(0))
        before = sensor.read(0.9, 1.0)
        for _ in range(50):
            sensor.read(0.9, 1.0)
        after = sensor.read(0.9, 1.0)
        self.assertLess(after["fuel_level"], before["fuel_level"])

    def test_torsion_stiffness_and_fatigue(self):
        cfg = VehicleConfig()
        bar = TorsionBar(cfg)
        a = bar.read(20000.0)
        b = bar.read(20000.0, stiffness_mult=0.6)
        self.assertGreater(b["torsion_twist_deg"], a["torsion_twist_deg"])
        self.assertGreater(b["torsion_cumulative_twist"], 0.0)

    def test_ae_activity_grows(self):
        cfg = VehicleConfig()
        sensor = AcousticEmissionSensor(cfg, np.random.default_rng(0))
        low = sensor.read(0.0, 1.0)
        high = sensor.read(1.0, 1.0)
        self.assertGreater(high["ae_event_rate"], low["ae_event_rate"])


class TestFaultManager(unittest.TestCase):
    def test_severity_ramp(self):
        fm = FaultManager(np.random.default_rng(0))
        fm.add("bearing_wear", start_step=100, ramp_steps=100)
        self.assertEqual(fm.parameters(50)["vib_severity"], 0.0)
        self.assertGreater(fm.parameters(500)["vib_severity"], 0.0)

    def test_parameters_multiplicative_default(self):
        fm = FaultManager(np.random.default_rng(0))
        p = fm.parameters(0)
        self.assertEqual(p["cooling_eff"], 1.0)
        self.assertEqual(p["pump_eff"], 1.0)
        self.assertEqual(p["vib_severity"], 0.0)


class TestVehicleSimulator(unittest.TestCase):
    def test_run_shape_and_labels(self):
        cfg = VehicleConfig()
        fm = FaultManager(np.random.default_rng(0))
        fm.add("bearing_wear", start_step=200, ramp_steps=100)
        sim = VehicleSimulator(cfg, faults=fm, seed=0)
        records = sim.run()
        self.assertGreater(len(records), 1000)
        self.assertIn("fault_bearing_wear", records[0])
        self.assertEqual(records[0]["fault_bearing_wear"], 0.0)
        self.assertEqual(records[-1]["fault_bearing_wear"], 1.0)

    def test_fault_changes_signals(self):
        cfg = VehicleConfig()
        fm = FaultManager(np.random.default_rng(0))
        fm.add("cooling_failure", start_step=100, ramp_steps=100)
        sim = VehicleSimulator(cfg, faults=fm, seed=0)
        records = sim.run()
        temps = [r["coolant_temp"] for r in records]
        self.assertGreater(temps[-1], temps[0])


if __name__ == "__main__":
    unittest.main()