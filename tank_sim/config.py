"""Default parameters of the simulated battle-tank digital twin.

All values are representative for a heavy tracked military vehicle
(e.g. a main battle tank) and are used by the physics modules to turn
raw physical quantities into realistic sensor readings.
"""

from dataclasses import dataclass, field


@dataclass
class TankConfig:
    # --- Universal constants ----------------------------------------------------
    R_UNIVERSAL: float = 8.314          # J/(mol K)
    P_REF_AIR: float = 20e-6            # Pa, acoustic reference pressure
    EPSILON_0: float = 8.854e-12        # F/m, vacuum permittivity

    # --- Engine -----------------------------------------------------------------
    engine_displacement: float = 28.0      # L (diesel V12)
    max_speed_rpm: float = 2600.0          # RPM
    idle_speed_rpm: float = 800.0          # RPM
    engine_mass_thermal: float = 120.0     # kg (effective thermal mass)
    c_p_coolant: float = 4180.0            # J/(kg K) water-glycol coolant
    max_fuel_energy_rate: float = 2.5e6    # J/s full-throttle heat release
    coolant_power: float = 900.0           # J/s per K of coolant delta-T

    # --- Vibration --------------------------------------------------------------
    n_bearings: int = 12                   # rolling elements per bearing
    bearing_pitch_d: float = 0.130         # m
    ball_d: float = 0.024                  # m
    contact_angle: float = 15.0            # deg
    drive_pinion_teeth: int = 18           # Z_p gear-mesh
    shaft_amp_base: float = 0.6            # m/s^2 base sinusoidal amplitude
    vibration_noise: float = 0.05          # m/s^2 sensor noise sigma

    # --- Lubrication oil --------------------------------------------------------
    oil_density: float = 860.0             # kg/m^3
    oil_A: float = 1.6e-6                  # Pa*s Arrhenius pre-exponential
    oil_E: float = 26000.0                 # J/mol activation energy
    oil_ref_temp: float = 60.0             # deg C reference for nominal mu
    filter_r: float = 0.004                # m filter passage radius
    filter_L: float = 0.06                 # m filter passage length
    main_gallery_r: float = 0.008          # m main oil gallery radius
    main_gallery_L: float = 0.5            # m main oil gallery length
    pump_discharge_pressure: float = 5.5e5 # Pa nominal pump head
    oil_pressure_noise: float = 0.02e5     # Pa

    # --- Oil debris -------------------------------------------------------------
    debris_gain: float = 1.0               # particles per second baseline
    debris_noise: float = 0.2

    # --- Torque / drivetrain ----------------------------------------------------
    gear_ratio: float = 12.0               # overall drivetrain ratio
    drive_r: float = 0.4                   # m sprocket / track drive radius
    shaft_radius: float = 0.06             # m
    shaft_J: float = 6.5e-6                # m^4 polar moment of inertia
    shaft_shear_modulus: float = 79e9      # Pa steel G
    torque_efficiency: float = 0.92        # nominal drivetrain efficiency
    torque_noise: float = 15.0             # N*m

    # --- Exhaust ----------------------------------------------------------------
    exhaust_area: float = 0.03             # m^2
    exhaust_Rs: float = 287.0              # J/(kg K) for air-combustion mix
    stoich_afr: float = 14.7               # diesel stoichiometric A/F ratio
    lambda_noise: float = 0.01
    exhaust_pressure_base: float = 1.25e5  # Pa

    # --- Fuel / fluid levels ----------------------------------------------------
    fuel_tank_r: float = 0.30              # m
    fuel_tank_h: float = 0.90              # m
    fuel_permittivity: float = 2.1         # relative dielectric of diesel
    oil_sump_h: float = 0.35               # m oil level equivalent height
    coolant_h: float = 0.40                # m coolant expansion tank height
    fuel_burn_rate: float = 1.6e-4         # m^3/s full-load fuel burn

    # --- Hydraulics -------------------------------------------------------------
    hyd_pump_pressure: float = 2.1e7       # Pa (turret/stabilizer circuit)
    hyd_valve_area: float = 2.0e-4         # m^2
    hyd_flow_noise: float = 0.005          # m^3/s
    hyd_leak_area: float = 1.0e-7          # m^2 seal-leak equivalent area

    # --- Suspension / structure -------------------------------------------------
    suspension_E: float = 200e9            # Pa
    torsion_L: float = 1.8                 # m torsion bar length
    torsion_r: float = 0.03                # m torsion bar radius
    torsion_G: float = 79e9                # Pa
    strain_gauge_GF: float = 2.1           # gauge factor
    strain_gauge_R: float = 350.0          # Ohm nominal resistance
    suspension_area: float = 0.004         # m^2 effective loaded section
    vehicle_mass: float = 58000.0          # kg
    suspension_noise: float = 0.5          # kN

    # --- Acoustic ---------------------------------------------------------------
    acoustic_base_spl: float = 105.0       # dB baseline SPL
    acoustic_noise: float = 0.4            # dB
    ae_event_rate_base: float = 2.0        # events/s baseline
    ae_noise: float = 0.3

    # --- Control / sampling -----------------------------------------------------
    dt: float = 0.05                       # s digital-twin time step
    sample_rate: float = 1000.0            # Hz for high-frequency bursts
    window_samples: int = 2048             # samples per feature window
    noise_seed: int = 42