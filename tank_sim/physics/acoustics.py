"""Acoustic and acoustic-emission sensors.

Physics
-------
Acoustic pressure wave:

    p(t) = P0 + A sin(2 pi f t)

Sound-pressure level:

    SPL = 20 log10(p_RMS / p_ref),   p_ref = 20 uPa (air)

Frequency-domain analysis (Fourier transform):

    X(f) = integral x(t) e^(-j 2 pi f t) dt

Frequency analysis reveals characteristic signatures of gears, bearings,
pumps, shafts, valves and structural resonances.

Acoustic emission (high-frequency elastic waves from cracks, friction,
impact, material deformation) obeys the wave equation:

    d^2 u / dt^2 = c^2 nabla^2 u

Useful AE features: amplitude, energy, frequency, duration, event rate.
"""

from __future__ import annotations

import numpy as np

from ..config import TankConfig


class AcousticSensor:
    """Microphone measuring airborne mechanical noise (SPL + spectrum)."""

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def pressure(self, rpm: float, n_samples: int | None = None) -> np.ndarray:
        n = n_samples or self.cfg.window_samples
        t = np.arange(n) / self.cfg.sample_rate
        base_spl = self.cfg.acoustic_base_spl + 10 * np.log10(max(rpm / 1000.0, 0.1))
        p_rms = self.cfg.P_REF_AIR * 10 ** (base_spl / 20.0)
        p = p_rms * np.sqrt(2) * (
            np.sin(2 * np.pi * 120.0 * t)
            + 0.4 * np.sin(2 * np.pi * 2.0 * 120.0 * t)
            + 0.2 * np.sin(2 * np.pi * 3.0 * 120.0 * t)
        )
        return p + self.rng.normal(0.0, 0.02 * p_rms, n)

    def spl(self, pressure: np.ndarray) -> float:
        p_rms = float(np.sqrt(np.mean(pressure**2)))
        return 20.0 * np.log10(max(p_rms / self.cfg.P_REF_AIR, 1e-12))

    def features(self, rpm: float) -> dict[str, float]:
        p = self.pressure(rpm)
        spec = np.abs(np.fft.rfft(p))
        freqs = np.fft.rfftfreq(len(p), 1.0 / self.cfg.sample_rate)
        freqs = freqs[1:]
        spec = spec[1:]
        return {
            "spl_db": self.spl(p),
            "acoustic_dom_freq": float(freqs[int(np.argmax(spec))]),
            "acoustic_energy": float(np.sum(p**2)),
        }


class AcousticEmissionSensor:
    """High-frequency AE events associated with crack growth, friction,
    impact and material deformation.

    Event rate and energy rise as structural damage (fatigue cracks)
    progresses.
    """

    def __init__(self, cfg: TankConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.noise_seed)

    def read(self, severity: float, dt: float) -> dict[str, float]:
        cfg = self.cfg
        rate = cfg.ae_event_rate_base * (1.0 + 20.0 * severity)
        events = self.rng.poisson(max(rate * dt, 0.0))
        if events > 0:
            energy = events * (0.5 + 5.0 * severity)
            amp = 40.0 + 40.0 * severity
        else:
            energy = 0.0
            amp = 0.0
        return {
            "ae_event_rate": float(rate),
            "ae_events": float(events),
            "ae_energy": float(energy),
            "ae_amp_dB": float(amp),
            "ae_duration_s": float(events * 0.5e-3),
        }