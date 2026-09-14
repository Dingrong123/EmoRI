"""Radar constants and grids translated from ``config/get_params_value.m``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fftshiftfreqgrid import fftshiftfreqgrid


@dataclass
class Params:
    """MATLAB parameter fields with one-dimensional NumPy coordinate grids.

    Python reserves the word ``lambda``, so use ``params.lambda_``. For
    generic MATLAB-field consumers, ``getattr(params, 'lambda')`` also
    returns the wavelength. Derived fields are built once at creation,
    following the MATLAB function; instantiate a new Params to change
    configuration values and recompute its grids.
    """

    c: float = 299792458.0
    fc: float = 62e9
    Rx: int = 4
    Tx: int = 3
    Fs: float = 1e7
    sweepSlope: float = 66.011e12
    samples: int = 512
    loop: int = 100
    Tc: float = 160e-6
    fft_Rang: int = 512
    fft_Vel: int = 100
    fft_Ang: int = 16
    num_crop: int = 5
    max_value: float = 1e4
    lambda_: float = field(init=False)
    rng_grid: np.ndarray = field(init=False, repr=False)
    agl_grid: np.ndarray = field(init=False, repr=False)
    vel_grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lambda_ = self.c / self.fc
        setattr(self, "lambda", self.lambda_)
        freq_res = self.Fs / self.fft_Rang
        freq_grid = np.arange(self.fft_Rang, dtype=np.float64) * freq_res
        self.rng_grid = freq_grid * self.c / self.sweepSlope / 2
        self.agl_grid = np.arcsin(np.linspace(-1, 1, self.fft_Ang)) * 180 / np.pi
        dop_grid = fftshiftfreqgrid(self.fft_Vel, 1 / (self.Tc * 3))
        self.vel_grid = dop_grid * self.lambda_ / 2


def get_params_value() -> Params:
    """Construct exactly the configuration selected by the MATLAB source."""
    return Params()
