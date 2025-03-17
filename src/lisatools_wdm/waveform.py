try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np

import functools
import warnings





YRSID_SI = 31558149.763545603


class GBWave:
    def __init__(self, use_gpu=False):

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np


    def __call__(self, A, f, fdot, iota, phi0, psi, T, dt):
        # get the t array


        if not hasattr(self, "_t"):
            self._compute_times(T, dt)
        if T != self._T:
            warnings.warn("T has changed, recomputing t array")
            self._compute_times(T, dt)

        t = self._t
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot ** 2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc


    def _compute_times(self, T, dt):
        self._T = T
        self._dt = dt
        self._t = self.xp.arange(0.0, T * YRSID_SI, dt)