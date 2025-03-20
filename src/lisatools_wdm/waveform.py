try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np

import functools
import warnings





YRSID_SI = 31558149.763545603


class GBWave:
    def __init__(self, use_gpu=False, use_gaps=False):

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.use_gaps = use_gaps


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

        return hp + 1j * hc * self.gap_mask


    def _compute_times(self, T, dt):
        Tsec = T * YRSID_SI
        self._T = T
        self._dt = dt
        self._t = self.xp.arange(0.0, Tsec, dt)
        if self.use_gaps:
            self.gap_mask = generate_gap_mask(self._t, Tsec)
        else:
            self.gap_mask = self.xp.ones_like(self._t, dtype=int)



def generate_gap_mask(t,Tsec):
    gap_mask = np.ones_like(t, dtype=int)
    gap_duration = 7 * 3600  # 7 hours in seconds
    gap_interval = 14 * 24 * 3600  # 14 days in seconds

    for start in np.arange(0, Tsec, gap_interval):
        end = start + gap_duration
        gap_mask[(t >= start) & (t < end)] = 0

    return gap_mask
