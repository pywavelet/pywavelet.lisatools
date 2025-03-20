import numpy as np

class Gaps:

    def __init__(self, dt, T):
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T, dt)

    def generate_gap_mask(t, Tsec):
        gap_mask = np.ones_like(t, dtype=int)
        gap_duration = 7 * 3600  # 7 hours in seconds
        gap_interval = 14 * 24 * 3600  # 14 days in seconds

        for start in np.arange(0, Tsec, gap_interval):
            end = start + gap_duration
            gap_mask[(t >= start) & (t < end)] = 0

        return gap_mask