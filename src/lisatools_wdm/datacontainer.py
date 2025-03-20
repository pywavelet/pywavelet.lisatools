from lisatools import datacontainer
from pywavelet.types import Wavelet, FrequencySeries
import numpy as np
import functools
from typing import List, Optional


class DataResidualArray(datacontainer.DataResidualArray):

    def __init__(self,
                 data_res_in: List[np.ndarray] | np.ndarray | datacontainer.DataResidualArray,
                 Nf: int,
                 gap_mask: np.ndarray,
                 dt: Optional[float] = None,
                 f_arr: Optional[np.ndarray] = None,
                 df: Optional[float] = None,
                 **kwargs: dict, ):
        super().__init__(data_res_in, dt, f_arr, df, **kwargs)
        self.Nf = Nf
        self.gap_mask = gap_mask # time points [0 for gap, 1 for data]

    @property
    def f_grid(self) -> np.ndarray:
        return self.wdm[0].freq

    @property
    def t_grid(self) -> np.ndarray:
        return self.wdm[0].time

    @functools.cached_property
    def gap_wdm(self) -> np.ndarray:

        if not hasattr(self, "_gap_wdm"):
            nt = len(self.gap_mask)
            t = np.arange(0, nt * self.dt, self.dt)

            Dt = self.t_grid[1] - self.t_grid[0]
            Nt = len(self.t_grid)
            gap_wdm_time = np.zeros(Nt, dtype=int)
            for i, start_time in enumerate(gap_wdm_time):
                # Define the current interval from start_time to start_time+dt2
                indices = np.where((t >= start_time) & (t < start_time + Dt))[0]
                if indices.size > 0:
                    gap_wdm_time[i] = 1 if np.any(self.gap_mask[indices] == 1) else 0

            _gap_wdm = self.wdm.data.copy()
            _gap_wdm[:, gap_wdm_time == 0] = 0
            _gap_wdm[:, gap_wdm_time == 1] = 1
            self._gap_wdm = _gap_wdm
        return self._gap_wdm



    @property
    def wdm(self) -> List[Wavelet]:
        if not hasattr(self, "_wdm"):
            fseries = [
                FrequencySeries(data=self.data_res_arr[i], freq=self.f_arr)
                for i in range(len(self.data_res_arr))
            ]
            self._wdm = [f.to_wavelet(Nf=self.Nf) for f in fseries]

            self.

        return self._wdm

