from lisatools import sensitivity
from pywavelet.types import Wavelet
from pywavelet.utils import evolutionary_psd_from_stationary_psd
from typing import Any, Optional, List
import numpy as np


class SensitivityMatrix(sensitivity.SensitivityMatrix):

    def __init__(
        self,
        f: np.ndarray,
        sens_mat: (
            List[List[np.ndarray | sensitivity.Sensitivity]]
            | List[np.ndarray | sensitivity.Sensitivity]
            | np.ndarray
            | sensitivity.Sensitivity
        ),
        f_grid: np.ndarray,
        t_grid: np.ndarray,
        dt: float,
        *sens_args: tuple,
        **sens_kwargs: dict,
    ) -> None:
        super().__init__(f, sens_mat, *sens_args, **sens_kwargs)
        self.f_grid = f_grid
        self.t_grid = t_grid
        self.dt = dt


    @property
    def wdm(self)->List[Wavelet]:


        if not hasattr(self, "_wdm"):
            psds = self.sens_mat
            f = self.frequency_arr

            self._wdm = [
                evolutionary_psd_from_stationary_psd(
                    psd=psd,
                    psd_f=f,
                    f_grid=self.f_grid,
                    t_grid=self.t_grid,
                    dt=self.dt,
                )
                for psd in psds
            ]
        return self._wdm

