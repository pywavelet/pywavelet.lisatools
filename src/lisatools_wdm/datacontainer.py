from lisatools import datacontainer
from pywavelet.types import Wavelet, FrequencySeries
import numpy as np
from typing import List, Optional


class DataResidualArray(datacontainer.DataResidualArray):

    def __init__(self,
                 data_res_in: List[np.ndarray] | np.ndarray | datacontainer.DataResidualArray,
                 Nf: int,
                 dt: Optional[float] = None,
                 f_arr: Optional[np.ndarray] = None,
                 df: Optional[float] = None,
                 **kwargs: dict, ):
        super().__init__(data_res_in, dt, f_arr, df, **kwargs)
        self.Nf = Nf

    @property
    def f_grid(self) -> np.ndarray:
        return self.wdm[0].freq

    @property
    def t_grid(self) -> np.ndarray:
        return self.wdm[0].time

    @property
    def wdm(self) -> List[Wavelet]:
        if not hasattr(self, "_wdm"):
            fseries = [
                FrequencySeries(data=self.data_res_arr[i], freq=self.f_arr)
                for i in range(len(self.data_res_arr))
            ]
            self._wdm = [f.to_wavelet(Nf=self.Nf) for f in fseries]
        return self._wdm
