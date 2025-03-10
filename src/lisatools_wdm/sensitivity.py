from lisatools import sensitivity
from pywavelet.types import Wavelet
from pywavelet.utils import evolutionary_psd_from_stationary_psd
from typing import Any, Optional, List

class SensitivityMatrix(sensitivity.SensitivityMatrix):

    @property
    def wdm(self)->Wavelet:
        if not hasattr(self, "_wdm"):
            for s in self.sens_mat:
                fseries = evolutionary_psd_from_stationary_psd(s)
                self._wdm = fseries.to_wavelet(Nf=self.N


        return self._wdm

