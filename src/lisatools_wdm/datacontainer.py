from lisatools import datacontainer
from pywavelet.types import Wavelet, FrequencySeries


class DataResidualArray(datacontainer.DataResidualArray):

    def __init__(self, Nf: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Nf = Nf

    @property
    def wdm(self) -> Wavelet:
        if not hasattr(self, "_wdm"):
            fseries = FrequencySeries(data=self.data_res_arr, freq=self.f_arr)
            self._wdm = fseries.to_wavelet(Nf=self.Nf)
        return self._wdm

    def plot_wdm(self, *args, **kwargs):
        return self.wdm.plot(*args, **kwargs)
