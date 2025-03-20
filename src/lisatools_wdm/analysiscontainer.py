from lisatools import analysiscontainer
from typing import Any, Optional, List
from .datacontainer import DataResidualArray
from .sensitivity import SensitivityMatrix
from pywavelet.types import Wavelet
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm


class AnalysisContainer(analysiscontainer.AnalysisContainer):

    def __init__(
            self,
            data_res_arr: DataResidualArray,
            sens_mat: SensitivityMatrix,
            Nf: int,
            gap_mask: np.ndarray,
            signal_gen: Optional[callable] = None,
    ) -> None:
        super().__init__(data_res_arr, sens_mat, signal_gen)
        self.Nf = Nf
        self.gap_mask = gap_mask # time points [0 for gap, 1 for data]




    def calculate_wdm_likelihood(self,
                                 *args: Any,
                                 source_only: bool = False,
                                 waveform_kwargs: Optional[dict] = {},
                                 data_res_arr_kwargs: Optional[dict] = {},
                                 **kwargs: dict,
                                 ) -> float | complex:
        if data_res_arr_kwargs == {}:
            data_res_arr_kwargs = self.data_res_arr.init_kwargs

        template = DataResidualArray(
            data_res_in=self.signal_gen(*args, **waveform_kwargs),
            Nf=self.Nf,
            **data_res_arr_kwargs
        )

        kwargs = dict(psd=self.sens_mat, **kwargs)
        kwargs["include_psd_info"] = not source_only
        lnl = compute_likelihood(
            data=self.data_res_arr.wdm,
            template=template.wdm,
            psd=self.sens_mat.wdm,
        )
        return lnl

    def plot_wdm(self):
        num_channels = self.data_res_arr.ndim

        # plot data, (template), and PSD
        num_columns = 2
        fig, axes = plt.subplots(num_channels, num_columns, figsize=(7, 3 * num_channels))

        ax_psd_idx = 1

        for i in range(num_channels):
            self.data_res_arr.wdm[i].plot(ax=axes[i, 0], show_colorbar=False)
            psd = self.sens_mat.wdm[i].data
            norm = LogNorm(vmin=psd.min(), vmax=10**-43, clip=True)
            cmap = plt.get_cmap('PuBu')
            cmap.set_under('black')
            self.sens_mat.wdm[i].plot(
                ax=axes[i, ax_psd_idx], show_colorbar=False, norm=norm,
                cmap=cmap
            )


        axes[0, 0].set_title("Data")
        axes[0, ax_psd_idx].set_title("PSD")
        plt.tight_layout()

        return fig, axes


def compute_likelihood(
        data: List[Wavelet], template: List[Wavelet], psd: List[Wavelet], mask: Optional[List[Wavelet]] = None
) -> float:
    lnl = np.zeros(len(data))
    for i in range(len(data)):
        d = data[i].data
        h = template[i].data
        p = psd[i].data

        if mask is not None:
            d, h, p = d[mask[i].data], h[mask[i].data], p[mask[i].data]

        lnl[i] = -0.5 * np.nansum((d - h) ** 2 / p)

    return np.sum(lnl)
