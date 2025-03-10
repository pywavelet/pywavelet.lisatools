from lisatools import analysiscontainer
from typing import Any, Optional


class AnalysisContainer(analysiscontainer.AnalysisContainer):

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
            self.signal_gen(*args, **waveform_kwargs), **data_res_arr_kwargs
        )

        args_2 = (template,)

        if "include_psd_info" in kwargs:
            assert kwargs["include_psd_info"] == (not source_only)
            kwargs.pop("include_psd_info")

        kwargs = dict(psd=self.sens_mat, **kwargs)

        kwargs["include_psd_info"] = not source_only


        d_d = inner_product(
            self.data_res_arr, self.data_res_arr, psd=self.sens_mat, **kwargs_in
        )
        h_h = inner_product(template, template, psd=self.sens_mat, **kwargs_in)
        non_marg_d_h = inner_product(
            self.data_res_arr, template, psd=self.sens_mat, complex=True, **kwargs_in
        )
        d_h = np.abs(non_marg_d_h) if phase_maximize else non_marg_d_h.copy()
        self.non_marg_d_h = non_marg_d_h
        like_out = -1 / 2 * (d_d + h_h - 2 * d_h).real


        return like_out


