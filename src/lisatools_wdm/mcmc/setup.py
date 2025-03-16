import numpy as np
import matplotlib.pyplot as plt

from fastlisaresponse import ResponseWrapper
from lisatools.sensitivity import A1TDISens, E1TDISens
from lisatools.detector import EqualArmlengthOrbits

from lisatools_wdm.datacontainer import DataResidualArray
from lisatools_wdm.sensitivity import SensitivityMatrix
from lisatools_wdm.analysiscontainer import AnalysisContainer
from tqdm.auto import tqdm, trange

from lisatools_wdm.waveform import GBWave
from lisatools_wdm.plotting import plot_ae_time_domain, plot_ae_freq_domain

import dataclasses


@dataclasses.dataclass
class MCMCData:
    gb: GBWave
    T: float
    t0: float
    sampling_frequency: float
    dt: float
    order: int
    tdi_gen: str
    index_lambda: int
    index_beta: int
    tdi_kwargs_esa: dict
    gb_lisa_esa: ResponseWrapper
    Nf: int
    A: float
    f: float
    fdot: float
    iota: float
    phi0: float
    psi: float
    beta: float
    lam: float
    default_args: list
    data: DataResidualArray
    sens_mat: SensitivityMatrix
    analysis: AnalysisContainer



def setup(use_gpu = False):
    gb = GBWave(use_gpu)

    T = 2.0  # years
    t0 = 100000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

    sampling_frequency = 0.01
    dt = 1 / sampling_frequency

    # order of the langrangian interpolation
    order = 25

    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"

    index_lambda = 6
    index_beta = 7

    tdi_kwargs_esa = dict(
        order=order, tdi=tdi_gen, tdi_chan="AE",
    )

    gb_lisa_esa = ResponseWrapper(
        gb,
        T,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=False,  # set to True if waveform is h+ - ihx
        use_gpu=use_gpu,
        remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
        is_ecliptic_latitude=True,  # False if using polar angle (theta)
        remove_garbage=True,  # removes the beginning of the signal that has bad information
        orbits=EqualArmlengthOrbits(),
        **tdi_kwargs_esa,
    )

    Nf = 64

    # define GB parameters
    A = 1.084702251e-22
    f = 2.35962078e-3
    fdot = 1.47197271e-17
    iota = 1.11820901
    phi0 = 4.91128699
    psi = 2.3290324

    beta = 0.9805742971871619
    lam = 5.22979888
    default_args = [
        A,
        f,
        fdot,
        iota,
        phi0,
        psi,
        lam,
        beta,
    ]

    ae_data = gb_lisa_esa(*default_args)
    data = DataResidualArray(ae_data, dt=dt, Nf=Nf)

    sens_mat = SensitivityMatrix(
        data.f_arr,
        [A1TDISens, E1TDISens],
        f_grid=data.f_grid,
        t_grid=data.t_grid,
        dt=dt
    )
    analysis = AnalysisContainer(data, sens_mat, signal_gen=gb_lisa_esa, Nf=Nf)
    snr = analysis.snr()

    plot_ae_time_domain(ae_data, dt, "tdi_time_domain.png")
    plot_ae_freq_domain(analysis, "tdi_freq_domain.png")
    fig, ax = analysis.plot_wdm()
    fig.savefig("tdi_wdm.png")
    plt.close(fig)

    return MCMCData(
        gb=gb,
        T=T,
        t0=t0,
        sampling_frequency=sampling_frequency,
        dt=dt,
        order=order,
        tdi_gen=tdi_gen,
        index_lambda=index_lambda,
        index_beta=index_beta,
        tdi_kwargs_esa=tdi_kwargs_esa,
        gb_lisa_esa=gb_lisa_esa,
        Nf=Nf,
        A=A,
        f=f,
        fdot=fdot,
        iota=iota,
        phi0=phi0,
        psi=psi,
        beta=beta,
        lam=lam,
        default_args=default_args,
        data=data,
        sens_mat=sens_mat,
        analysis=analysis
    )