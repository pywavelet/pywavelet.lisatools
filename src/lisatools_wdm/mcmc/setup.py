import numpy as np
import matplotlib.pyplot as plt
import dataclasses
from fastlisaresponse import ResponseWrapper
from lisatools.sensitivity import A1TDISens, E1TDISens
from lisatools.detector import EqualArmlengthOrbits
from lisatools_wdm.datacontainer import DataResidualArray
from lisatools_wdm.sensitivity import SensitivityMatrix
from lisatools_wdm.analysiscontainer import AnalysisContainer
from lisatools_wdm.waveform import GBWave
from lisatools_wdm.plotting import plot_ae_time_domain, plot_ae_freq_domain, plot_signal_on_characteristic_strain
from tqdm.auto import tqdm, trange
import os
from gap_study_utils.gaps import GapWindow

from typing import List, Tuple


def wrapper_likelihood(x, fixed_parameters, analysis:AnalysisContainer, wdm, **kwargs):
    all_parameters = np.zeros(8)
    lna = x[0]
    lnf = x[1]
    lnfdot = x[2]
    all_parameters[:] = np.array([np.exp(lna), np.exp(lnf), np.exp(lnfdot), *fixed_parameters])

    if wdm:
        ll = analysis.calculate_wdm_likelihood(
            *all_parameters,
            source_only=True,
        )
    else:
        ll = analysis.calculate_signal_likelihood(
            *all_parameters,
            source_only=True,
        )
    return ll





@dataclasses.dataclass
class MCMCData:
    gb: GBWave
    T: float
    t0: float
    sampling_frequency: float
    dt: float
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
    prior_bounds: List[Tuple[float, float]]
    true: np.ndarray
    fixed_parameters: List[float]
    gap_object: GapWindow
    outdir: str



def generate_gap_mask(t0: float, T: float, dt: float):
    Tinsec = T * 365.25 * 24 * 3600
    t = np.arange(0, Tinsec+dt, dt)
    gap_mask = np.ones_like(t, dtype=bool)
    gap_duration = 7 * 3600  # 7 hours in seconds
    gap_interval = 14 * 24 * 3600  # 14 days in seconds

    for start in np.arange(0, Tinsec, gap_interval):
        end = start + gap_duration
        gap_mask[(t >= start) & (t < end)] = False

    return gap_mask


def setup(
        use_gpu=False,
        T=2.0,
        t0=100000.0,
        sampling_frequency=0.01,
        order=25,
        tdi_gen="2nd generation",
        Nf=64,
        A=1.084702251e-22,
        f=2.35962078e-3,
        fdot=1.47197271e-17,
        iota=1.11820901,
        phi0=4.91128699,
        psi=2.3290324,
        beta=0.9805742971871619,
        lam=5.22979888,
        use_gaps=False,
        outdir='out_mcmc'
):
    os.makedirs(outdir, exist_ok=True)
    default_args = [A, f, fdot, iota, phi0, psi, lam, beta]
    index_lambda = len(default_args) - 2
    index_beta = len(default_args) - 1
    dt = 1 / sampling_frequency

    tdi_kwargs_esa = {
        "order": order,
        "tdi": tdi_gen,
        "tdi_chan": "AE",
    }

    # if use_gaps:
    #     gap_mask = generate_gap_mask(t0, T, dt)
    # else:
    #     Tinsec = T * 365.25 * 24 * 3600
    #     n = int(Tinsec / dt) + 1
    #     gap_mask = np.ones(n, dtype=bool)

    gb = GBWave(use_gpu)
    char_strain = gb(A, f, fdot, iota, phi0, psi, T=T, dt=dt)
    gb_lisa_esa = ResponseWrapper(
        gb,
        T,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=False,
        use_gpu=use_gpu,
        remove_sky_coords=True,
        is_ecliptic_latitude=True,
        remove_garbage=True,
        orbits=EqualArmlengthOrbits(),
        **tdi_kwargs_esa,
    )

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
    precision = A / np.sqrt(snr)
    a_range = np.array([A - 1.5 * precision, A + 1.5 * precision])
    lnA = np.log(A)
    lnf = np.log(f)
    lnfdot = np.log(fdot)

    prior_bounds = [
        (np.log(a_range[0]), np.log(a_range[1])),
        (lnf - 0.001, lnf + 0.001),
        (lnfdot - 0.001, lnfdot + 0.001),
    ]
    true = np.array([lnA, lnf, lnfdot])
    fixed_parameters = [iota, phi0, psi, lam, beta]

    plot_signal_on_characteristic_strain(char_strain, dt, f"{outdir}/char_strain.png")
    plot_ae_time_domain(ae_data, dt, f"{outdir}/tdi_time_domain.png")
    plot_ae_freq_domain(analysis, f"{outdir}/tdi_freq_domain.png")
    try:
        fig, ax = analysis.plot_wdm()
        fig.savefig(f"{outdir}/tdi_wdm.png")
        plt.close(fig)
    except:
        pass

    return MCMCData(
        gb=gb,
        T=T,
        t0=t0,
        sampling_frequency=sampling_frequency,
        dt=dt,
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
        analysis=analysis,
        prior_bounds=prior_bounds,
        true=true,
        fixed_parameters=fixed_parameters,
        outdir=outdir
    )
