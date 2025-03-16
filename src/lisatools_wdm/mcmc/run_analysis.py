from .setup import MCMCData

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import trange
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State


def wrapper_likelihood(x, fixed_parameters, freqs, analysis, **kwargs):
    all_parameters = np.zeros(12)
    mT = x[0]
    q = x[1]
    all_parameters[0] = mT / (1 + q)
    all_parameters[1] = mT * q / (1 + q)
    all_parameters[5] = x[2]
    all_parameters[-1] = x[3]

    all_parameters[np.array([2, 3, 4, 6, 7, 8, 9, 10])] = fixed_parameters

    ll = analysis.calculate_signal_likelihood(
        *all_parameters,
        waveform_kwargs=dict(
            length=1024,
            combine=False,  # TODO: check this
            direct=False,
            fill=True,
            squeeze=True,
            freqs=freqs
        ),
        source_only=True
        # data_arr_kwargs=dict(f_arr=freqs)
    )
    return ll


def run_analysis(mcmc_data:MCMCData):

    snr = mcmc_data.analysis.snr()
    A = mcmc_data.A
    analysis = mcmc_data.analysis
    default_args = mcmc_data.default_args

    priors = {"mbh": ProbDistContainer({
        0: uniform_dist(1e5, 5e6),
        1: uniform_dist(0.05, 0.999999),
        2: uniform_dist(0.0, 2 * np.pi),
        3: uniform_dist(0.0, Tobs + 24 * 3600.0),
    })}

    injection_params = np.array([
        m1 + m2,
        m2 / m1,
        chi1,
        chi2,
        dist,
        phi_ref,
        f_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref
    ])

    fixed_parameters = np.array([
        chi1,
        chi2,
        dist,
        f_ref,
        inc,
        lam,
        beta,
        psi,
    ])

    periodic = {"mbh": {2: 2 * np.pi}}

    ntemps = 10
    nwalkers = 32
    ndims = {"mbh": 4}
    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        wrapper_likelihood,
        priors,
        args=(fixed_parameters, freqs, analysis),
        branch_names=["mbh"],
        tempering_kwargs=dict(ntemps=ntemps),
        nleaves_max=dict(mbh=1),
        periodic=periodic
    )

    injection_params_sub = np.array([m1 + m2, m2 / m1, phi_ref, t_ref])
    start_params = injection_params_sub[None, None, None, :] * (
                1 + 1e-7 * np.random.randn(ntemps, nwalkers, 1, injection_params_sub.shape[0]))
    start_state = State({"mbh": start_params})
    sampler.compute_log_prior(start_state.branches_coords)
    sampler.run_mcmc(start_state, 10, progress=True)