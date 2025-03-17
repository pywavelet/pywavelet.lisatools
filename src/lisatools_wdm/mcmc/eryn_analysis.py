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


def run_eryn(mcmc_data: MCMCData, ntemps=10, nwalkers=10, nsteps=1000):
    prior_bounds = mcmc_data.prior_bounds
    trues = mcmc_data.true
    fixed_parameters = mcmc_data.fixed_parameters
    analysis = mcmc_data.analysis
    ndim = len(trues)

    priors = {
        "gb": ProbDistContainer(
            {i: uniform_dist(*prior_bounds[i]) for i in range(ndim)}),
    }

    ndims = {"gb": 3}
    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        wrapper_likelihood,
        priors,
        args=(fixed_parameters, analysis),
        branch_names=["gb"],
        tempering_kwargs=dict(ntemps=ntemps),
        nleaves_max=dict(gb=1),
    )

    start_params = trues[None, None, None, :] * (
            1 + 1e-7 * np.random.randn(ntemps, nwalkers, 1, trues.shape[0]))
    start_state = State({"gb": start_params})
    sampler.compute_log_prior(start_state.branches_coords)
    sampler.run_mcmc(start_state, nsteps, progress=True)
    return sampler
