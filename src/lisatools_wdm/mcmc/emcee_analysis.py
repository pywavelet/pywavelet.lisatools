import emcee
from .setup import setup, wrapper_likelihood, MCMCData
import numpy as np








def run_emcee(mcmc_data:MCMCData, nwalkers=10, nsteps=1000):

    prior_bounds = mcmc_data.prior_bounds
    trues = mcmc_data.true
    fixed_parameters = mcmc_data.fixed_parameters
    analysis = mcmc_data.analysis
    ndim = len(trues)


    def ln_prior(params, fixed_parameters):
        for i, (low, high) in enumerate(prior_bounds):
            if not (low < params[i] < high):
                return -np.inf
        return 0.0

    def ln_prob(params, fixed_parameters, analysis):
        lp = ln_prior(params, fixed_parameters)
        if not np.isfinite(lp):
            return -np.inf
        return lp + wrapper_likelihood(params, fixed_parameters, analysis)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim=ndim,
        log_prob_fn=ln_prob,
        args=(fixed_parameters, analysis),
    )
    mcmc_start = trues + 1e-7 * np.random.randn(nwalkers, ndim)
    sampler.run_mcmc(mcmc_start, nsteps, progress=True)
    return sampler