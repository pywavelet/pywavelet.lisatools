import numpy as np
import matplotlib.pyplot as plt
import dataclasses
from lisatools_wdm.mcmc.setup import setup
from lisatools_wdm.mcmc.emcee_analysis import run_emcee
import arviz as az

mcmc_data = setup(
        T=0.2,
        sampling_frequency=0.01
)
sampler = run_emcee(mcmc_data, nwalkers=10, nsteps=1000)

inf_obj = az.from_emcee(sampler, var_names=['lna', 'lnf', 'lnfdot'])
# save the inference data to a netcdf file
inf_obj.to_netcdf("emcee_inference.nc")

axes = az.plot_trace(inf_obj)
for i, ax_set in enumerate(axes):
    ax_set[0].axvline(mcmc_data.true[i], color="red", zorder=-1)
    ax_set[1].axhline(mcmc_data.true[i], color="red", zorder=-1)