import os
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from lisatools_wdm.mcmc.emcee_analysis import run_emcee
from lisatools_wdm.mcmc.setup import setup

# get args from command line

if len(sys.argv) > 1:
    wdm = sys.argv[1] == "--wdm"
else:
    wdm = False
print("Running for WDM: ", wdm)
OUTDIR = 'out_mcmc'

if wdm:
    OUTDIR += "_wdm"
else:
    OUTDIR += "_freq"

os.makedirs(OUTDIR, exist_ok=True)

mcmc_data = setup(
    T=0.2,
    sampling_frequency=0.01,
    outdir=OUTDIR,

)
sampler = run_emcee(mcmc_data, nwalkers=10, nsteps=300, wdm=wdm)
samples = sampler.get_chain(flat=True, discard=100)
# save samples to file
np.savetxt(f"{OUTDIR}/samples.txt", samples)

inf_obj = az.from_emcee(sampler, var_names=["lnA", "lnf", "lnfdot"])
# az.to_netcdf(inf_obj, "mcmc_data.nc")
posterior = inf_obj.posterior.stack(draws=("chain", "draw"))

trues = mcmc_data.true

# ditch the burnin
burnin = 100
# inf_obj = inf_obj.sel(draw=slice(burnin, None))
# post = inf_obj.posterior.stack(draws=("chain", "draw"))
post = inf_obj.posterior.sel(draw=slice(burnin, None))
axes = az.plot_trace(post, compact=True, )
for i, ax_set in enumerate(axes):
    ax_set[0].axvline(trues[i], color="red", zorder=-1)
    ax_set[1].axhline(trues[i], color="red", zorder=-1)
plt.savefig(f"{OUTDIR}/trace.png")
