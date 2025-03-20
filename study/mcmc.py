import os
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from lisatools_wdm.mcmc.emcee_analysis import run_emcee
from lisatools_wdm.mcmc.setup import setup

# get args from command line
def main():
    wdm = "--wdm" in sys.argv
    print("Running for WDM: ", wdm)
    outdir = 'out_mcmc_wdm' if wdm else 'out_mcmc_freq'
    os.makedirs(outdir, exist_ok=True)

    nsteps = 30
    mcmc_data = setup(
        T=0.2,
        sampling_frequency=0.01,
        outdir=outdir,
    )
    sampler = run_emcee(mcmc_data, nwalkers=10, nsteps=nsteps, wdm=wdm)

    burnin = int(0.3 * nsteps)
    samples = sampler.get_chain(flat=True, discard=burnin)
    np.savetxt(f"{outdir}/samples.txt", samples)

    inf_obj = az.from_emcee(sampler, var_names=["lnA", "lnf", "lnfdot"])
    posterior = inf_obj.posterior.sel(draw=slice(burnin, None))
    axes = az.plot_trace(posterior, compact=True)

    trues = mcmc_data.true
    for i, ax_set in enumerate(axes):
        ax_set[0].axvline(trues[i], color="red", zorder=-1)
        ax_set[1].axhline(trues[i], color="red", zorder=-1)
    plt.savefig(f"{outdir}/trace.png")


# load samples.txt

def load_samples():
    samples = np.loadtxt("out_mcmc_wdm/samples.txt")
    print(samples.shape)


if __name__ == "__main__":
    main()