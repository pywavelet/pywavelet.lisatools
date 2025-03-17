from lisatools_wdm.mcmc.eryn_analysis import run_analysis
from lisatools_wdm.mcmc.setup import setup


def test_e2e():
    mcmc_data = setup()
    run_analysis(mcmc_data)
