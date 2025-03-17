from lisatools_wdm.mcmc.lnl_1d_check import run_check
from lisatools_wdm.mcmc.setup import setup


def test_e2e():
    mcmc_data = setup()
    run_check(mcmc_data)
