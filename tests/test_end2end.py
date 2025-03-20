from lisatools_wdm.mcmc.lnl_1d_check import run_check
from lisatools_wdm.mcmc.setup import setup


def test_e2e(outdir):
    mcmc_data = setup(outdir=f"{outdir}/basic", T=0.2)
    run_check(mcmc_data, fname=f"{outdir}/basic/lnl_1d_check.png")
    mcmc_data = setup(outdir=f"{outdir}/gaps", use_gaps=True, T=0.2)
    run_check(mcmc_data, fname=f"{outdir}/gaps/lnl_1d_check.png")

