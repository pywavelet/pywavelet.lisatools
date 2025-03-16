from .setup import MCMCData

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import trange




def run_check(mcmc_data:MCMCData, fname="lnl_scan.png"):

    snr = mcmc_data.analysis.snr()
    A = mcmc_data.A
    analysis = mcmc_data.analysis
    default_args = mcmc_data.default_args


    # 1D LnL scan over amplitude
    precision = A / np.sqrt(snr)
    a_range = np.linspace(A - 1.5 * precision, A + 1.5 * precision, 9)
    lnl = np.zeros_like(a_range)
    wdm_lnl = np.zeros_like(a_range)
    for i in trange(len(a_range)):
        args = default_args.copy()
        args[0] = a_range[i]
        lnl[i] = analysis.calculate_signal_likelihood(*args)
        wdm_lnl[i] = analysis.calculate_wdm_likelihood(*args)

    plot_lnl(a_range, lnl, wdm_lnl, A, precision, fname)



def plot_lnl(a_range, lnl, wdm_lnl, A, precision, fname="lnl_scan.png"):
    fig, ax = plt.subplots()
    ax.plot(a_range, lnl, label="Freq", color="b")
    # plot the WDM likelihood
    ax2 = ax.twinx()
    ax2.plot(a_range, wdm_lnl, label="WDM", linestyle="--", color="b")
    ymin, ymax = ax.get_ylim()
    ax.vlines(A, ymin, ymax, color="r", linestyle="--")
    ax.set_xlabel("A")
    ax.set_xlim(a_range.min(), a_range.max())
    ax.axvspan(A - precision, A + precision, alpha=0.5, color="gray")
    ax.set_ylim(bottom=lnl.min())
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="b", linestyle="-", label="Freq"),
            plt.Line2D([0], [0], color="b", linestyle="--", label="WDM"),
        ]
    )
    ax.set_ylabel("Freq LnL")
    ax2.set_ylabel("WDM LnL")
    plt.tight_layout()
    fig.savefig("lnl_scan.png")
    plt.close(fig)

