
import numpy as np
import matplotlib.pyplot as plt

from fastlisaresponse import ResponseWrapper
from lisatools.sensitivity import A1TDISens, E1TDISens
from lisatools.detector import EqualArmlengthOrbits

from lisatools_wdm.datacontainer import DataResidualArray
from lisatools_wdm.sensitivity import SensitivityMatrix
from lisatools_wdm.analysiscontainer import AnalysisContainer
from tqdm.auto import tqdm, trange

YRSID_SI = 31558149.763545603


class GBWave:
    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0, gaps:bool=False):
        # get the t array
        t = np.arange(0.0, T * YRSID_SI, dt)
        cos2psi = np.cos(2.0 * psi)
        sin2psi = np.sin(2.0 * psi)
        cosiota = np.cos(iota)

        fddot = 11.0 / 3.0 * fdot ** 2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
                2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
                - phi0
        )

        hSp = -np.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -np.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        signal = hp + 1j * hc

        if gaps:
            # ZERO data every 14 days for 7 hours
            gap_mask = np.zeros_like(t)
            gap_mask[::int(14*24*3600/dt)] = 1
            signal *= gap_mask

        return signal



def test_e2e():
    gb = GBWave()
    use_gpu = False

    T = 2.0  # years
    t0 = 100000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

    sampling_frequency = 0.01
    dt = 1 / sampling_frequency

    # order of the langrangian interpolation
    order = 25

    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"

    index_lambda = 6
    index_beta = 7

    tdi_kwargs_esa = dict(
        order=order, tdi=tdi_gen, tdi_chan="AE",
    )

    gb_lisa_esa = ResponseWrapper(
        gb,
        T,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=False,  # set to True if waveform is h+ - ihx
        use_gpu=use_gpu,
        remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
        is_ecliptic_latitude=True,  # False if using polar angle (theta)
        remove_garbage=True,  # removes the beginning of the signal that has bad information
        orbits=EqualArmlengthOrbits(),
        **tdi_kwargs_esa,
    )

    Nf = 64

    # define GB parameters
    A = 1.084702251e-22
    f = 2.35962078e-3
    fdot = 1.47197271e-17
    iota = 1.11820901
    phi0 = 4.91128699
    psi = 2.3290324

    beta = 0.9805742971871619
    lam = 5.22979888
    default_args = [
        A,
        f,
        fdot,
        iota,
        phi0,
        psi,
        lam,
        beta,
    ]

    ae_data = gb_lisa_esa(*default_args)
    data = DataResidualArray(ae_data, dt=dt, Nf=Nf)

    fig, ax = plt.subplots(2, 1, sharex=True)
    for i, lab in enumerate(["A", "E"]):
        ax[i].plot(np.arange(len(ae_data[0])) * dt / YRSID_SI, ae_data[i])
        ax[i].set_ylabel(lab)
    ax[-1].set_xlabel("t [yrs]")
    fig.savefig("tdi_timedomain.png")
    plt.close(fig)

    sens_mat = SensitivityMatrix(
        data.f_arr,
        [A1TDISens, E1TDISens],
        f_grid=data.f_grid,
        t_grid=data.t_grid,
        dt=dt
    )
    analysis = AnalysisContainer(data, sens_mat, signal_gen=gb_lisa_esa, Nf=Nf)

    # plot data
    fig, ax = analysis.loglog()
    snr = analysis.snr()
    for i, lab in enumerate(["A", "E"]):
        ax[i].set_ylabel(f"ASD {lab}")
        ax[i].set_xlabel("f [Hz]")
    fig.suptitle(f"SNR: {snr:.2f}")
    plt.savefig("tdi_freqdomain.png")
    plt.close(fig)


    fig, ax = analysis.plot_wdm()
    fig.savefig("tdi_wdm.png")
    plt.close(fig)

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
