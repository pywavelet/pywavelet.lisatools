import numpy as np
import matplotlib.pyplot as plt
from lisatools.sensitivity import get_sensitivity
from scipy.signal import periodogram

from .waveform import YRSID_SI

def plot_ae_time_domain(ae_data, dt, fname="tdi_timedomain.png"):
    fig, ax = plt.subplots(2, 1, sharex=True)
    for i, lab in enumerate(["A", "E"]):
        ax[i].plot(np.arange(len(ae_data[0])) * dt / YRSID_SI, ae_data[i])
        ax[i].set_ylabel(lab)
    ax[-1].set_xlabel("t [yrs]")
    fig.savefig(fname)
    plt.close(fig)


def plot_ae_freq_domain(analysis, fname="tdi_freqdomain.png"):
    fig, ax = analysis.loglog()
    snr = analysis.snr()
    for i, lab in enumerate(["A", "E"]):
        ax[i].set_ylabel(f"ASD {lab}")
        ax[i].set_xlabel("f [Hz]")
    fig.suptitle(f"SNR: {snr:.2f}")
    plt.savefig(fname)
    plt.close(fig)

def plot_signal_on_characteristic_strain(char_strain, dt, fname="char_strain.png"):
    fn = np.logspace(-5, -1, 10000)
    Sn_char_strain = get_sensitivity(fn, sens_fn="LISASens", return_type="char_strain")
    plt.figure()
    plt.loglog(fn, Sn_char_strain)
    f, Pxx = periodogram(char_strain, 1/dt, nfft=2**14)
    plt.loglog(f, np.sqrt(Pxx))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Characteristic Strain")
    plt.ylim(min(Sn_char_strain), max(Sn_char_strain))
    plt.savefig(fname)
    plt.close()
