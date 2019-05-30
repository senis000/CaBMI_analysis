import h5py
import numpy as np


def fft_filter(sig, start, end):
    ftf = np.fft.fft(sig)
    ftred = np.copy(ftf)
    ftred[:start] = 0
    ftred[end:] = 0
    return abs(np.fft.ifft(ftred))


def discretize_deconv(deconv):
    return np.array([deconv[i] if deconv[i] >= max(deconv[max(i - 1, 0)],
                                                   deconv[min(i + 1, len(
                                                       deconv) - 1)]) else 0 for
                     i in range(len(deconv))])


def fake_neuron(burst, dur, p0=0.3):
    """Burst: bursty ratio signifying after the peak how more likely the neuron would keep firing"""
    p1 = min(burst * p0, 0.99)
    fake = np.zeros(dur)
    i = np.random.geometric(p0)
    while i < dur:
        fake[i] = np.random.geometric(1 - p1)
        b = np.random.random()
        if b < p1:
            i += 2
        else:
            i += np.random.geometric(p0) + 2
    return fake


def neuron_ispi(sig):
    """Calculates the Inter Spike Peak Interval"""
    dis_deconv = discretize_deconv(sig)
    peaks = [i for i in range(len(disc_deconv)) if disc_deconv[i] > 0]
    isi = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    return isi


def neuron_dc_fano(sig, W=100, step=1):
    """Taking in deconvolved signal and calculates the fano"""
    dis_deconv = discretize_deconv(sig)
    return neuron_raw_fano(dis_deconv, W, step)


def neuron_raw_fano(sig, W=100, step=1, debug=False):
    dis_deconv = sig
    inds = [i for i in range(0, len(sig) - W + 1, step)]
    ms, ss, fanos = [], [], []
    csum, csquare = 0, 0
    for i in inds:
        if i == 0:
            psig = dis_deconv[i:i + W]
            csum, csquare = np.sum(psig), np.sum(np.square(psig))
            m = csum / W
            s = csquare / W - m ** 2
        else:
            csum = csum - sig[i - 1] + sig[i + W - 1]
            csquare = csquare - sig[i - 1] ** 2 + sig[i + W - 1] ** 2
            m = csum / W
            s = csquare / W - m ** 2
        if debug:
            print(i, m, s)
        ms.append(m)
        ss.append(s)
        if np.isclose(m, 0) and np.isclose(s, 0):
            fanos.append(1)
        else:
            fanos.append(s / (abs(m) + 1e-14))
    return ms, ss, fanos





def neuron_calcium_ipri(sig, perc=50, ptp=False):
    """ Returns the IPRI in calcium neural signals, if ptp, calculate it based on peak-to-peak,
    else, calculate it based on tail-to-tail
    """
    peak_regions, IPIs = signal_partition(sig, perc)
    if ptp:
        peaks = [p[0] + np.argmax(sig[p[0]:p[1]]) for p in peak_regions]
        return [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    else:
        return [len(ipi) for ipi in IPIs]


def neuron_pr_fano(sig, perc=30, W=300, step=1, debug=False):
    """ Returns the IPRI in calcium neural signals, if ptp, calculate it based on peak-to-peak,
    else, calculate it based on tail-to-tail
    """
    peak_regions, IPIs = signal_partition(sig, perc)
    peaks = [p[0] + np.argmax(sig[p[0]:p[1]]) for p in peak_regions]
    sig_prime = np.zeros_like(sig)
    for p in peaks:
        sig_prime[p] = sig[p]
    if debug:
        return neuron_raw_fano(sig_prime, W, step), sig_prime
    else:
        return neuron_raw_fano(sig_prime, W, step)