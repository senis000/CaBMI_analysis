import numpy as np
import matplotlib.pyplot as plt
import os, h5py
from shuffling_functions import shuffle_peaks_1d


def test_zero_cross():
    pass


def visualize_percentage_gradient(a, p):
    pass


def load_data():
    dp = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/processed"
    dfile = os.path.join(dp, 'full_IT3_181004__data.hdf5')
    with h5py.File(dfile, 'r') as hf:
        C_copy = np.array(hf['C'])[hf['nerden']]
    return C_copy


def cross_val(n, a, i):
    """Return True if n belongs to closed set [a[i-1], a[i]] (or [a[i],
    a[i-1]]),
    finds value
    crossing"""
    return (a[i] - n) * (n-a[i-1]) >= 0


def demo(i, perc=50, grad_view=True, start=0, end=None):
    Cs = load_data()
    data = Cs[i, :][start:end]
    shuffled, peaks = shuffle_peaks_1d(data, lambda a: a, debug=True)
    plt.subplot(211)
    if grad_view:
        grads = np.gradient(data)
        negs_thres = np.percentile(grads[grads<0], perc)
        oned = np.std(np.gradient(data))
        plt.plot(np.vstack((data, grads)).T)
        plt.plot(np.ones_like(data, oned), color='red')
        conds = (np.logical_and(grads <= 0, grads > negs_thres))
        plt.scatter(np.where(conds)[0], grads[conds], s=3, color='magenta')
    else:
        plt.plot(data)
    others = np.delete(np.arange(len(data)), peaks)
    plt.scatter(others, data[others], s=3, color='green')
    plt.scatter(peaks, data[peaks], s=3, color='red')
    plt.subplot(212)
    if grad_view:
        plt.plot(np.vstack((shuffled, np.gradient(shuffled))).T)
    else:
        plt.plot(shuffled)
    plt.suptitle("Correlation coefficient: {}".format(np.corrcoef(shuffled,
                                                               data)[0, 1]))
    plt.show()


def get_corr_coeff_shuffle(Cs, visual=True):
    corrs = []
    for i in range(Cs.shape[0]):
        data = Cs[i]
        shuffled, peaks = shuffle_peaks_1d(data, lambda a: a, debug=True)
        corrs.append(np.corrcoef(shuffled, data)[0, 1])
    if visual:
        plt.plot(corrs)
        plt.suptitle("Mean Linear {}, Mean Squared {}".format(np.mean(corrs),
                                        np.mean(np.square(corrs))))