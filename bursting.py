import numpy as np
from scipy.signal import find_peaks
from scipy import io
import seaborn as sns
import os, h5py
from shuffling_functions import signal_partition
from plotting_functions import best_nbins
from utils_loading import get_PTIT_over_days, path_prefix_free
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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


def neuron_ipi(t_series):
    """Calculates the Inter Peak Interval"""
    return np.diff(t_series)


def neuron_fano(sig, W=None, T=100):
    """Calculates the Fano Factor for signal using W random unreplaced samples of length T"""
    nrow, ncol = len(sig) // T, T
    sigs = np.reshape(sig[:nrow * ncol], (nrow, ncol))
    if W is not None:
        if W < 1:
            W = int(len(sig) * W)
        inds = np.arange(nrow)
        np.random.shuffle(inds)
        sigs = sigs[inds[:W]]
    binned = np.sum(sigs, axis=1)
    m = np.mean(binned)
    if m == 0:
        return np.nan
    v = np.var(binned)
    return v / m


def neuron_ispi(sig):
    disc_deconv, _ = find_peaks(sig)
    peaks = np.where(disc_deconv > 0)[0]
    return neuron_ipi(peaks)


def neuron_calcium_ipri(sig, perc=30, ptp=True):
    """ Returns the IPRI in calcium neural signals, if ptp, calculate it based on peak-to-peak,
    else, calculate it based on tail-to-tail
    """
    peak_regions, IPIs = signal_partition(sig, perc)
    if ptp:
        peaks = [p[0] + np.argmax(sig[p[0]:p[1]]) for p in peak_regions]
        return neuron_ipi(peaks)
    else:
        return [len(ipi) for ipi in IPIs]


def neuron_dc_pk_fano(sig, W=None, T=100):
    """Taking in deconvolved signal and calculates the fano"""
    peaks, _ = find_peaks(sig)
    dis_deconv = np.zeros_like(sig)
    dis_deconv[peaks] = sig[peaks]
    return neuron_fano(dis_deconv, W, T)


def neuron_pr_fano(sig, perc=30, W=None, T=100, debug=False):
    """ Returns the IPRI in calcium neural signals, if ptp, calculate it based on peak-to-peak,
    else, calculate it based on tail-to-tail
    """
    peak_regions, IPIs = signal_partition(sig, perc)
    peaks = [p[0] + np.argmax(sig[p[0]:p[1]]) for p in peak_regions]
    sig_prime = np.zeros_like(sig)
    for p in peaks:
        sig_prime[p] = sig[p]
    if debug:
        return neuron_fano(sig_prime, W, T), sig_prime
    else:
        return neuron_fano(sig_prime, W, T)


def neuron_fano_norm(sig, W=None, T=100, lingress=False, pre=True):
    peaks, _ = find_peaks(sig, threshold=1e-08) # use 1E-08 as a threshold to distinguish from 0
    if len(peaks) == 0:
        peaks = [np.argmax(sig)]
        if np.isclose(sig[peaks[0]], 0):
            return 0
    lmaxes = sig[peaks]
    positves = lmaxes[~np.isclose(lmaxes, 0)]
    if lingress:
        A = np.vstack((positves), np.ones_like(positves)).T
    n = np.min(positves)
    if pre:
        return neuron_fano(sig / (n), W, T)
    else:
        return neuron_fano(sig, W, T) / (n)


def calcium_IBI_single_session(inputs, out, window=None, perc=30, ptp=True):
    """Returns a metric matrix and meta data of IBI metric
    Params:
        inputs: str, h5py.File, tuple, or np.ndarray
            if str/h5py.File: string that represents the filename of hdf5 file
            if tuple: (path, animal, day), that describes the file location
            if np.ndarray: array C of calcium traces
        out: str
            Output path for saving the metrics in a hdf5 file
            outfile: h5py.File
                N: number of neurons
                s: number of sliding sessions
                K: maximum number of IBIs extracted
                'mean': N * s matrix, means of IBIs
                'stds': N * s matrix, stds of IBIs
                'CVs': N * s matrix, CVs of IBIs
                'IBIs': N * s * K, IBIs
        window: None or int
            sliding window for calculating IBIs.
            if None, use 'blen' in hdf5 file instead, but inputs have to be str/h5py.File
        perc: float
            hyperparameter for partitioning algorithm, correlated with tail length of splitted calcium trace
        ptp: boolean
            True if IBI is based on peak to peak measurement, otherwise tail to tail

    Alternatively, could store data in:
        mat_ibi: np.ndarray
            N * s * m matrix, , where N is the number of neurons, s is number of sliding sessions,
            m is the number of metrics
        meta: dictionary
            meta data of form {axis: labels}
    """
    if isinstance(inputs, np.ndarray):
        C = inputs
        window = C.shape[1]
        animal, day = None, None
    else:
        if isinstance(inputs, str):
            opts = path_prefix_free(inputs, '/').split('_')
            animal, day = opts[1], opts[2]
            f = h5py.File(inputs, 'r')
        elif isinstance(inputs, h5py.File):
            opts = path_prefix_free(inputs.filename, '/').split('_')
            animal, day = opts[1], opts[2]
            f = inputs
        elif isinstance(inputs, tuple):
            path, animal, day = inputs
            hfile = os.path.join(path, animal, "full_{}_{}__data.hdf5".format(animal, day))
            f = h5py.File(hfile, 'r')
        else:
            raise RuntimeError("Input Format Unknown!")
        C = np.array(f['C'])
        if window is None:
            window = f['blen']
        f.close()
    nsessions = int(np.ceil(C.shape[1] / window))
    rawibis = {}
    maxLen = -1
    for i in range(C.shape[0]):
        rawibis[i] = {}
        for s in range(nsessions):
            slide = C[i, s*window:min(C.shape[1], (s+1) * window)]
            ibis = neuron_calcium_ipri(slide, perc, ptp)
            rawibis[i][s] = ibis
            maxLen = max(len(ibis), maxLen)

    all_ibis = np.full((C.shape[0], nsessions, maxLen), np.nan)
    for i in range(C.shape[0]):
        for s in range(nsessions):
            all_ibis[i][s][:len(rawibis[i][s])] = rawibis[i][s]
    means = np.nanmean(all_ibis, axis=2)
    stds = np.nanstd(all_ibis, axis=2)
    cvs = stds / means
    if animal is None:
        savepath = os.path.join(out, 'sample_IBI.hdf5')
    else:
        savepath = os.path.join(out, 'IBI', animal)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        savepath = os.path.join(savepath, "IBI_{}_{}_perc{}{}_window{}.hdf5"
                                .format(animal, day, perc, '_ptp' if ptp else "", window))
    outfile = h5py.File(savepath, 'w-')
    outfile['mean'], outfile['stds'], outfile['CVs'] = means, stds, cvs
    outfile['IBIs'] = all_ibis
    outfile.close()
    return outfile.filename
    #return np.concatenate([means, stds, cvs], axis=2), {2: ['mean', 'stds', 'CVs']}


def calcium_IBI_all_sessions(folder, window=None, perc=30, ptp=True):
    all_files = get_PTIT_over_days(folder)


def deconv_fano_contrast_single_pair(hIT, hPT, fano_opt='raw', density=True):
    nneg = True
    W = None
    step = 100
    OPT = 'IT VS PT'
    if fano_opt == 'raw':
        fano_metric = neuron_fano
    elif fano_opt == 'norm_pre':
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=True)
    else:
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=False)

    def get_datas(hfile, data_opt, expr_opt):
        redlabels = np.array(hfile['redlabel'])
        datas = np.array(hfile[data_opt])
        blen = hfile.attrs['blen']
        if nneg:
            datas = datas - np.min(datas, axis=1, keepdims=True)
        if expr_opt.find('IT') != -1:
            datas_it = datas[np.logical_and(redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(~redlabels, hfile['nerden'])]
        elif expr_opt.find('PT') != -1:
            datas_it = datas[np.logical_and(~redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(redlabels, hfile['nerden'])]
        else:
            raise RuntimeError('NOT PT OR IT')
        return {'IT': {'N': datas_it.shape[0], 'data': datas_it,
                       'base': datas_it[:, :blen], 'online': datas_it[:, blen:]},
                'PT': {'N': datas_pt.shape[0], 'data': datas_pt,
                       'base': datas_pt[:, :blen], 'online': datas_pt[:, blen:]}}

    def fano_series(all_data, W, step, out=None, label=None):
        datas, datas_base, datas_online, N = all_data['data'], all_data['base'], \
                                             all_data['online'], all_data['N']
        nfanos = np.empty(N)
        base_fanos = np.empty(N)
        online_fanos = np.empty(N)
        for j in range(N):
            fano = fano_metric(datas[j], W, step)
            fano_base = fano_metric(datas_base[j], W, step)
            fano_online = fano_metric(datas_online[j], W, step)
            print(j, fano)
            nfanos[j] = fano
            base_fanos[j] = fano_base
            online_fanos[j] = fano_online
        if out:
            out['nfanos'][label], out['base_fanos'][label], out['online_fanos'][label] = \
                nfanos, base_fanos, online_fanos
        else:
            return out

    datas_IT_expr = get_datas(hIT, 'neuron_act', 'IT')
    datas_PT_expr = get_datas(hPT, 'neuron_act', 'PT')

    def subroutine(W, step):
        vars = ['IT_expr_IT', 'IT_expr_PT', 'PT_expr_IT', 'PT_expr_PT']
        labels = ['nfanos', 'base_fanos', 'online_fanos']
        plot_datas = {'nfanos': {d: None for d in vars},
                      'base_fanos': {d: None for d in vars},
                      'online_fanos': {d: None for d in vars}}
        fano_series(datas_IT_expr['IT'], W, step, plot_datas, 'IT_expr_IT')
        fano_series(datas_IT_expr['PT'], W, step, plot_datas, 'IT_expr_PT')
        fano_series(datas_PT_expr['IT'], W, step, plot_datas, 'PT_expr_IT')
        fano_series(datas_PT_expr['PT'], W, step, plot_datas, 'PT_expr_PT')

        for v in vars:
            ax[0][0].plot(plot_datas['nfanos'][v])
        ax[0][0].legend(vars)
        ax[0][0].set_xlabel("Neuron")
        ax[0][0].set_title("Fano Factor for all neurons")
        all_stats = {l: {v: {} for v in vars} for l in labels}
        all_stats['meta'] = {'W': W if W else -1, 'T': step}
        for i, label in enumerate(labels):
            curr = i+1
            stat = [None] * 12
            r, c = curr // 2, curr % 2
            for j, v in enumerate(vars):
                fanos = plot_datas[label][v]
                # Choice of bin size: Ref: https://www.fmrib.ox.ac.uk/datasets/techrep/tr00mj2/tr00mj2/node24
                # .html
                miu, sigma, N = np.around(np.nanmean(fanos), 5), np.around(np.nanstd(fanos), 5), len(fanos)
                binsize = 3.49 * sigma * N ** (-1/3) # or 2 IQR N ^(-1/3)
                if density:
                    sns.distplot(fanos, bins=int((max(fanos)- min(fanos)) / binsize + 1), kde=True,
                                 norm_hist=True, ax=ax[r][c])
                else:
                    ax[r][c].hist(fanos, bins=int((max(fanos)- min(fanos)) / binsize + 1), density=True,
                              alpha=0.6)
                stat[j], stat[j+4], stat[j+8] = miu, sigma, N
                all_stats[label][v]['mean'] = stat[j]
                all_stats[label][v]['std'] = stat[j+4]
                all_stats[label][v]['N'] = stat[j + 8]
            ax[r][c].legend(vars)
            ax[r][c].set_title(
                "{}, Mean(ITIT, ITPT, PTIT, PTPT): {}|{}|{}|{}\nStd: {}|{}|{}|{}, N: {}|{}|{}|{}"
                               .format(label, *stat), fontsize=10)
        outpath = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/bursty_log"
        io.savemat(os.path.join(outpath, 'fano_{}_stats_{}.mat'.format(fano_opt, all_stats['meta'])),
                   all_stats)
    #plt.style.use('bmh')
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.5)
    fig.suptitle(OPT)
    subroutine(W, step)
    axcolor = 'lightgoldenrodyellow'
    axW = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
    W_slider = Slider(axW, 'Window', valmin=50, valmax=1000, valinit=W if W else -1, valstep=1)
    axstep = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    step_slider = Slider(axstep, 'step', valmin=1, valmax=1000, valinit=step, valstep=1)

    def update(val):
        W, step = int(W_slider.val), int(step_slider.val)
        if W == -1:
            W = None
        for cax in ax.ravel():
            cax.clear()
        subroutine(W, step)
        fig.canvas.draw_idle()

    step_slider.on_changed(update)
    W_slider.on_changed(update)
    plt.show()


def deconv_fano_contrast_avg_days(root, fano_opt='raw', W=None, step=100, eps=True):
    all_files = get_PTIT_over_days(root)
    print(all_files)
    nneg = True
    OPT = 'IT VS PT bursting {}'.format(fano_opt)
    if fano_opt == 'raw':
        fano_metric = neuron_fano
    elif fano_opt == 'norm_pre':
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=True)
    else:
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=False)

    def get_datas(hfile, data_opt, expr_opt):
        redlabels = np.array(hfile['redlabel'])
        datas = np.array(hfile[data_opt])
        blen = hfile.attrs['blen']
        if nneg:
            datas = datas - np.min(datas, axis=1, keepdims=True)
        if expr_opt.find('IT') != -1:
            datas_it = datas[np.logical_and(redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(~redlabels, hfile['nerden'])]
        elif expr_opt.find('PT') != -1:
            datas_it = datas[np.logical_and(~redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(redlabels, hfile['nerden'])]
        else:
            raise RuntimeError('NOT PT OR IT')
        return {'IT': {'N': datas_it.shape[0], 'data': datas_it,
                       'base': datas_it[:, :blen], 'online': datas_it[:, blen:]},
                'PT': {'N': datas_pt.shape[0], 'data': datas_pt,
                       'base': datas_pt[:, :blen], 'online': datas_pt[:, blen:]}}

    def fano_series(all_data, W, step, day=None, out=None, label=None):
        datas, datas_base, datas_online, N = all_data['data'], all_data['base'], \
                                             all_data['online'], all_data['N']
        nfanos = np.empty(N)
        base_fanos = np.empty(N)
        online_fanos = np.empty(N)
        for j in range(N):
            fano = fano_metric(datas[j], W, step)
            fano_base = fano_metric(datas_base[j], W, step)
            fano_online = fano_metric(datas_online[j], W, step)
            print(j, fano)
            nfanos[j] = fano
            base_fanos[j] = fano_base
            online_fanos[j] = fano_online
        if out:
            if day:
                if out['nfanos'][day][label] is None:
                    out['nfanos'][day][label], out['base_fanos'][day][label], \
                    out['online_fanos'][day][label] = nfanos, base_fanos, online_fanos
                else:
                    out['nfanos'][day][label] = np.concatenate((out['nfanos'][day][label], nfanos))
                    out['base_fanos'][day][label] = np.concatenate((out['base_fanos'][day][label], base_fanos))
                    out['online_fanos'][day][label] = np.concatenate((out['online_fanos'][day][label], online_fanos))
            else:
                if out['nfanos'][label] is None:
                    out['nfanos'][label], out['base_fanos'][label], out['online_fanos'][label] = \
                        nfanos, base_fanos, online_fanos
                else:
                    out['nfanos'][label] = np.concatenate((out['nfanos'][label], nfanos))
                    out['base_fanos'][label] = np.concatenate((out['base_fanos'][label], base_fanos))
                    out['online_fanos'][label] = np.concatenate((out['online_fanos'][label], online_fanos))
        else:
            return nfanos, base_fanos, online_fanos

    vars = ['IT_expr_IT', 'IT_expr_PT', 'PT_expr_IT', 'PT_expr_PT']
    labels = ['nfanos', 'base_fanos', 'online_fanos']
    day_range = range(1, max(len(all_files['IT']), len(all_files['PT']))+1)
    plot_datas = {label: {i: {d: None for d in vars} for i in day_range} for label in labels}

    for group in 'IT', 'PT':
        for day in all_files[group]:
            for expr in all_files[group][day]:
                hfile = h5py.File(expr, 'r')
                for celltype in 'IT', 'PT':
                    print(group, day, celltype)
                    data_expr = get_datas(hfile, 'neuron_act', group)
                    var = '{}_expr_{}'.format(group, celltype)
                    fano_series(data_expr[celltype], W, step, day, plot_datas, var)

    plt.style.use('bmh')
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.5)
    fig.suptitle(OPT)
    all_stats = {l: {d: {v: {} for v in vars} for d in day_range} for l in labels}
    all_stats['meta'] = {'W': W if W else -1, 'T': step}
    outpath = "/home/user/bursting/plots/ITPT_contrast"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for day in day_range:
        for v in vars:
            ax[0][0].plot(plot_datas['nfanos'][day][v])
        ax[0][0].legend(vars)
        ax[0][0].set_xlabel("Neuron")
        ax[0][0].set_title("Fano Factor for all neurons")

        for i, label in enumerate(labels):
            curr = i + 1
            stat = ['NA'] * 12
            r, c = curr // 2, curr % 2
            legs = []
            for j, v in enumerate(vars):
                fanos = plot_datas[label][day][v]
                # Choice of bin size: Ref: https://www.fmrib.ox.ac.uk/datasets/techrep/tr00mj2/tr00mj2/node24
                # .html
                if fanos is not None:
                    miu, sigma, N = np.around(np.nanmean(fanos), 5), np.around(np.nanstd(fanos), 5), len(fanos)
                    nbins = best_nbins(fanos)
                    ax[r][c].hist(fanos, bins=int((max(fanos) - min(fanos)) / nbins + 1), density=True, alpha=0.6)
                    stat[j], stat[j + 4], stat[j + 8] = miu, sigma, N
                    all_stats[label][day][v]['mean'] = stat[j]
                    all_stats[label][day][v]['std'] = stat[j + 4]
                    all_stats[label][day][v]['N'] = stat[j + 8]
                legs.append(v)

            ax[r][c].legend(legs)
            ax[r][c].set_title("{}".format(label), fontsize=10)
            fig.savefig(os.path.join(outpath, "d{}_ITPT_contrast_deconvFano_{}_{}_{}.png".format(day, fano_opt, W, step)))
            if eps:
                fig.savefig(os.path.join(outpath,"d{}_ITPT_contrast_deconvFano_{}_{}_{}.eps".format(day, fano_opt, W, step)))
            plt.close('all')
    io.savemat(os.path.join(outpath, 'fano_{}_stats_{}.mat'.format(fano_opt, all_stats['meta'])), all_stats)
    io.savemat(os.path.join(outpath, 'plot_data_fano_{}_{}.mat'.format(fano_opt, all_stats['meta'])), plot_datas)


def burstITPT_contrast_plot(file, fano_opt, W, step, eps=True):
    plot_datas = io.loadmat(file)
    OPT='ITPT_contrast'
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.5)
    fig.suptitle(OPT)
    outpath = "/home/user/bursting/plots/ITPT_contrast"
    vars = ['IT_expr_IT', 'IT_expr_PT', 'PT_expr_IT', 'PT_expr_PT']
    labels = ['nfanos', 'base_fanos', 'online_fanos']
    day_range = range(1, max(len(plot_datas['IT']), len(plot_datas['PT'])) + 1)
    for day in day_range:
        for v in vars:
            ax[0][0].plot(plot_datas['nfanos'][day][v])
        ax[0][0].legend(vars)
        ax[0][0].set_xlabel("Neuron")
        ax[0][0].set_title("Fano Factor for all neurons")

        for i, label in enumerate(labels):
            curr = i + 1
            r, c = curr // 2, curr % 2
            legs = []
            for j, v in enumerate(vars):
                fanos = plot_datas[label][day][v]
                # Choice of bin size: Ref: https://www.fmrib.ox.ac.uk/datasets/techrep/tr00mj2/tr00mj2/node24
                # .html
                if fanos:
                    miu, sigma, N = np.around(np.nanmean(fanos), 5), np.around(np.nanstd(fanos), 5), len(
                        fanos)
                    nbins = min(best_nbins(fanos), 200)
                    sns.distplot(fanos, bins=nbins, kde=True, ax=ax[r][c])
                legs.append(v)
            ax[r][c].legend(legs)
            ax[r][c].set_title("{}".format(label), fontsize=10)
            fig.savefig(
                os.path.join(outpath, "d{}_ITPT_contrast_deconvFano_{}_{}_{}.png".format(day, fano_opt, W, step)))
            if eps:
                fig.savefig(os.path.join(outpath,
                                         "d{}_ITPT_contrast_deconvFano_{}_{}_{}.eps".format(day, fano_opt,
                                                                                            W, step)))


if __name__ == '__main__':
    root = "/home/user/CaBMI_analysis/processed"
    W, T = None, 100
    for opt in 'norm_pre', 'raw', 'norm_post':
        deconv_fano_contrast_avg_days(root, fano_opt=opt, W=W, step=T, eps=True)
