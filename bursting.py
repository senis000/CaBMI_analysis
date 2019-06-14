import numpy as np
from scipy.signal import find_peaks
from shuffling_functions import signal_partition


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


def neuron_calcium_ipri(sig, perc=50, ptp=False):
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


def neuron_fano_norm(sig, W=None, T=100, pre=True):
    peaks, _ = find_peaks(sig)
    n = min(peaks, key=lambda p: sig[p])
    if pre:
        return neuron_fano(sig / n, W, T)
    else:
        return neuron_fano(sig, W, T) / n

def raw_deconv_fano_contrast(hIT, hPT):
    norm = True
    W = None
    step = 100
    OPT = 'IT VS PT'


    def get_datas(hfile, data_opt, expr_opt):
        redlabels = np.array(hfile['redlabel'])
        datas = np.array(hfile[data_opt])
        blen = hfile.attrs['blen']
        if norm:
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


    datas_IT_expr = get_datas(hIT, 'neuron_act', 'IT')
    datas_PT_expr = get_datas(hPT, 'neuron_act', 'PT')


    def subroutine(W, step, perc):
        def common(datas, datas_base, datas_online, N):
            nfanos = []
            base_fanos = []
            online_fanos = []
            print(datas.shape)
            for j in range(N):
                fano = neuron_pr_fano(datas[j], perc, W, step)
                fano_base = neuron_pr_fano(datas_base[j], perc, W, step)
                fano_online = neuron_pr_fano(datas_online[j], perc, W, step)
                print(j, fano)
                nfanos.append(fano)
                base_fanos.append(fano_base)
                online_fanos.append(fano_online)
            return nfanos, base_fanos, online_fanos
        vars = ['IT_expr_IT', 'IT_expr_PT', 'PT_expr_IT', 'PT_expr_PT']
        plot_datas = {'nfanos': {d: None for d in vars},
                      'base_fanos': {d: None for d in vars},
                      'online_fanos': {d: None for d in vars}}
        plot_datas['nfanos']['IT_expr_IT'], plot_datas['base_fanos']['IT_expr_IT'], \
        plot_datas['online_fanos']['IT_expr_IT'] = common(datas_IT_expr['IT'],datas_it, datas_base_it,
                                                          datas_online_it, N_it)
        nfanos_pt, base_fanos_pt, online_fanos_pt = common(datas_pt, datas_base_pt, datas_online_pt, N_pt)

        ax[0][0].plot(nfanos_it)
        ax[0][0].plot(nfanos_pt)
        ax[0][0].legend(['IT', 'PT'])
        ax[0][0].set_xlabel("Neuron")
        ax[0][0].set_title("Fano Factor for all neurons")
        ax[0][1].hist(base_fanos_it, bins=100)
        ax[0][1].hist(base_fanos_pt, bins=100)
        ax[0][1].legend(['IT', 'PT'])
        ax[0][1].set_title("baseline FANO, Mean(it, pt): {}|{} Std: {}|{}, N: {}|{}"
                           .format(np.around(np.nanmean(base_fanos_it), 5),
                                   np.around(np.nanmean(base_fanos_pt), 5),
                                   np.around(np.nanstd(base_fanos_it), 2), np.around(np.nanstd(base_fanos_pt), 2),
                                   N_it, N_pt))
        ax[1][0].hist(online_fanos_it, bins=100)
        ax[1][0].hist(online_fanos_pt, bins=100)
        ax[1][0].legend(['IT', 'PT'])
        ax[1][0].set_title("online FANO, Mean(it, pt): {}|{} Std: {}|{}, N: {}|{}"
                           .format(np.around(np.nanmean(online_fanos_it), 5),
                                   np.around(np.nanmean(online_fanos_pt), 5),
                                   np.around(np.nanstd(online_fanos_it), 2),
                                   np.around(np.nanstd(online_fanos_pt), 2), N_it, N_pt))
        ax[1][1].hist(nfanos_it, bins=100)
        ax[1][1].hist(nfanos_pt, bins=100)
        ax[1][1].legend(['IT', 'PT'])
        ax[1][1].set_title("whole_expr FANO, Mean(it, pt): {}|{} Std: {}|{}, N: {}|{}"
                           .format(np.around(np.nanmean(nfanos_it), 5), np.around(np.nanmean(nfanos_pt), 5),
                                   np.around(np.nanstd(nfanos_it), 2), np.around(np.nanstd(nfanos_pt), 2), N_it,
                                   N_pt))


    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.5)
    fig.suptitle(OPT)
    subroutine(W, step, perc)
    axcolor = 'lightgoldenrodyellow'
    axW = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
    W_slider = Slider(axW, 'Window', valmin=50, valmax=1000, valinit=W if W else -1, valstep=1)
    axstep = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
    step_slider = Slider(axstep, 'step', valmin=1, valmax=1000, valinit=step, valstep=1)
    axperc = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    perc_slider = Slider(axperc, 'percentage', valmin=1, valmax=100, valinit=perc, valstep=1)


    def update(val):
        W, step, perc = int(W_slider.val), int(step_slider.val), perc_slider.val
        if W == -1:
            W = None
        for cax in ax.ravel():
            cax.clear()
        subroutine(W, step, perc)
        fig.canvas.draw_idle()


    step_slider.on_changed(update)
    W_slider.on_changed(update)
    perc_slider.on_changed(update)
    plt.show()

