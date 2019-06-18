from bursting import neuron_fano, neuron_fano_norm
from plotting_functions import best_nbins
from caiman.source_extraction.cnmf import deconvolution
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
plt.style.use('bmh')


def deconv_fano_spikefinder(dataset, fano, p=2, W=None, T=100, binT=1, sample_deconv=True, outpath=None):
    dataset = os.path.join(dataset, '{}')
    if fano == 'raw':
        fano_metric = neuron_fano
    elif fano == 'norm_pre':
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=True)
    else:
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=False)
    measures = {'spike': {}, 'calcium': {}, 'deconv_corr': {}}
    for i in range(1, 11):
        print(i)
        calcium_train = pd.read_csv(dataset.format(i) + '.train.calcium.csv')
        spikes_train = pd.read_csv(dataset.format(i) + '.train.spikes.csv')
        neurons = spikes_train.columns
        measures['deconv_corr'][i] = np.empty(len(neurons))
        for m in measures.keys():
            if m != 'deconv_corr':
                measures[m][i] = {}
                measures[m][i]['neurons'] = neurons
                measures[m][i]['fano'] = np.empty(len(neurons))

        for n in neurons:
            spike, calcium = spikes_train[n], calcium_train[n]
            nonnan = ~np.isnan(spike)
            fano_spike = neuron_fano(np.array(spike[nonnan]), W, T)
            deconv = deconvolution.constrained_foopsi(np.array(calcium[nonnan]), p=p)[5]
            corr = np.corrcoef(deconv, spike[nonnan])[0, 1]
            if outpath:
                fano_record = np.around(fano_spike, 4)
                deconv_ptv = deconv[~np.isclose(deconv, 0)]
                if binT > 1:
                    r, c = len(deconv) // binT, binT
                    r_p, c_p = len(deconv_ptv) // binT, binT
                    deconv_bin = np.sum(deconv[:r * c].reshape((r, c)), axis=1).ravel()
                    deconv_ptv_bin = np.sum(deconv_ptv[:r_p * c_p].reshape((r_p, c_p)), axis=1).ravel()
                else:
                    deconv_bin, deconv_ptv_bin = deconv, deconv_ptv
                bsize1 = best_nbins(deconv_bin)
                bsize2 = best_nbins(deconv_ptv_bin)
                plt.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.5)
                plt.subplot(211)
                plt.hist(deconv_bin, bins=bsize1)
                plt.title('Deconv All')
                plt.subplot(212)
                plt.hist(deconv_ptv_bin, bins=bsize2)
                plt.title('Deconv Positive')
                plt.suptitle('{}_{} #{} Neuron {}, Fano: {}'.format(fano, p, i, n, fano_record))
                savepath = os.path.join(outpath, 'distribution_binT_{}'.format(binT),
                                        "{}_T{}_W{}_p{}".format(fano, T, W, p))
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                plt.savefig(os.path.join(savepath, "spikefinder_{}_neuron{}_fano_{}_corr_{}.png"
                                         .format(i, n, fano_record, np.around(corr, 4))))
                plt.close('all')
                if sample_deconv:
                    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(20, 10))
                    axes[0].plot(spike[:300])
                    axes[0].legend('spike')
                    axes[1].plot(calcium[:300])
                    axes[1].legend('calcium')
                    axes[2].plot(deconv[:300])
                    axes[2].legend('deconv')
                    plt.savefig(savepath + '/signal_{}_neuron{}_corr_{}.png'.format(i, n, np.around(corr, 4)))
                    plt.close('all')
            fano_calcium = fano_metric(deconv, W, T)
            measures['spike'][i]['fano'][int(n)] = fano_spike
            measures['calcium'][i]['fano'][int(n)] = fano_calcium
            measures['deconv_corr'][i][int(n)] = corr
            print(int(n), fano_spike, fano_calcium)
    return measures


def visualize_measure(measures, outpath, saveopt):
    all_spikes = np.concatenate([measures['spike'][i]['fano'] for i in measures['spike']])
    all_calcium = np.concatenate([measures['calcium'][i]['fano'] for i in measures['calcium']])
    idsort = np.argsort(all_spikes)
    sorted_spikes, sorted_calc = all_spikes[idsort], all_calcium[idsort]
    corrR = np.corrcoef(all_spikes, all_calcium)
    corr = corrR[0, 1]
    plt.style.use('bmh')
    plt.plot(sorted_spikes, sorted_calc)
    plt.xlabel('spikes')
    plt.ylabel('calcium')
    plt.title('Fano Corr spike vs calcium {}'.format(corr))
    plt.savefig(os.path.join(outpath, saveopt))
    plt.close()


def test_fano():
    root = "/home/user/bursting/"
    # fano = 'raw' # Fano Measure Method
    # p = 2 # AR order for foopsi algorithm
    source_name = 'spikefinder'
    T = 10
    W = None
    binT = 10
    for fano, p in list(itertools.product(['norm_pre', 'raw', 'norm_post'], [1, 2])):
        print('opt:', fano, p)
        saveopt = 'deconvFano_T{}_p{}_{}_{}'.format(T, p, fano, source_name)
        outpath = "/home/user/bursting/plots"
        dataset = os.path.join(root, source_name)
        measures = deconv_fano_spikefinder(dataset, fano, p, W=W, T=T, binT=binT, outpath=outpath)
        io.savemat(os.path.join(root, 'datalog', saveopt + '.mat'), measures)
        visualize_measure(measures, os.path.join(outpath, "deconvFano_T{}_W{}".format(T, W), saveopt))


if __name__ == '__main__':
    test_fano()
