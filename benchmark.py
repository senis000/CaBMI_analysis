from bursting import neuron_fano, neuron_fano_norm
from plotting_functions import best_nbins
from caiman.source_extraction.cnmf import deconvolution
from scipy import stats
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
import itertools
import os


def deconv_fano_spikefinder(dataset, fano, p=2, outpath=None):
    dataset = os.path.join(dataset, '{}')
    W = None
    T = 100
    if fano == 'raw':
        fano_metric = neuron_fano
    elif fano == 'norm_pre':
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=True)
    else:
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=False)
    measures = {'spike': {}, 'calcium': {}}
    for i in range(1, 11):
        print(i)
        calcium_train = pd.read_csv(dataset.format(i) + '.train.calcium.csv')
        spikes_train = pd.read_csv(dataset.format(i) + '.train.spikes.csv')
        neurons = spikes_train.columns
        for m in measures.keys():
            measures[m][i] = {}
            measures[m][i]['neurons'] = neurons
            measures[m][i]['fano'] = np.empty(len(neurons))
        for n in neurons:
            spike, calcium = spikes_train[n], calcium_train[n]
            nonnan = ~np.isnan(spike)
            fano_spike = neuron_fano(np.array(spike[nonnan]), W, T)
            deconv = deconvolution.constrained_foopsi(np.array(calcium[nonnan]), p=p)[0]
            if outpath:
                fano_record = np.around(fano_spike, 4)
                deconv_ptv = deconv[~np.isclose(deconv, 0)]
                bsize1 = best_nbins(deconv)
                bsize2 = best_nbins(deconv_ptv)
                plt.subplot(211)
                plt.hist(deconv, bins=bsize1)
                plt.subplot(212)
                plt.hist(deconv_ptv, bins=bsize2)
                plt.suptitle('{}_{} #{} Neuron {}, Fano: {}'.format(fano, p, i, n, fano_record))
                savepath = os.path.join(outpath, 'distribution', "{}_{}")
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                plt.savefig(os.path.join(savepath, "spikefinder_{}_neuron{}_fano_{}.png"
                    .format(i, n, fano_record)))
                plt.close('all')
                fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(20, 10))
                axes[0].plot(spike[:300])
                plt.legend('spike')
                axes[1].plot(calcium[:300])
                plt.legend('calcium')
                axes[2].plot(deconv[:300])
                plt.legend('deconv')
                plt.savefig(savepath+'/{}_{}.png'.format(i, n))
                plt.close('all')
            fano_calcium = fano_metric(deconv, W, T)
            measures['spike'][i]['fano'][int(n)] = fano_spike
            measures['calcium'][i]['fano'][int(n)] = fano_calcium
            print(n, fano_spike, fano_calcium)
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
    #fano = 'raw' # Fano Measure Method
    #p = 2 # AR order for foopsi algorithm
    source_name = 'spikefinder'
    for fano, p in list(itertools.product(['norm_pre', 'raw', 'norm_post'], [1, 2])):
        print('opt:', fano, p)
        saveopt = 'deconvFano_p{}_{}_{}'.format(p, fano, source_name)
        outpath = "/home/user/bursting/plots"
        dataset = os.path.join(root, source_name)
        measures = deconv_fano_spikefinder(dataset, fano, p, outpath=outpath)
        io.savemat(os.path.join(root, 'datalog', saveopt + '.mat'), measures)
        visualize_measure(measures, outpath, saveopt)


if __name__ == '__main__':
    test_fano()




