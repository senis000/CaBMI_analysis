from bursting import neuron_fano, neuron_fano_norm
from caiman.source_extraction.cnmf import deconvolution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os


def deconv_fano_spikefinder(dataset, fano, p=2):
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
    root = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/"
    #fano = 'raw' # Fano Measure Method
    #p = 2 # AR order for foopsi algorithm
    source_name = 'spikefinder'
    for fano, p in list(itertools.product(['norm_pre', 'raw', 'norm_post'], [1, 2])):
        print('opt:', fano, p)
        saveopt = 'deconvFano_p{}_{}_{}'.format(p, fano, source_name)
        outpath = "/Users/albertqu/Documents/7.Research/BMI/plots/bursty"
        dataset = os.path.join(root, source_name)
        measures = deconv_fano_spikefinder(dataset, fano, p)
        visualize_measure(measures, outpath, saveopt)


if __name__ == '__main__':
    test_fano()




