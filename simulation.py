#! /usr/bin/env python

# Copyright 2012, Olav Stetter
#
# This file is part of TE-Causality.
#
# TE-Causality is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TE-Causality is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TE-Causality.  If not, see <http://www.gnu.org/licenses/>.

# NEST simulator designed to iterate over a number of input topologies
# (YAML) and to adjst the internal synaptic weight to always achieve an
# equal bursting rate across networks.

# adapated for Calcium BMI project. Albert Qu

# system
import sys, os
import time
import json, pickle
import h5py
import re
import subprocess
# data
import numpy as np
import scipy
from scipy.sparse import linalg as sp_linalg
from scipy.sparse import diags as spdiags
from sklearn.metrics import roc_curve, auc
import random
import pandas as pd
# plotting
import matplotlib.pyplot as plt
# utils
from analysis_functions import calcium_dff, statsmodel_granger, granger_select_order
from utils_gte import run_gte, create_gte_input_files
from utils_cabmi import ProgressBar
from utils_loading import path_prefix_free
from ExpGTE import fc_te_caulsaity
# simulation
try:
    import nest
except ModuleNotFoundError:
    print('NEST package not installed, some functions are unusable')


"""-------------------------------------------------
-------- network topology & data management --------
----------------------------------------------------"""


# TODO: create topologies with positions of neuron, with connection probability related to distance
def create_jsons(size):
    # TODO add creation time
    jsonobj = {'size': size, 'nodes': [None] * size}
    for i in range(size):
        jsonobj['nodes'][i] = {'id': i}


def load_connectivity_from_network(netjson):
    with open(netjson, 'r') as jf:
        jobj = json.load(jf)
    s = jobj['size']
    cmatrix = np.zeros((s, s))
    nodes = jobj['nodes']
    for i in range(s):
        cmatrix[i, nodes[i]['connectedTo']] = 1
    return cmatrix


def spike_pairs_to_hdf5(folder, rm=False):
    removes = []
    for f in os.listdir(folder):
        m = re.search(r"s_index_(\w+).dat", f)
        if m:
            f1 = os.path.join(folder, f)
            f2 = os.path.join(folder, f"s_times_{m.group(1)}.dat")
            removes.append(f1)
            removes.append(f2)
            s_index = np.loadtxt(f1, dtype=np.int)
            s_times = np.loadtxt(f2, dtype=np.float)
            with h5py.File(os.path.join(folder, f'sim_spike_{m.group(1)}.hdf5'), 'w-') as hf:
                hf.create_dataset('neuron', data=s_index)
                hf.create_dataset('spike', data=s_times)
    if rm:
        for r in removes:
            os.remove(r)


def get_sim_files(simulation, inet, keywords=None, ntype='exc'):
    # returns network, spike, calcium files
    calciums = os.path.join(simulation, 'calcium')
    if keywords is None:
        for f in os.listdir(calciums):
            m = re.search(fr"(\w+)_net(\d+)_(\w+).(\w+)", f)
            if m and 'net'+m.group(2) == inet:
                keywords = m.group(3)
                break
        raise RuntimeError(f"Can't find {inet} in {calciums}")

    identifier = f"{inet}_{keywords}"
    spike = os.path.join(simulation, 'spikes', f'sim_spike_{identifier}.hdf5')
    calcium = os.path.join(calciums, f'calcium_{identifier}.hdf5')
    network = os.path.join(simulation, 'networks', f'sim_{ntype}_{identifier}.json')
    assert os.path.exists(spike), f"Can't find file {spike}"
    assert os.path.exists(calcium), f"Can't find file {calcium}"
    assert os.path.exists(network), f"Can't find file {network}"
    return network, spike, calcium


def regularize_simulation_name_codes(simulation):
    # Loop throw network folder and change filenames in calcium & spike folder so that their keywords match
    network = os.path.join(simulation, 'networks')

    def change_names(folder, opt, inet, keywords, change_original=None):
        netw = os.path.join(folder, 'networks')
        dest = os.path.join(folder, opt)
        for fi in os.listdir(dest):
            m = re.search(fr"(\w+)_{inet}_(\w+).(\w+)", fi)
            if m:
                pk = m.group(2).split("_")[1]
                identifier = f"{inet}_{keywords}_{pk}"
                newname = os.path.join(dest, f"{m.group(1)}_{identifier}.{m.group(3)}")
                if change_original is not None:
                    prefix, ftype = change_original
                    noriginal = os.path.join(netw, f'{prefix}_{identifier}{ftype}')
                    original = os.path.join(netw, f'{prefix}_{inet}_{keywords}{ftype}')
                    if os.path.exists(original):
                        # print(original, noriginal)
                        os.rename(original, noriginal)

                os.rename(os.path.join(dest, fi), newname)
                # print(fi, newname, m.group(2))
    for f in os.listdir(network):
        if f[-5:] == '.json':
            fparts = f[:-5].split("_")
            inet = fparts[2]
            keywords = f"{fparts[3]}_{fparts[4]}"
            change_names(simulation, 'spikes', inet, keywords, (f"{fparts[0]}_{fparts[1]}", f[-5:]))
            change_names(simulation, 'calcium', inet, keywords)


"""-------------------------------------------------
--------------- calcium simulation -----------------
----------------------------------------------------"""
# def spike_to_calcium_C():
# #     exe_code = subprocess.call([
# #         f"./te-causality/transferentropy-sim/{method}", control_file_name
# #     ])


class SpikeCalciumizer:

    MODELS = ['Leogang']
    tauImg = 100 #ms;
    fluorescence_model = "Leogang"
    std_noise = 0.03 # percentage of the saturation level
    fluorescence_saturation = 300.
    cutoff = 1000.
    DeltaCalciumOnAP = 50. #uM
    tauCa = 400. #ms
    ALIGN_TO_FIRST_SPIKE = True

    def __init__(self, params=None):
        if params is not None:
            for p in params:
                if hasattr(self, p):
                    setattr(self, p, params[p])
                else:
                    raise RuntimeError(f'Unknown Parameter: {p}')
        assert self.fluorescence_model in self.MODELS

    # TODO: potentially offset the time signature such that file is aligned with the first spike
    def apply_transform(self, spikes, size=None, sample=None):
        # spikes: pd.DataFrame
        times, neurons = spikes['spike'].values, spikes['neuron'].values
        if self.ALIGN_TO_FIRST_SPIKE:
            times = times - np.min(times) # alignment to 1st spike
        if size is None:
            size = int(np.max(neurons)) + 1
        if sample is None:
            # only keep up to largest multiples of tauImg
            t_end = np.max(times)
        else:
            t_end = sample * self.tauImg
        time_bins = np.arange(0, t_end+1, self.tauImg)
        all_neuron_acts = np.empty((size, len(time_bins) - 1))
        for i in range(size):
            neuron = neurons == i
            all_neuron_acts[i] = np.histogram(times[neuron], time_bins)[0]
        return self.binned_spikes_to_calcium(all_neuron_acts)

    def apply_tranform_from_file(self, *args, sample=None): #TODO: add #neurons to simulated spike,
        # last item possibly
        # args: (index, time) or one single hdf5 file
        if len(args) == 2:
            fneurons, ftimes = args
            assert ftimes[-4:] == '.dat' and fneurons[-4:] == '.dat' \
                   and 'times' in ftimes and 'index' in fneurons
            s_index = np.loadtxt(fneurons, dtype=np.int)
            s_times = np.loadtxt(ftimes, dtype=np.float)
            spikes = pd.DataFrame({'spike': s_times, 'neuron': s_index})
        elif len(args) == 1:
            fspike = args[0]
            assert fspike[-5:] == '.hdf5'
            with h5py.File(fspike, 'r') as hf:
                spikes = pd.DataFrame({'spike': hf['spike'], 'neuron': hf['neuron']})
        else:
            raise RuntimeError("Bad Arguments")
        return self.apply_transform(spikes, sample=sample)

    def binned_spikes_to_calcium(self, neuron_acts, fast_inverse=False):
        """
        :param neuron_acts: np.ndarray N x T (neuron x samples)
        :param fast_inverse: whether to use fast reverse. two methods return the same values
        :return:
        """
        # TODO; determine how many spikes were in the first bin
        calcium = np.zeros_like(neuron_acts)
        T = neuron_acts.shape[-1]
        gamma = 1-self.tauImg/self.tauCa
        fluor_gain = self.DeltaCalciumOnAP * neuron_acts
        if self.fluorescence_model == 'Leogang':
            if fast_inverse:
                G = spdiags([np.ones(T), np.full(T, -gamma)], [0, -1], format='csc')
                calcium = fluor_gain @ sp_linalg.inv(G.T)
            else:
                calcium[:, 0] = fluor_gain[:, 0]
                for t in range(1, T):
                    calcium[:, t] = gamma * calcium[:, t-1] + fluor_gain[:, t]
        else:
            raise NotImplementedError(f"Unidentified Model {self.fluorescence_model}")
        if self.fluorescence_saturation > 0:
            calcium = self.fluorescence_saturation * calcium / (calcium + self.fluorescence_saturation)
        if self.std_noise:
            calcium += np.random.normal(0, self.std_noise * self.fluorescence_saturation, calcium.shape)
        return calcium

    def loop_test(self, length, iterations=1000, fast_inv=False):
        # Run time tests of simulation algorithms
        times = [None] * iterations
        N = 10
        for j in range(iterations):
            t0 = time.time()
            rs = np.random.randint(0, 30, (N, length))
            # rs = np.random.random(length)
            self.binned_spikes_to_calcium(rs, fast_inv)
            times[j] = time.time() - t0
        return times


def generate_calcium_data_from_spikes(folder, out):
    # TODO: save img rate!
    if not os.path.exists(out):
        os.makedirs(out)
    networks = set()
    s_calc = SpikeCalciumizer()
    for f in os.listdir(folder):
        m = re.search(r"(\w+)_net(\d+)_(\w+).(\w+)", f)
        if m:
            ind = m.group(1)
            n = int(m.group(2))
            opts = m.group(3)
            ftype = m.group(4)
            if n not in networks and (ftype == 'dat' or ftype == 'hdf5'):
                print(networks, 'processing', n)
                networks.add(n)
                outname = os.path.join(out, f"calcium_net{n}_{opts}.hdf5")
                if ftype == 'hdf5':
                    calcium = s_calc.apply_tranform_from_file(os.path.join(folder, f))
                elif ftype == 'dat':
                    s_index = os.path.join(folder, f"s_index_net{n}_{opts}.dat")
                    s_times = os.path.join(folder, f"s_times_net{n}_{opts}.dat")
                    calcium = s_calc.apply_tranform_from_file(s_index, s_times)
                elif ftype == 'txt':
                    continue
                else:
                    raise NotImplementedError(f"Unknown file type {ftype}")
                xs = np.arange(calcium.shape[-1])
                dff = np.empty_like(calcium)
                for i in range(calcium.shape[0]):
                    dff[i] = calcium_dff(xs, calcium[i])

                with h5py.File(outname, 'w-') as hf:
                    hf.create_dataset('calcium', data=calcium)
                    hf.create_dataset('dff', data=dff)


"""-------------------------------------------------
---------- network statistics calculation ----------
----------------------------------------------------"""


def determine_burst_rate(xindex, xtimes, tauMS, total_timeMS, size):
    # this code was directly translated from te-datainit.cpp
    burst_treshold = 0.4
    assert (len(xindex) == len(xtimes))
    if len(xindex) < 1:
        print("-> no spikes recorded!")
        return 0.
    # print "DEBUG: spike times ranging from "+str(xtimes[0])+" to "+str(xtimes[-1])
    print("-> " + str(len(xtimes)) + " spikes from " + str(len(np.unique(xindex))) + " of " + str(
        size) + " possible cells recorded.")
    print("-> single cell spike rate: " + str(
        1000. * float(len(xtimes)) / (float(total_timeMS) * float(size))) + " Hz")
    samples = int(xtimes[-1] / float(tauMS))
    # 1.) generate HowManyAreActive-signal (code directly translated from te-datainit.cpp)
    startindex = -1
    endindex = 0
    tinybit_spikenumber = -1
    HowManyAreActive = []
    for s in range(samples):
        ttExactMS = s * tauMS
        HowManyAreActiveNow = 0
        while (endindex + 1 < len(xtimes) and xtimes[endindex + 1] <= ttExactMS + tauMS):
            endindex += 1
        HowManyAreActiveNow = len(np.unique(xindex[max(0, startindex):endindex + 1]))
        # print "DEBUG: startindex "+str(startindex)+", endindex "+str(endindex)+": HowManyAreActiveNow = "+str(HowManyAreActiveNow)

        if startindex <= endindex:
            startindex = 1 + endindex

        if float(HowManyAreActiveNow) / size > burst_treshold:
            HowManyAreActive.append(1)
        else:
            HowManyAreActive.append(0)

    # 2.) calculate inter-burst-intervals
    oldvalue = 0
    IBI = 0
    IBIsList = []
    for s in HowManyAreActive:
        switch = [oldvalue, s]
        if switch == [0, 0]:
            IBI += 1
        elif switch == [0, 1]:
            # print "up"
            IBIsList.append(IBI)
            IBI = 0  # so we want to measure burst rate, not actually the IBIs
        oldvalue = s
    if IBI > 0 and len(IBIsList) > 0:
        IBIsList.append(IBI)

    print("DEBUG: " + str(len(IBIsList)) + " bursts detected.")
    # 3.) calculate burst rate in Hz
    if len(IBIsList) == 0 or sum(IBIsList) == 0:
        return 0.
    else:
        try:
            return 1. / (float(tauMS) / 1000. * float(sum(IBIsList)) / float(len(IBIsList)))
        except:
            print('error occur 0 division',tauMS, sum(IBIsList), IBIsList)
            sys.exit()


def compare_fc_metrics(folder, relative=True):
    # TODO: try saving p vals from granger test
    # TODO: visualize calcium traces, dff
    # TODO: compare statsmodel, tcgc, stats_autolag, tcgc_autolag
    if relative:
        simu = os.path.join(folder, 'utils', 'simulation')
    else:
        simu = folder
    spike = os.path.join(simu, 'spikes')
    calcium = os.path.join(simu, 'calcium')
    network = os.path.join(simu, 'networks')
    FC = os.path.join(simu, 'FC')
    # granger causality params
    DEFAULT_LAG = 2
    MAXLAG = 5
    METRIC = 'bic'

    # Comparisons start
    totalS = sum([1 for f in os.listdir(calcium) if re.search(r"(\w+)_net(\d+)_(\w+).(\w+)", f)])
    pbar = ProgressBar(totalS)
    for f in os.listdir(calcium):
        m = re.search(r"(\w+)_net(\d+)_(\w+).(\w+)", f)
        if m:
            pbar.loop_start()
            inet = f'net{m.group(2)}'
            inet_path = os.path.join(FC, inet)
            if not os.path.exists(inet_path):
                os.makedirs(inet_path)
            nfile, sfile, cfile = get_sim_files(simu, inet, m.group(3))
            for record_type in ['calcium']: #'calcium', 'dff':
                with h5py.File(cfile, 'r') as hf:
                    cdata = np.array(hf[record_type])
                cdata = cdata - np.min(cdata, axis=1, keepdims=True)
                # TODO: if name scheme gets too confusing use __ as separater
                try:
                    autolag = granger_select_order(cdata, MAXLAG)[METRIC]
                    common_keywords = f'{record_type}_'
                    # Run GC with te causality and save with pickle
                    results_tegc_dlag = fc_te_caulsaity(inet, cdata, common_keywords, lag=DEFAULT_LAG,
                                                        method='mi', pickle_path=inet_path)
                    results_tegc_autolag = fc_te_caulsaity(inet, cdata, common_keywords+'auto',
                                                        lag=autolag, method='mi', pickle_path=inet_path)

                    # # Run GC with te causality and save with pickle
                    # results_tegc_dlag = fc_te_caulsaity(inet, cdata, common_keywords+'tegc', lag=DEFAULT_LAG,
                    #                                     pickle_path=inet_path)
                    # results_tegc_autolag = fc_te_caulsaity(inet, cdata, common_keywords+'tegc_auto',
                    #                                             lag=autolag, pickle_path=inet_path)

                    # # RUN gc with statsmodel
                    # biggerLag = max(autolag, DEFAULT_LAG)
                    # fstats_dlag = os.path.join(inet_path,
                    #                            f'{inet}_{common_keywords}stats_order_{DEFAULT_LAG}.p')
                    # fstats_autolag = os.path.join(inet_path,
                    #                            f'{inet}_{common_keywords}statsauto_order_{autolag}.p')
                    # fstats_autolag_pvals = os.path.join(inet_path,
                    #                               f'{inet}_{common_keywords}statsautoPVAL_order_{autolag}.p')
                    #
                    # gcs_vals, p_vals = statsmodel_granger(cdata, maxlag=biggerLag, useLast=False)
                    #
                    # results_stats_dlag = gcs_vals[:, :, DEFAULT_LAG-1]
                    # results_stats_autolag = gcs_vals[:, :, autolag-1]
                    # results_stats_autolag_pvals = p_vals['ssr_chi2test'][:, :, autolag-1]
                    # with open(fstats_dlag, 'wb') as p_file:
                    #     pickle.dump(results_stats_dlag, p_file)
                    # with open(fstats_autolag, 'wb') as p_file:
                    #     pickle.dump(results_stats_autolag, p_file)
                    # with open(fstats_autolag_pvals, 'wb') as p_file:
                    #     pickle.dump(results_stats_autolag_pvals, p_file)
                except scipy.linalg.LinAlgError:
                    print(f"skipping {inet}")
            pbar.loop_end(inet)


def connection_probability(jfile):
    N = int(jfile['size'])
    return int(jfile['con']) * 2 / (N * (N - 1))


def fc_evaluation_ROC(fc_vals, truth, tag, ax=None, lag=None):
    # cite: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    assert fc_vals.shape == truth.shape
    if len(fc_vals.shape) == 2:
        fc_vals, truth = fc_vals[:, :, np.newaxis], truth[:, :, np.newaxis]
        assert lag is not None

    roc_aucs = []
    for i in range(fc_vals.shape[-1]):
        fpr, tpr, _ = roc_curve(truth[:, :, i].ravel(order='C'), fc_vals[:, :, i].ravel(order='C'))
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        if ax is not None:
            ax.plot(fpr, tpr,
                     label=f'lag {lag if lag is not None else i + 1} ROC curve {tag} (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC curve')
            ax.legend(loc="lower right")
        else:
            plt.plot(fpr, tpr, label=f'lag {lag if lag else i+1} ROC curve {tag} (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC curve')
            plt.legend(loc="lower right")
            plt.show()
    return roc_aucs


def fc_evaluation_granger_statsmodel(gcs_val, p_vals, truth, tag):
    # TODO: add ROC
    if isinstance(p_vals, dict):
        p_vals = p_vals['ssr_chi2test']
    # pval test:
    THRES = 0.05
    pvs = np.zeros_like(p_vals)
    pvs[p_vals <= 0.05] = 1
    L = gcs_val.shape[-1]
    # Correlation test
    corrs = [np.corrcoef(gcs_val[:, :, i].ravel(order='C'), truth.ravel(order='C'))[0, 1] for i in range(L)]
    corrs2 = [np.corrcoef(pvs[:, :, i].ravel(order='C'), truth.ravel(order='C'))[0, 1] for i in range(L)]
    TPs = [np.sum(pvs[:, :, i] * truth) / np.sum(truth) for i in range(5)]
    test_mat = {'gc_value': gcs_val, 'complement_p_values': 1-p_vals}
    roc_aucs = {}
    fig, axes = plt.subplots(nrows=2, ncols=1)
    for i, k in enumerate(test_mat):
        roc_aucs[k] = fc_evaluation_ROC(test_mat[k], truth, tag+'_'+k, ax=axes[i])
    return corrs, corrs2, TPs


def fc_evaluation_granger(inet_folder, truth, tag):
    # TODO: add ROC
    allfiles = [f for f in os.listdir(inet_folder) if f[-2:] == '.p']
    all_aucs = {}
    for f in os.listdir(inet_folder):
        signs = f.split('_')
        tag = signs[2]
        with open(os.path.join(inet_folder, f), 'rb') as pfile:
            fc_vals = pickle.load(pfile)
        if isinstance(fc_vals, list):
            fc_vals = fc_vals[0]
        if 'PVAL' in f:
            fc_vals = 1-fc_vals
        aucs = fc_evaluation_ROC(fc_vals, truth, tag, ax=None, lag=signs[4])
        all_aucs[tag] = aucs
    return all_aucs


"""-------------------------------------------------
------------- nest network generation --------------
----------------------------------------------------"""

# TODO: try test the algorithms with this simple example
def voltmeter_example():
    models = ['iaf_psc_alpha', 'iaf_psc_delta', 'iaf_psc_exp', 'aeif_cond_alpha', 'izhikevich']
    NMODEL = models[0]
    JNOISE=4.
    WEIGHT=5.
    dt=0.1
    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": 1, "resolution": dt})

    # CUSTOM SETTINGS
    neuron_params = {"C_m": 1.0,
                     "tau_m": 20.0,
                     "t_ref": 2.0,
                     "E_L": -70.0,
                     "V_th": -55.0}
    nest.SetDefaults(models[0], neuron_params)
    nest.SetDefaults("tsodyks_synapse",{"delay": 1.5,"tau_rec": 500.0, "tau_fac": 0.0,"U":0.3})

    neuron = nest.Create(NMODEL)
    neuron2 = nest.Create(NMODEL)
    noise = nest.Create("poisson_generator", 1, {"rate": 1.6})
    nest.CopyModel("static_synapse", "poisson", {"weight": JNOISE})
    nest.SetDefaults()
    nest.Connect(noise, neuron+neuron2, syn_spec="poisson")
    nest.Connect(neuron, neuron2, syn_spec={'model':'tsodyks_synapse', 'weight':WEIGHT})
    # nest.SetStatus(neuron, "I_e", 376.0)

    vm = nest.Create('voltmeter')
    nest.SetStatus(vm, "withtime", True)

    sd = nest.Create('spike_detector')

    nest.Connect(vm, neuron+ neuron2)
    nest.Connect(neuron+neuron2, sd)

    nest.Simulate(10000.)

    potentials = nest.GetStatus(vm, "events")[0]["V_m"]
    times = nest.GetStatus(vm, "events")[0]["times"]
    vm_senders = nest.GetStatus(vm, 'events')[0]['senders']
    spike_senders = nest.GetStatus(sd, 'events')[0]['senders']
    neuron1TAG = vm_senders == neuron[0]
    neuron2TAG = vm_senders == neuron2[0]
    spike_neuron1_tag = spike_senders == neuron[0]
    spike_neuron2_tag = spike_senders == neuron2[0]
    neuron1_times = times[neuron1TAG]
    neuron1_potentials = potentials[neuron1TAG]
    neuron2_times = times[neuron2TAG]
    neuron2_potentials = potentials[neuron2TAG]
    spikes = nest.GetStatus(sd, 'events')[0]['times']
    spike_times_neuron1 = spikes[spike_neuron1_tag]
    spike_times_neuron2 = spikes[spike_neuron2_tag]

    plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(neuron1_times, neuron1_potentials, c='b')
    plt.scatter(spike_times_neuron1, np.full_like(spike_times_neuron1, -55), c='r')
    plt.subplot(212)
    plt.plot(neuron2_times, neuron2_potentials, c='b')
    plt.scatter(spike_times_neuron2, np.full_like(spike_times_neuron2, -55), c='r')
    plt.show()


def create_network(jsonobj, weight, JENoise, noise_rate, syn_type='tsodyks_synapse',
                   save_path=None, mutable=False, print_output=1):
    """
    :param jsonobj: dictionary or jsonobj loaded from json file. Must contain {size, p}
    :param weight:
    :param JENoise:
    :param noise_rate:
    :param syn_type:
    :param save_path:
    :param print_output:
    :return:
    """
    # size = yamlobj.get('size')
    # cons = yamlobj.get('cons')
    size = jsonobj['size']
    cons = jsonobj['con'] if 'con' in jsonobj else None
    print("-> We have a network of " + str(size) + " nodes" + f" and {cons} connections" if cons else '')
    print("Resetting and creating network...")
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1, "print_time": True, "overwrite_files": True})
    # initialize parameters
    neuron_params = {"C_m": 1.0,
                     "tau_m": 20.0,
                     "t_ref": 2.0,
                     "E_L": -70.0,
                     "V_th": -55.0}
    nest.SetDefaults("iaf_psc_alpha", neuron_params)
    # Warning: delay is overwritten later if weights are given in the json file!
    nest.SetDefaults(syn_type, {"delay": 1.5, "tau_rec": 500.0, "tau_fac": 0.0, "U": 0.3})
    nest.CopyModel(syn_type, "exc", {"weight": weight}) # TODO: add inhibitory conns as well
    nest.CopyModel("static_synapse", "poisson", {"weight": JENoise})

    neuronsE = nest.Create("iaf_psc_alpha", size)
    # save GID offset of first neuron - this has the advantage that the output later will be
    # independent of the point at which the neurons were created
    GIDoffset = neuronsE[0]
    espikes = nest.Create("spike_detector")
    noise = nest.Create("poisson_generator", 1, {"rate": noise_rate})
    nest.Connect(neuronsE, espikes)

    nest.Connect(noise, neuronsE, model="poisson")
    # print "Loading connections from json file..."
    added_connections = 0
    # print additional information if present in YAML file
    if print_output:
        if 'notes' in jsonobj:
            print("-> notes of JSON file: " + jsonobj['notes'])
        if 'createdAt' in jsonobj:
            print("-> created: " + jsonobj['createdAt'])

    # Determining network connection stats
    p = jsonobj['p']
    if 'nodes' not in jsonobj: # NEW network if jsonobj is just a dict
        network = {'nodes': [{'id': i} for i in range(size)]}
    else:
        network = jsonobj
    for i in range(len(network['nodes'])):  # i starts counting at 0
        thisnode = network['nodes'][i]
        # id starts counting with 0
        cfrom = int(thisnode['id'])
        # quick fix: make sure we are reading the neurons in order and that none is skipped
        assert cfrom == neuronsE[cfrom] - GIDoffset
        assert i == cfrom

        if 'connectedTo' in thisnode:
            cto_list = thisnode['connectedTo']
        elif 'nodes' not in jsonobj:
            cto_list = [ci for ci, n in enumerate(neuronsE) if n != neuronsE[i]]
        else:
            cto_list = None
        ctos = []
        if cto_list:
            # if 'weights' not in thisnode:
            #     GID_list = [GIDoffset + cj for cj in cto_list]
            #     for j, gj in enumerate(GID_list):
            #         assert GID_list[j] == neuronsE[cto_list[j]]
            #     conn_dict = {'rule': 'pairwise_bernoulli', 'p': p}
            #     nest.Connect([cfrom+GIDoffset], GID_list, conn_dict)
            # else:
            for j in range(len(cto_list)):
                if random.random() <= p:  # choose only subset of connections
                    # todo: double check code
                    assert cto_list[j] + GIDoffset == neuronsE[int(cto_list[j])]
                    ctos.append(cto_list[j])
                    if 'weights' in thisnode:
                        assert (len(thisnode['weights']) == len(cto_list))
                        syn_dict = {"model": "exc",
                                    "weight": weight * thisnode.get('weights')[j]}
                    else:
                        syn_dict = 'exc'
                    nest.Connect([neuronsE[cfrom]], [GIDoffset+cto_list[j]], syn_spec=syn_dict)

                    if print_output > 1:
                        print("-> added connection: from #" + str(cfrom) + " to #" + str(int(cto_list[j])))
        added_connections += len(ctos)
        if save_path is not None:
            # TODO: ADD SAVE TIME LOGGING
            network['nodes'][i]['connectedTo'] = ctos
    if cons is None:
        final_cons = added_connections
        cons = len(neuronsE) * (len(neuronsE) - 1)
    else:
        final_cons = cons
    if save_path is not None and 'nodes' not in jsonobj:
        if not mutable:
            network['p'] = 1
        network['con'] = added_connections
        network['size'] = size
        inet = len([f for f in os.listdir(save_path) if f[-5:] == '.json'])
        with open(os.path.join(save_path, f'sim_exc_net{inet}_size_{size}.json'), 'w') as fp:
            json.dump(network, fp)

    print("-> " + str(added_connections) + " out of " + str(cons) + " connections (in YAML source) created.")
    return [network, neuronsE, espikes, noise, GIDoffset]


def main_simulation():
    print("------ adaptive-multibursts, Olav Stetter, Fri 14 Oct 2011, adapted by Albert J. Qu ------")
    # first, make sure command line parameters are fine
    cmd_arguments = sys.argv
    if len(cmd_arguments) != 3:
        print("usage: ./multibursts startindex endindex")
        print("Automatically generating new files")
        network_indices = [None]
    else:
        startindex = int(cmd_arguments[1])  # inclusively in file name starting from 1
        assert ((startindex > 0) and (startindex <= 100))
        endindex = int(cmd_arguments[2])  # inclusively in file name starting from 1
        assert ((startindex > 0) and (startindex <= 100))
        assert (endindex >= startindex)
        network_indices = range(startindex, endindex + 1, 1)

    # ------------------------------ Flags to customize output ------------------------------ #
    LIST_CONNECTIONS = False
    SAVE_SPIKES_TO_FILE = True
    SAVE_DETAILS_OF_ADAPATION_TO_FILE = True
    DYNAMIC = False
    SAVE_TERMINATED = False

    # ------------------------------ Simulation parameters ------------------------------ #
    MAX_ADAPTATION_ITERATIONS = 100  # maximum number of iterations to find parameters for target bursting rate
    ADAPTATION_SIMULATION_TIME = 200 * 1000.  # in ms
    hours = 1.
    SIMULATION_TIME = hours * 60. * 60. * 1000.  # in ms
    TARGET_BURST_RATE = 0.1  # in Hz
    TARGET_BURST_RATE_ACCURACY_GOAL = 0.01  # in Hz
    INITIAL_WEIGHT_JE = 5.  # internal synaptic weight, initial value, in pA
    WEIGHT_NOISE = 4.  # external synaptic weight, in pA
    NOISE_RATE = 2 * 2 * 0.4  # rate of external inputs, in Hz

    ROOT = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/nest_network"
    INPUT_PATH = os.path.join(ROOT, 'networks')
    OUTPUT_PATH = os.path.join(ROOT, 'spikes')
    if not os.path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # ------------------------------ Define iteration lists ------------------------------ #

    print(" DEBUG: network_indices: " + str(network_indices))
    p_list = [0.1, 0.3, 0.5, 0.7, 1.]  # iterate over vaious (randomly chosen) fractions of the connectivity
    # p_list = [1.]
    p_indices = range(len(p_list))
    #cc_list = [31, 63, 125, 250, 500, 1000, 2000, 4000]  # for the scaling test
    cc_list = [250, 500, 1000, 2000, 4000]
    cc_indices = range(len(cc_list))

    # ------------------------------ Main loop starts here ------------------------------ #
    adaptParList = []
    iteration = 0
    iterations = len(network_indices) * len(p_list) * len(cc_indices)
    print("launching " + str(iterations) + " iterations...")
    # TODO: make inet a unique identifier of networks, take in network folder and generate ID = #+1
    for inet in network_indices:  # this is outermost to be able to use an intermediate result of the computation
        for icc in cc_indices:
            for ip in p_indices:
                iteration += 1
                startbuild = time.time()
                ClusteringID = str(cc_list[icc])

                print("\n\n------- adaptive-multiburst: simulation " + str(iteration) + " of " + str(
                    iterations) + "-------")
                outNet = str(inet) if inet else str(len([f for f in os.listdir(INPUT_PATH)
                                                         if f[-5:] == '.json']))
                outputindexstring = "net" + outNet + "_cc" + str(icc) + "_p" + str(ip) + "_w0"

                # map and display indices
                FractionOfConnections = p_list[ip]
                print("set up of this iteration:")
                print("- simulation #" + str(inet))
                print("- network topology id: \"" + ClusteringID + "\", #" + str(icc))
                print("- fraction of connections: " + str(FractionOfConnections) + ", #" + str(ip))

                # json loading section
                print("Loading topology from disk...")
                if inet is None:
                    jsonobj = {'size': cc_list[icc], 'p': FractionOfConnections}
                else:
                    JSONinputfilename = os.path.join(INPUT_PATH,
                                                     f'sim_exc_net{inet}_size_{ClusteringID}.json')
                    with open(JSONinputfilename, "r") as filestream:
                        jsonobj = json.load(filestream)
                    if 'p' not in jsonobj:  # for p=1, immutable json files, copy the network as it is
                        jsonobj['p'] = FractionOfConnections
                copyjson = jsonobj
                # --- adaptation phase ---
                print("Starting adaptation phase...")
                weight = INITIAL_WEIGHT_JE
                burst_rate = -1
                adaptation_iteration = 1
                last_burst_rates = []
                last_JEs = []
                overheats = []
                terminate = False

                while abs(burst_rate - TARGET_BURST_RATE) > TARGET_BURST_RATE_ACCURACY_GOAL:
                    if len(last_burst_rates) < 2 or last_burst_rates[-1] == last_burst_rates[-2]:
                        if len(last_burst_rates) > 0:
                            print(
                                "---------------------- auto-burst stage II.) changing weight by 10% -------------------")
                            if burst_rate > TARGET_BURST_RATE:
                                weight *= 0.9
                            else:
                                weight *= 1.1
                        else:
                            print(
                                "------------------------- auto-burst stage I.) initial run -----------------------------")
                    else:
                        print(
                            "------------------- auto-burst stage III.) linear extrapolation --------------------------")
                        weight = ((TARGET_BURST_RATE - last_burst_rates[-2]) * (
                                last_JEs[-1] - last_JEs[-2]) / (
                                          last_burst_rates[-1] - last_burst_rates[-2])) + last_JEs[-2]
                    assert weight > 0.
                    print("adaptation #" + str(adaptation_iteration) + ": setting weight to " + str(
                        weight) + " ...")
                    if DYNAMIC:
                        jsonobj['p'] = FractionOfConnections
                    [network, neuronsE, espikes, noise, GIDoffset] = create_network(jsonobj, weight,
                                                                                    WEIGHT_NOISE, NOISE_RATE,
                                                                                    save_path=INPUT_PATH,
                                                                                    print_output=1 + LIST_CONNECTIONS)
                    size = int(network['size']) if isinstance(network['size'], str) else network['size']
                    jsonobj = network
                    nest.Simulate(ADAPTATION_SIMULATION_TIME)
                    tauMS = 50
                    xtimes = nest.GetStatus(espikes, "events")[0]["times"].flatten().tolist()
                    burst_rate = determine_burst_rate(nest.GetStatus(espikes, "events")[0]["senders"]
                                                      .flatten().tolist(), xtimes, tauMS,
                                                      ADAPTATION_SIMULATION_TIME, size)
                    print("-> the burst rate is " + str(burst_rate) + " Hz")
                    adaptation_iteration += 1
                    last_burst_rates.append(burst_rate)
                    last_JEs.append(weight)
                    single_cell_fr = 1000. * float(len(xtimes)) / \
                                     (float(ADAPTATION_SIMULATION_TIME) * float(size))
                    # TODO: tune Poisson rate according to fr
                    if single_cell_fr > 200:
                        overheats.append(single_cell_fr)
                    if len(overheats) >= 5 or adaptation_iteration >= MAX_ADAPTATION_ITERATIONS:
                        terminate = True
                        print("Network anomaly occurs, terminate now....")
                        break

                    "------------------------- auto-burst stage IV.) actual simulation -----------------------------"
                # same jsonobj

                if DYNAMIC:
                    jsonobj['p'] = FractionOfConnections
                adaptParList.append(outputindexstring + ": " + str(weight))
                adaptParList.append(f'Connection Probability: {p_list[ip]}, Clustering Size: {cc_list[icc]}')
                if terminate:
                    weight = INITIAL_WEIGHT_JE * 2
                    if len(overheats) >= 5:
                        adaptParList.append(f'overheated neurons, {np.around(overheats, 2)}\n')
                    if adaptation_iteration >= MAX_ADAPTATION_ITERATIONS:
                        adaptParList.append("simulation overflow\n")
                    outputindexstring = 'faulty_' + outputindexstring

                [network, _, espikes, _, GIDoffset] = create_network(jsonobj, weight, WEIGHT_NOISE,
                                                                     NOISE_RATE,
                                                                     save_path=INPUT_PATH,
                                                                     print_output=1 + LIST_CONNECTIONS)
                size = int(network['size']) if isinstance(network['size'], str) else network['size']
                endbuild = time.time()
                if not terminate or SAVE_TERMINATED:
                    # --- simulate ---
                    print("Simulating...")
                    nest.Simulate(SIMULATION_TIME)
                    endsimulate = time.time()

                    build_time = endbuild - startbuild
                    sim_time = endsimulate - endbuild

                    totalspikes = nest.GetStatus(espikes, "n_events")[0]
                    print("Number of neurons : ", size)
                    print("Number of spikes recorded: ", totalspikes)
                    print(
                        "Avg. spike rate of neurons: %.2f Hz" % (totalspikes / (size * SIMULATION_TIME / 1000.)))
                    print("Building time: %.2f s" % build_time)
                    print("Simulation time: %.2f s" % sim_time)

                    if SAVE_SPIKES_TO_FILE:
                        hf_name = os.path.join(OUTPUT_PATH, f"sim_spike_{outputindexstring}.hdf5")
                        print("Saving spike times to disk...")
                        # output spike times, in ms
                        s_times = nest.GetStatus(espikes, "events")[0]["times"]
                        # remove offset, such that the output array starts with 0
                        s_index = nest.GetStatus(espikes, "events")[0]["senders"]
                        if not isinstance(s_index, np.ndarray):
                            print("Some major version changes must've happened, nest no longer gives np.array..")
                            s_index = np.array(s_index, dtype=np.int) - GIDoffset
                        else:
                            s_index -= GIDoffset
                        with h5py.File(hf_name, 'w-') as hf:
                            hf.attrs['size'] = size
                            hf.create_dataset('neuron', data=s_index)
                            hf.create_dataset('spike', data=s_times)
                        # spiketimefilename = os.path.join(str(OUTPUT_PATH),
                        #                                  "s_times_" + outputindexstring + ".dat")
                        # spikeindexfilename = os.path.join(str(OUTPUT_PATH),
                        #                                   "s_index_" + outputindexstring + ".dat")
                        # inputFile = open(spiketimefilename, "w")
                        # # output spike times, in ms
                        # print("\n".join([str(x) for x in nest.GetStatus(espikes, "events")[0]["times"]]),
                        #       file=inputFile)
                        # inputFile.close()
                        #
                        # inputFile = open(spikeindexfilename, "w")
                        # # remove offset, such that the output array starts with 0
                        # print(
                        #     "\n".join(
                        #         [str(x - GIDoffset) for x in nest.GetStatus(espikes, "events")[0]["senders"]]),
                        #     file=inputFile)
                        # inputFile.close()
                if SAVE_DETAILS_OF_ADAPATION_TO_FILE:
                    adaptiveparsfilename = os.path.join(OUTPUT_PATH, f"adaptivepars_{outputindexstring}.txt")
                    adaptiveparsFile = open(adaptiveparsfilename, "w")
                    for par in adaptParList:
                        print(str(par) + "\n", file=adaptiveparsFile)
                    adaptiveparsFile.close()

    # ------------------------------ Main loop ends here ------------------------------ #


"""-------------------------------------------------
--------------- network visualization --------------
----------------------------------------------------"""


def spike_raster_plots(spike_file, ax=None, T=None, ns=None):
    with h5py.File(spike_file, 'r') as hf:
        neurons = np.array(hf['neuron'])
        spikes = np.array(hf['spike'])
    neur_iters = np.unique(neurons) if ns is None else ns
    for n in neur_iters:
        spike_times = spikes[neurons==n] / 1000
        if T is not None:
            spike_times = spike_times[spike_times <= T]

        if ax is None:
            plt.eventplot(spike_times, lineoffsets=n, linelengths=0.3)
            plt.xlabel('time (s)')
            plt.ylabel('Neuron #')
        else:
            ax.eventplot(spike_times, lineoffsets=n, linelengths=0.3)
            ax.set_xlabel('time (s)')
            ax.set_ylabel('Neuron #')


def visualize_simulated_activity(simulation, inet, ns, T=None):
    if not hasattr(ns, '__iter__'):
        ns = [ns]
    assert len(ns) <= 5, 'Too many traces!'
    network, spike, calcium = get_sim_files(simulation, inet)
    FR = None
    with h5py.File(calcium, 'r') as hf:
        cal, dff = hf['calcium'], hf['dff']
        if T is None:
            T = hf['calcium'].shape[1]
        if ns is None:
            ns = np.arange(hf['calcium'].shape[0])
        if 'fr' in hf.attrs:
            FR = hf.attrs['fr']
        else:
            FR = 1000 / SpikeCalciumizer.tauImg
        TS = int(T * FR)
        cal = cal[ns, :TS]
        dff = dff[ns, :TS]
        if not isinstance(cal, np.ndarray):
            cal = np.array(cal)
            dff = np.array(dff)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    spike_raster_plots(spike, axes[1], T=T, ns=ns)

    xs = np.arange(cal.shape[1]) / FR
    axes[0].plot(xs, cal.T)
    axes[0].legend(ns)
    axes[2].set_ylabel("calcium")
    axes[2].plot(xs, dff.T)
    axes[2].legend(ns)
    axes[2].set_ylabel("dff")
    plt.show()


if __name__ == '__main__':
    main_simulation()
