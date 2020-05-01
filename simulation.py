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
import json
import h5py
import re
import subprocess
# data
import numpy as np
import random
# plotting
import matplotlib.pyplot as plt
# simulation
import nest


"""-------------------------------------------------
-------- network topology & data management --------
----------------------------------------------------"""


def create_jsons(size):
    # TODO add creation time
    jsonobj = {'size': size, 'nodes': [None] * size}
    for i in range(size):
        jsonobj['nodes'][i] = {'id': i}


def spike_pairs_to_hdf5(folder):
    removes = []
    for f in os.listdir(folder):
        m = re.match("s_index_(\w+).dat", f)
        if m:
            f1 = os.path.join(folder, f)
            f2 = os.path.join(folder, f"s_times_{m.group(0)}.dat")
            removes.append(f1)
            removes.append(f2)
            s_index = np.loadtxt(f1, dtype=np.int)
            s_times = np.loadtxt(f2, dtype=np.float)
            with h5py.File(os.path.join(folder, f'sim_spike_{m.group(9)}.hdf5'), 'w-') as hf:
                hf.create_dataset('neuron', data=s_index)
                hf.create_dataset('spike', data=s_times)
    for r in removes:
        os.remove(r)


def spike_to_calcium_C():
    exe_code = subprocess.call([
        f"./te-causality/transferentropy-sim/{method}", control_file_name
    ])


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


"""-------------------------------------------------
------------- nest network generation ----------
----------------------------------------------------"""


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
    nest.SetDefaults("tsodyks_synapse",{"delay":1.5,"tau_rec":500.0,"tau_fac":0.0,"U":0.3})

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


def spike_raster_plots(spike_file):
    with h5py.File(spike_file, 'r') as hf:
        neurons = np.array(hf['neuron'])
        spikes = np.array(hf['spikes'])
    for n in np.unique(neurons):
        plt.eventplot(spikes[neurons==n], lineoffsets=n, linelengths=0.3)

if __name__ == '__main__':
    main_simulation()