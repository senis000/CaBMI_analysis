import pdb
import sys
import os
import numpy as np
import pickle
import time
import shutil
import warnings
import h5py
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.stats import zscore
from sklearn import preprocessing
import networkx as nx
from networkx.algorithms import community
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ExpGTE import ExpGTE
from ExpGTECopy import ExpGTECopy
from utils_cabmi import *
from plotting_functions import *
from analysis_functions import *
from utils_gte import *
from utils_clustering import *
from clustering_functions import *

def count_experiments():
    """
    Counts the number of experiments and animals for which a reward end GTE
    file is created.
    """

    num_ITs = 0
    num_PTs = 0
    num_IT_experiments = 0
    num_PT_experiments = 0
    num_IT_reward_trials = 0
    num_PT_reward_trials = 0
    processed_dir = './processed/'
    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        include_animal = False
        num_experiments = 0
        num_reward_trials = 0
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            num_reward_trials += len(reward_end)
            include_animal = True
            num_experiments += 1
        if include_animal:
            if animal_dir.startswith('IT'):
                num_ITs +=1
                num_IT_experiments += num_experiments
                num_IT_reward_trials += num_reward_trials
            else:
                num_PTs += 1
                num_PT_experiments += num_experiments
                num_PT_reward_trials += num_reward_trials
    print('Number of ITs: ' + str(num_ITs))
    print('Experiments: ' + str(num_IT_experiments))
    print('Reward Trials: ' + str(num_IT_reward_trials))
    print()
    print('Number of PTs: ' + str(num_PTs))
    print('Experiments: ' + str(num_PT_experiments))
    print('Reward Trials: ' + str(num_PT_reward_trials))
            

def plot_earlylate_ITPT():
    """
    Plots the mean and standard deviation of information transfer in four
    contexts: early reward trials in IT, late reward trials in IT, early
    reward trials in PT, late reward trials in PT.
    """

    early_it = []
    late_it = []
    early_pt = []
    late_pt = []
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            total_num_rewards = len(reward_end)
            num_rewards = total_num_rewards//3 # Process first and last third
            early = reward_end[:num_rewards]
            late = reward_end[-num_rewards:]
            early = [m.flatten().tolist() for m in early]
            late = [m.flatten().tolist() for m in late]
            early = [val for sublist in early for val in sublist]
            late = [val for sublist in late for val in sublist]

            if animal_dir.startswith('IT'):
                early_it += early
                late_it += late
            else:
                early_pt += early
                late_pt += late
    early_it_mean = np.nanmean(early_it)
    early_it_std = np.nanstd(early_it)
    late_it_mean = np.nanmean(late_it)
    late_it_std = np.nanstd(late_it)
    early_pt_mean = np.nanmean(early_pt)
    early_pt_std = np.nanstd(early_pt)
    late_pt_mean = np.nanmean(late_pt)
    late_pt_std = np.nanstd(late_pt)
    means = [early_it_mean, late_it_mean, early_pt_mean, late_pt_mean]
    stds = [early_it_std, late_it_std, early_pt_std, late_pt_std]
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    labels = ['IT Early Rewards', 'IT Late Rewards',
        'PT Early Rewards', 'PT Late Rewards'
        ]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Distribution in Reward Trials')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_ITPT():
    """
    Plots the mean and std dev of information transfer in IT animals vs
    PT animals
    """

    it = []
    pt = []
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            reward_end = [m.flatten().tolist() for m in reward_end]
            reward_end = [val for sublist in reward_end for val in sublist]
            if animal_dir.startswith('IT'):
                it += reward_end
            else:
                pt += reward_end
    num_it_exp = len(it)
    num_pt_exp  = len(pt)
    it_mean = np.nanmean(it)
    it_std = np.nanstd(it)
    pt_mean = np.nanmean(pt)
    pt_std = np.nanstd(pt)
    means = [it_mean, pt_mean]
    stds = [it_std, pt_std]
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    labels = ['IT: ' + str(num_it_exp) + ' Trials',
        'PT: ' + str(num_pt_exp) + ' Trials'
        ]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Distribution in Reward Trials')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_IT_redgreen():
    """
    Plots the mean and std dev of directional information transfer between
    red and green neurons in IT animals.
    """

    results = []
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path) or not animal_dir.startswith('IT'):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
            except: # In case another process is accessing the file already.
                continue
            nerden = np.array(f['nerden'])
            redlabel = np.array(f['redlabel'])
            redlabel = redlabel[nerden]
            redlabel = redlabel.astype(int) # Group 1 is red, Group 0 is green
            reward_end_rg = [\
                group_result(m, redlabel, ignore_diagonal=False)\
                for m in reward_end
                ] 
            results += reward_end_rg
    results_means = np.nanmean(results, axis=0)
    results_stds = np.nanstd(results, axis=0)
    results_means = results_means.flatten()
    results_stds = results_stds.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    labels = ['Gr->Gr', 'Gr->Red', 'Red->Gr','Red->Red']
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, results_means, yerr=results_stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Transfer in IT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_PT_redgreen():
    """
    Plots the mean and std dev of information transfer bewteen red neurons and
    green neurons in PT animals.
    """

    results = []
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path) or not animal_dir.startswith('PT'):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
            except:
                continue # In case another process is already accessing the file
            nerden = np.array(f['nerden'])
            redlabel = np.array(f['redlabel'])
            redlabel = redlabel[nerden]
            redlabel = redlabel.astype(int) # Group 1 is red, Group 0 is green
            reward_end_rg = [\
                group_result(m, redlabel, ignore_diagonal=False)\
                for m in reward_end
                ] 
            results += reward_end_rg
    results_means = np.nanmean(results, axis=0)
    results_stds = np.nanstd(results, axis=0)
    results_means = results_means.flatten()
    results_stds = results_stds.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    labels = ['Gr->Gr', 'Gr->Red', 'Red->Gr','Red->Red']
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, results_means, yerr=results_stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Transfer in  PT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_IT_e2earlylate():
    """
    Plots the mean and std dev of information transfer between indirect neurons
    and E2 neurons in IT animals during early reward trials and late reward
    trials.
    """

    results = []
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path) or not animal_dir.startswith('IT'):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            num_neurons = reward_end[0].shape[0]
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
            except: # In case another process is already accessing the file
                continue
            e2_neur = np.array(f['e2_neur'])
            ens_neur = np.array(f['ens_neur'])
            e2_neur = ens_neur[e2_neur]
            nerden = np.array(f['nerden'])
            e2_mask = np.zeros(nerden.size)
            e2_mask[e2_neur] = 1
            e2_mask = e2_mask[nerden]
            e2_neur = np.where(e2_mask==1)
            grouping = np.zeros(num_neurons)
            grouping[e2_neur] = 1
            reward_end_e2 = [\
                group_result(m, grouping, ignore_diagonal=False)\
                for m in reward_end
                ] 
            results += reward_end_e2
    results_means = np.nanmean(results, axis=0)
    results_stds = np.nanstd(results, axis=0)
    results_means = results_means.flatten()
    results_stds = results_stds.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    labels = ['Ind.->Ind.', 'Ind.->E2', 'E2->Ind.','E2->E2']
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, results_means, yerr=results_stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Transfer in IT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_PT_e2earlylate():
    """
    Plots the mean and std dev of information transfer between indirect neurons
    and E2 neurons in PT animals during early reward trials and late reward
    trials.
    """

    results = []
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path) or not animal_dir.startswith('PT'):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            num_neurons = reward_end[0].shape[0]
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
            except: # In case another process is already accessing the file
                continue
            e2_neur = np.array(f['e2_neur'])
            ens_neur = np.array(f['ens_neur'])
            e2_neur = ens_neur[e2_neur]
            nerden = np.array(f['nerden'])
            e2_mask = np.zeros(nerden.size)
            e2_mask[e2_neur] = 1
            e2_mask = e2_mask[nerden]
            e2_neur = np.where(e2_mask==1)
            grouping = np.zeros(num_neurons)
            grouping[e2_neur] = 1
            reward_end_e2 = [\
                group_result(m, grouping, ignore_diagonal=False)\
                for m in reward_end
                ] 
            results += reward_end_e2
    results_means = np.nanmean(results, axis=0)
    results_stds = np.nanstd(results, axis=0)
    results_means = results_means.flatten()
    results_stds = results_stds.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    labels = ['Ind.->Ind.', 'Ind.->E2', 'E2->Ind.','E2->E2']
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, results_means, yerr=results_stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Transfer in PT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_PT_depth():
    """
    Plots the mean and std dev of information transfer between different depths
    in PT animals. A 2D heatmap is generated for the mean, and a 2D heatmap is
    generated for the std dev.
    """

    reward_ends = []
    depth_locations = []
    min_depth = 999
    max_depth = 0
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path) or not animal_dir.startswith('PT'):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
            except: # In case another process is already acessinng the file.
                continue
            nerden = np.array(f['nerden'])
            depth_location = np.array(f['com_cm'])[:,2]
            depth_location = depth_location[nerden]

            # Add experiment information to global variables
            depth_locations.append(depth_location)
            reward_ends.append(reward_end)
            max_exp_depth = int(np.nanmax(depth_locations))
            min_exp_depth = int(np.nanmin(depth_locations))
            if max_exp_depth > max_depth:
                max_depth = max_exp_depth
            if min_exp_depth < min_depth:
                min_depth = min_exp_depth

    num_depths = max_depth - min_depth + 1
    num_exp = 0
    depth_mat = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
    for idx, reward_end in reward_ends:
        depth_location = depth_locations[idx]
        num_neurons = reward_end[0].shape[0]
        for re in reward_end:
            num_exp += 1
            for neuron1 in range(num_neurons):
                for neuron2 in range(num_neurons):
                    if neuron1 == neuron2:
                        continue
                    depth1 = int(depth_location[neuron1])
                    depth2 = int(depth_location[neuron2])
                    depth_mat[depth1][depth2].append(re[neuron1, neuron2])
    means = [[np.nanmean(entry) for entry in row] for row in depth_mat]
    means = np.array(means)
    stds = [[np.nanstd(entry) for entry in row] for row in depth_mat]
    stds = np.array(stds)
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    im = ax.imshow(means)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Depth (in microns)')
    ax.set_ylabel('Depth (in microns)')
    ax.set_title('Average Info. Transfer in ' + str(num_exp) + ' PT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    im = ax.imshow(stds)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Depth (in microns)')
    ax.set_ylabel('Depth (in microns)')
    ax.set_title('Std Dev of Info. Transfer in ' + str(num_exp) + ' PT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_shallowIT_depth():
    """
    Plots the mean and std dev of information transfer between different depths
    in IT animals where E2 is chosen to be above 300 microns. A 2D heatmap is
    generated for the mean, and a 2D heatmap is generated for the std dev.
    """

    reward_ends = []
    depth_locations = []
    min_depth = 999
    max_depth = 0
    processed_dir = './processed/'
    E2_depth_thresh = 300

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path) or not animal_dir.startswith('IT'):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
            except: # In case another process is already acessinng the file.
                continue
            nerden = np.array(f['nerden'])
            depth_location = np.array(f['com_cm'])[:,2]
            e2_neur = np.array(f['e2_neur'])
            ens_neur = np.array(f['ens_neur'])
            e2_neur = ens_neur[e2_neur]
            if np.max(depth_location[e2_neur]) > E2_depth_thresh:
                continue
            depth_location = depth_location[nerden]

            # Add experiment information to global variables
            depth_locations.append(depth_location)
            reward_ends.append(reward_end)
            max_exp_depth = int(np.nanmax(depth_locations))
            min_exp_depth = int(np.nanmin(depth_locations))
            if max_exp_depth > max_depth:
                max_depth = max_exp_depth
            if min_exp_depth < min_depth:
                min_depth = min_exp_depth

    num_depths = max_depth - min_depth + 1
    num_exp = 0
    depth_mat = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
    for idx, reward_end in reward_ends:
        depth_location = depth_locations[idx]
        num_neurons = reward_end[0].shape[0]
        for re in reward_end:
            num_exp += 1
            for neuron1 in range(num_neurons):
                for neuron2 in range(num_neurons):
                    if neuron1 == neuron2:
                        continue
                    depth1 = int(depth_location[neuron1])
                    depth2 = int(depth_location[neuron2])
                    depth_mat[depth1][depth2].append(re[neuron1, neuron2])
    means = [[np.nanmean(entry) for entry in row] for row in depth_mat]
    means = np.array(means)
    stds = [[np.nanstd(entry) for entry in row] for row in depth_mat]
    stds = np.array(stds)
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    im = ax.imshow(means)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Depth (in microns)')
    ax.set_ylabel('Depth (in microns)')
    ax.set_title('Average Info. Transfer in ' + str(num_exp) +\
        ' Shallow IT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    im = ax.imshow(stds)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Depth (in microns)')
    ax.set_ylabel('Depth (in microns)')
    ax.set_title('Std Dev of Info. Transfer in ' + str(num_exp) +\
        ' Shallow iT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_deepIT_depth():
    """
    Plots the mean and std dev of information transfer between different depths
    in IT animals where E2 is chosen to be below 300 microns. A 2D heatmap is
    generated for the mean, and a 2D heatmap is generated for the std dev.
    """

    reward_ends = []
    depth_locations = []
    min_depth = 999
    max_depth = 0
    processed_dir = './processed/'
    E2_depth_thresh = 300

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path) or not animal_dir.startswith('IT'):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
            except: # In case another process is already acessinng the file.
                continue
            nerden = np.array(f['nerden'])
            depth_location = np.array(f['com_cm'])[:,2]
            e2_neur = np.array(f['e2_neur'])
            ens_neur = np.array(f['ens_neur'])
            e2_neur = ens_neur[e2_neur]
            if np.min(depth_location[e2_neur]) < E2_depth_thresh:
                continue
            depth_location = depth_location[nerden]

            # Add experiment information to global variables
            depth_locations.append(depth_location)
            reward_ends.append(reward_end)
            max_exp_depth = int(np.nanmax(depth_locations))
            min_exp_depth = int(np.nanmin(depth_locations))
            if max_exp_depth > max_depth:
                max_depth = max_exp_depth
            if min_exp_depth < min_depth:
                min_depth = min_exp_depth

    num_depths = max_depth - min_depth + 1
    num_exp = 0
    depth_mat = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
    for idx, reward_end in reward_ends:
        depth_location = depth_locations[idx]
        num_neurons = reward_end[0].shape[0]
        for re in reward_end:
            num_exp += 1
            for neuron1 in range(num_neurons):
                for neuron2 in range(num_neurons):
                    if neuron1 == neuron2:
                        continue
                    depth1 = int(depth_location[neuron1])
                    depth2 = int(depth_location[neuron2])
                    depth_mat[depth1][depth2].append(re[neuron1, neuron2])
    means = [[np.nanmean(entry) for entry in row] for row in depth_mat]
    means = np.array(means)
    stds = [[np.nanstd(entry) for entry in row] for row in depth_mat]
    stds = np.array(stds)
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    im = ax.imshow(means)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Depth (in microns)')
    ax.set_ylabel('Depth (in microns)')
    ax.set_title('Average Info. Transfer in ' + str(num_exp) +\
        ' Deep IT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    im = ax.imshow(stds)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Depth (in microns)')
    ax.set_ylabel('Depth (in microns)')
    ax.set_title('Std Dev of Info. Transfer in ' + str(num_exp) +\
        ' Deep IT Experiments')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_learning():
    """
    Plots the mean and std dev of information transfer in learning days vs
    non-learning days. Arbitrarily, we will use the slope of the hits/5-min
    graph as a metric for learning. Experiments with slope < 0.2 are considered
    non-learning while experiments with slope > 0.4 are considered learning.
    """

    learning = []
    non_learning = []
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            reward_end_path = day_path + 'reward_end.p'
            if not os.path.isfile(reward_end_path):
                continue
            with open(reward_end_path, 'rb') as f:
                reward_end = pickle.load(f)
            reward_end = [m.flatten().tolist() for m in reward_end]
            reward_end = [val for sublist in reward_end for val in sublist]
            try:
                _, _, reg = learning_params('./', animal, day, bin_size=5)
            except: # In case another process is already acessinng the file.
                continue
            slope = reg.coef_[0]
            if slope < 0.2:
                non_learning += reward_end
            if slope > 0.4:
                learning += reward_end
    num_learning = len(learning)
    num_non_learning = len(non_learning)
    learning_mean = np.nanmean(learning)
    learning_std = np.nanstd(learning)
    non_learning_mean = np.nanmean(non_learning)
    non_learning_std = np.nanstd(non_learning)
    means = [learning_mean, non_learning_mean]
    stds = [learning_std, non_learning_std]
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    labels = ['Learning Experiments', 'Non-learning Experiments']
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Distribution in Reward Trials')
    ax.yaxis.grid(True)
    plt.show(block=True)
