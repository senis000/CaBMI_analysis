import pdb
import sys
import os
from os.path import isfile
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
    num_IT_base = 0
    num_IT_expend = 0
    num_PT_base = 0
    num_PT_expend = 0
    processed_dir = './processed/'
    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        include_animal = False
        num_baseline = 0
        num_expend = 0
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            baseline_path = day_path + 'baseline.p'
            expend_path = day_path + 'experiment_end.p'
            if os.path.isfile(baseline_path):
                num_baseline += 1
                include_animal = True
            if os.path.isfile(expend_path):
                num_expend += 1
                include_animal = True
        if include_animal:
            if animal_dir.startswith('IT'):
                num_ITs +=1
                num_IT_base += num_baseline
                num_IT_expend += num_expend
            else:
                num_PTs += 1
                num_PT_base += num_baseline
                num_PT_expend += num_expend
    print('Number of ITs: ' + str(num_ITs))
    print('Baselines: ' + str(num_IT_base))
    print('Experiment Ends: ' + str(num_IT_expend))
    print()
    print('Number of PTs: ' + str(num_PTs))
    print('Baselines: ' + str(num_PT_base))
    print('Experiment Ends: ' + str(num_PT_expend))

def plot_learning():
    """
    Comparing info. transfer in baseline vs experiment end for learning vs
    non-learning days.
    """

    learning_baseline = OnlineNormalEstimator()
    learning_expend = OnlineNormalEstimator()
    non_learning_baseline = OnlineNormalEstimator()
    non_learning_expend = OnlineNormalEstimator()
    num_learning = 0
    num_non_learning = 0
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            baseline_path = day_path + 'baseline.p'
            expend_path = day_path + 'experiment_end.p'
            if not isfile(baseline_path) or not isfile(expend_path):
                continue
            with open(baseline_path, 'rb') as f:
                baseline = pickle.load(f)[0]
            with open(expend_path, 'rb') as f:
                expend = pickle.load(f)[0]
            baseline = [val for val in baseline.flatten() if not np.isnan(val)]
            expend = [val for val in expend.flatten() if not np.isnan(val)]
            try:
                _, _, reg = learning_params('./', animal_dir, day_dir, bin_size=5)
            except: # In case another process is already accessing this file
                continue
            slope = reg.coef_[0]
            if slope < 0.2:
                for val in baseline:
                    non_learning_baseline.handle(val)
                for val in expend:
                    non_learning_expend.handle(val)
                num_non_learning += 1
            if slope > 0.4:
                for val in baseline:
                    learning_baseline.handle(val)
                for val in expend:
                    learning_expend.handle(val)
                num_learning += 1
    means = [
        learning_baseline.mean(), learning_expend.mean(),
        non_learning_baseline.mean(), non_learning_expend.mean()
        ]
    stds = [
        learning_baseline.std(), learning_expend.std(),
        non_learning_baseline.std(), non_learning_expend.std()
        ]
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    labels = [
        'Learning: Baseline\n(%d Total)'%num_learning,
        'Learning: Exp End\n(%d Total)'%num_learning,
        'Non-Learning: Baseline\n(%d Total)'%num_non_learning,
        'Non-Learning: Exp End\n(%d Total)'%num_non_learning
        ]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Distribution')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_ITPT():
    """
    Comparing info. transfer in baseline vs experiment end for IT vs PT animals.
    """

    IT_baseline = OnlineNormalEstimator()
    IT_expend = OnlineNormalEstimator()
    PT_baseline = OnlineNormalEstimator()
    PT_expend = OnlineNormalEstimator()
    num_it = 0
    num_pt = 0
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            baseline_path = day_path + 'baseline.p'
            expend_path = day_path + 'experiment_end.p'
            if not isfile(baseline_path) or not isfile(expend_path):
                continue
            with open(baseline_path, 'rb') as f:
                baseline = pickle.load(f)[0]
            with open(expend_path, 'rb') as f:
                expend = pickle.load(f)[0]
            baseline = [val for val in baseline.flatten() if not np.isnan(val)]
            expend = [val for val in expend.flatten() if not np.isnan(val)]
            if animal_dir.startswith('PT'):
                for val in baseline:
                    PT_baseline.handle(val)
                for val in expend:
                    PT_expend.handle(val)
                num_pt += 1
            if animal_dir.startswith('IT'):
                for val in baseline:
                    IT_baseline.handle(val)
                for val in expend:
                    IT_expend.handle(val)
                num_it += 1
    means = [
        IT_baseline.mean(), IT_expend.mean(),
        PT_baseline.mean(), PT_expend.mean()
        ]
    stds = [
        IT_baseline.std(), IT_expend.std(),
        PT_baseline.std(), PT_expend.std()
        ]
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    labels = [
        'IT: Baseline\n(%d Total)'%num_it,
        'IT: Exp End\n(%d Total)'%num_it,
        'PT: Baseline\n(%d Total)'%num_pt,
        'PT: Exp End\n(%d Total)'%num_pt
        ]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE Distribution')
    ax.yaxis.grid(True)
    plt.show(block=True)

def plot_E2_learning():
    """
    Comparing E2 info. transfer in baseline vs experiment end for learning vs
    non-learning experiments in IT/PT
    """

    IT_learning_baseline = OnlineNormalEstimator()
    IT_learning_expend = OnlineNormalEstimator()
    IT_non_learning_baseline = OnlineNormalEstimator()
    IT_non_learning_expend = OnlineNormalEstimator()
    PT_learning_baseline = OnlineNormalEstimator()
    PT_learning_expend = OnlineNormalEstimator()
    PT_non_learning_baseline = OnlineNormalEstimator()
    PT_non_learning_expend = OnlineNormalEstimator()
    num_it_learning = 0
    num_pt_learning = 0
    num_it_non_learning = 0
    num_pt_non_learning = 0
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            baseline_path = day_path + 'baseline.p'
            expend_path = day_path + 'experiment_end.p'
            if not isfile(baseline_path) or not isfile(expend_path):
                continue
            with open(baseline_path, 'rb') as f:
                baseline = pickle.load(f)[0]
            with open(expend_path, 'rb') as f:
                expend = pickle.load(f)[0]
            num_neurons = baseline.shape[0]
            try:
                f = h5py.File(day_path + 'full_' + animal_dir + '_' + day_dir +\
                    '__data.hdf5')
                _, _, _, reg = learning_params('./', animal_dir, day_dir, bin_size=5)
            except: # In case another process is already accessing the file
                continue
            try:
                e2_neur = np.array(f['e2_neur'])
            except:
                print(day_path)
                continue
            ens_neur = np.array(f['ens_neur'])
            e2_neur = ens_neur[e2_neur]
            nerden = np.array(f['nerden'])
            e2_mask = np.zeros(nerden.size)
            e2_mask[e2_neur] = 1
            e2_mask = e2_mask[nerden]
            baseline = group_result(baseline, e2_mask, ignore_diagonal=False)
            expend = group_result(expend, e2_mask, ignore_diagonal=False)
            slope = reg.coef_[0]
            if slope < 0:
                if animal_dir.startswith('IT'):
                    IT_non_learning_baseline.handle(baseline[0,1])
                    IT_non_learning_baseline.handle(baseline[1,0])
                    IT_non_learning_expend.handle(expend[0,1])
                    IT_non_learning_expend.handle(expend[1,0])
                    num_it_non_learning += 1
                else:
                    PT_non_learning_baseline.handle(baseline[0,1])
                    PT_non_learning_baseline.handle(baseline[1,0])
                    PT_non_learning_expend.handle(expend[0,1])
                    PT_non_learning_expend.handle(expend[1,0])
                    num_pt_non_learning += 1
            if slope > 0.2:
                if animal_dir.startswith('IT'):
                    IT_learning_baseline.handle(baseline[0,1])
                    IT_learning_baseline.handle(baseline[1,0])
                    IT_learning_expend.handle(expend[0,1])
                    IT_learning_expend.handle(expend[1,0])
                    num_it_learning += 1
                else:
                    PT_learning_baseline.handle(baseline[0,1])
                    PT_learning_baseline.handle(baseline[1,0])
                    PT_learning_expend.handle(expend[0,1])
                    PT_learning_expend.handle(expend[1,0])
                    num_pt_learning += 1
    means = [
        IT_learning_baseline.mean(), IT_learning_expend.mean(),
        IT_non_learning_baseline.mean(), IT_non_learning_expend.mean(),
        PT_learning_baseline.mean(), PT_learning_expend.mean(),
        PT_non_learning_baseline.mean(), PT_non_learning_expend.mean()
        ]
    stds = [
        IT_learning_baseline.std(), IT_learning_expend.std(),
        IT_non_learning_baseline.std(), IT_non_learning_expend.std(),
        PT_learning_baseline.std(), PT_learning_expend.std(),
        PT_non_learning_baseline.std(), PT_non_learning_expend.std()
        ]
    fig, ax = plt.subplots(1, 1, figsize=(9,3))
    labels = [
        'IT Learning:\nBaseline\n(%d Total)'%num_it_learning,
        'IT Learning:\nExp End\n(%d Total)'%num_it_learning,
        'IT Non-Learning:\nBaseline\n(%d Total)'%num_it_non_learning,
        'IT Non-Learning:\nExp End\n(%d Total)'%num_it_non_learning,
        'PT Learning:\nBaseline\n(%d Total)'%num_pt_learning,
        'PT Learning:\nExp End\n(%d Total)'%num_pt_learning,
        'PT Non-Learning:\nBaseline\n(%d Total)'%num_pt_non_learning,
        'PT Non-Learning:\nExp End\n(%d Total)'%num_pt_non_learning
        ]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, {'fontsize': 12})
    ax.set_title('GTE into/out-of E2 Neurons')
    ax.yaxis.grid(True)
    pdb.set_trace()
    plt.show(block=True)

def plot_E2_ITPT():
    """
    Comparing E2 info. transfer in baseline vs experiment end for
    IT vs PT animals.
    """

    IT_baseline = OnlineNormalEstimator()
    IT_expend = OnlineNormalEstimator()
    PT_baseline = OnlineNormalEstimator()
    PT_expend = OnlineNormalEstimator()
    num_it = 0
    num_pt = 0
    processed_dir = './processed/'

    for animal_dir in os.listdir(processed_dir):
        animal_path = processed_dir + animal_dir + '/'
        if not os.path.isdir(animal_path):
            continue
        for day_dir in os.listdir(animal_path):
            day_path = animal_path + day_dir + '/'
            baseline_path = day_path + 'baseline.p'
            expend_path = day_path + 'experiment_end.p'
            if not isfile(baseline_path) or not isfile(expend_path):
                continue
            with open(baseline_path, 'rb') as f:
                baseline = pickle.load(f)[0]
            with open(expend_path, 'rb') as f:
                expend = pickle.load(f)[0]
            num_neurons = baseline.shape[0]
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
            baseline = group_result(baseline, e2_mask, ignore_diagonal=False)
            expend = group_result(expend, e2_mask, ignore_diagonal=False)
            if animal_dir.startswith('PT'):
                PT_baseline.handle(baseline[0,1])
                PT_baseline.handle(baseline[1,0])
                PT_expend.handle(expend[0,1])
                PT_expend.handle(expend[1,0])
                num_pt += 1
            else:
                IT_baseline.handle(baseline[0,1])
                IT_baseline.handle(baseline[1,0])
                IT_expend.handle(expend[0,1])
                IT_expend.handle(expend[1,0])
                num_it += 1
    means = [
        IT_baseline.mean(), IT_expend.mean(),
        PT_baseline.mean(), PT_expend.mean()
        ]
    stds = [
        IT_baseline.std(), IT_expend.std(),
        PT_baseline.std(), PT_expend.std()
        ]
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    labels = [
        'IT: Baseline\n(%d Total)'%num_it,
        'IT: Exp End\n(%d Total)'%num_it,
        'PT: Baseline\n(%d Total)'%num_pt,
        'PT: Exp End\n(%d Total)'%num_pt
        ]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=stds, align='center',
        alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Information Transfer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('GTE into/out-of E2 Neurons')
    ax.yaxis.grid(True)
    plt.show(block=True)
