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
from ExpGTECopy import ExpGTECopy
from utils_cabmi import *
from plotting_functions import *
from analysis_functions import *
from utils_gte import *
from utils_clustering import *
from clustering_functions import *

def plot_learning():
    """
    Comparing info. transfer in baseline vs experiment end for learning vs
    non-learning days.
    """

    learning_baseline = []
    learning_expend = []
    non_learning_baseline = []
    non_learning_expend = []
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
            baseline = baseline.flatten().tolist()
            expend = expend.flatten().tolist()
            try:
                _, _, reg = learning_params('./', animal, day, bin_size=5)
            except: # In case another process is already accessing this file
                continue
            slope = reg.coef_[0]
            if slope < 0.2:
                non_learning_baseline += baseline
                non_learning_expend += expend
            if slope > 0.4:
                learning_baseline += baseline
                learning_expend += expend
    learning_baseline_mean = np.nanmean(learning_baseline)
    learning_expend_mean = np.nanmean(learning_expend)
    non_learning_baseline_mean = np.nanmean(non_learning_baseline)
    non_learning_expend_mean = np.nanmean(non_learning_expend)
    learning_baseline_std = np.nanstd(learning_baseline)
    learning_expend_std = np.nanstd(learning_expend)
    non_learning_baseline_std = np.nanstd(non_learning_baseline)
    non_learning_expend_std = np.nanstd(non_learning_expend)
    means = [
        learning_baseline_mean, learning_expend_mean,
        non_learning_baseline_mean, non_learning_expend_mean
        ]
    stds = [
        learning_baseline_std, learning_expend_std,
        non_learning_baseline_std, non_learning_expend_std
        ]
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    labels = [
        'Learning: Baseline', 'Learning: Exp End',
        'Non-Learning: Baseline', 'Non-Learning: Exp End'
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

    IT_baseline = []
    IT_expend = []
    PT_baseline = []
    PT_expend = []
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
            baseline = baseline.flatten().tolist()
            expend = expend.flatten().tolist()
            if animal_dir.startswith('PT'):
                PT_baseline += baseline
                PT_expend += expend
            if animal_dir.startswith('IT'):
                IT_baseline += baseline
                IT_expend += expend
    IT_baseline_mean = np.nanmean(IT_baseline)
    IT_expend_mean = np.nanmean(IT_expend)
    PT_baseline_mean = np.nanmean(PT_baseline)
    PT_expend_mean = np.nanmean(PT_expend)
    IT_baseline_std = np.nanstd(IT_baseline)
    IT_expend_std = np.nanstd(IT_expend)
    PT_baseline_std = np.nanstd(PT_baseline)
    PT_expend_std = np.nanstd(PT_expend)
    means = [
        IT_baseline_mean, IT_expend_mean,
        PT_baseline_mean, PT_expend_mean
        ]
    stds = [
        IT_baseline_std, IT_expend_std,
        PT_baseline_std, PT_expend_std
        ]
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    labels = [
        'IT: Baseline', 'IT: Exp End',
        'PT: Baseline', 'PT: Exp End'
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
    non-learning experiments
    """

    learning_baseline = []
    learning_expend = []
    non_learning_baseline = []
    non_learning_expend = []
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
                _, _, reg = learning_params('./', animal, day, bin_size=5)
            except: # In case another process is already accessing the file
                continue
            e2_neur = np.array(f['e2_neur'])
            ens_neur = np.array(f['ens_neur'])
            e2_neur = ens_neur[e2_neur]
            grouping = np.zeros(num_neurons)
            grouping[e2_neur] = 1
            baseline = group_result(baseline, grouping, ignore_diagonal=False)
            expend = group_result(expend, grouping, ignore_diagonal=False)
            slope = reg.coef_[0]
            if slope < 0.2:
                non_learning_baseline.append(baseline[0,1])
                non_learning_baseline.append(baseline[1,0])
                non_learning_expend.append(expend[0,1])
                non_learning_expend.append(expend[1,0])
            if slope > 0.4:
                learning_baseline.append(baseline[0,1])
                learning_baseline.append(baseline[1,0])
                learning_expend.append(expend[0,1])
                learning_expend.append(expend[1,0])
    non_learning_baseline_mean = np.nanmean(non_learning_baseline)
    non_learning_baseline_std = np.nanstd(non_learning_baseline)
    non_learning_expend_mean = np.nanmean(non_learning_expend)
    non_learning_expend_std = np.nanstd(non_learning_expend)
    learning_baseline_mean = np.nanmean(learning_baseline)
    learning_baseline_std = np.nanstd(learning_baseline)
    learning_expend_mean = np.nanmean(learning_expend)
    learning_expend_std = np.nanstd(learning_expend)
    means = [
        learning_baseline_mean, learning_expend_mean
        non_learning_baseline_mean, non_learning_expend_mean
        ]
    stds = [
        learning_baseline_std, learning_expend_std
        non_learning_baseline_std, non_learning_expend_std
        ]
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    labels = [
        'Learning: Baseline', 'Learning: Exp End',
        'Non-Learning: Baseline', 'Non-Learning: Exp End'
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

def plot_E2_ITPT():
    """
    Comparing E2 info. transfer in baseline vs experiment end for
    IT vs PT animals.
    """

    IT_baseline = []
    IT_expend = []
    PT_baseline = []
    PT_expend = []
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
            grouping = np.zeros(num_neurons)
            grouping[e2_neur] = 1
            baseline = group_result(baseline, grouping, ignore_diagonal=False)
            expend = group_result(expend, grouping, ignore_diagonal=False)
            if animal_dir.startswith('PT'):
                PT_baseline.append(baseline[0,1])
                PT_baseline.append(baseline[1,0])
                PT_expend.append(expend[0,1])
                PT_expend.append(expend[1,0])
            else:
                IT_baseline.append(baseline[0,1])
                IT_baseline.append(baseline[1,0])
                IT_expend.append(expend[0,1])
                IT_expend.append(expend[1,0])
    PT_baseline_mean = np.nanmean(PT_baseline)
    PT_baseline_std = np.nanstd(PT_baseline)
    PT_expend_mean = np.nanmean(PT_expend)
    PT_expend_std = np.nanstd(PT_expend)
    IT_baseline_mean = np.nanmean(IT_baseline)
    IT_baseline_std = np.nanstd(IT_baseline)
    IT_expend_mean = np.nanmean(IT_expend)
    IT_expend_std = np.nanstd(IT_expend)
    means = [
        IT_baseline_mean, IT_expend_mean
        PT_baseline_mean, PT_expend_mean
        ]
    stds = [
        IT_baseline_std, IT_expend_std
        PT_baseline_std, PT_expend_std
        ]
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    labels = [
        'Learning: Baseline', 'Learning: Exp End',
        'Non-Learning: Baseline', 'Non-Learning: Exp End'
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