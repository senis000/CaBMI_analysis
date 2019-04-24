
# 2p to record the neuronal data
# Scanimage defines the format of data

__author__ = 'Nuria'

# C++ extension for data acquisition

import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import pandas as pd
import seaborn as sns
import os
import math
import random
import scipy
import copy
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.stats.mstats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib import interactive
import utils_cabmi as ut

interactive(True)

def learning(folder, animal, day, sec_var='', to_plot=True):
    '''
    Obtain the learning rate over time.
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in YYMMDD format
        TO_PLOT: Boolean; whether or not to plot the changing learning rate
    Outputs:
        HPM: Numpy array; hits per minute
        TTH: Numpy array; time to hit (in frames)
        PERCENTAGE_CORRECT: float; the proportion of hits out of all trials
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        ) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fr = f.attrs['fr']
    blen = f.attrs['blen']
    hits = np.asarray(f['hits'])
    miss = np.asarray(f['miss'])
    array_t1 = np.asarray(f['array_t1'])
    array_miss = np.asarray(f['array_miss'])
    trial_end = np.asarray(f['trial_end'])
    trial_start = np.asarray(f['trial_start'])
    percentage_correct = hits.shape[0]/trial_end.shape[0]
    bins = np.arange(0, trial_end[-1]/fr, 60)
    [hpm, xx] = np.histogram(hits/fr, bins)

    tth = trial_end[array_t1] + 1 - trial_start[array_t1]
    
    if to_plot:
        fig1 = plt.figure()
        ax = fig1.add_subplot(121)
        sns.regplot(xx[1:]/60.0, hpm, label='hits per min')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Hit Rate (hit/min)')
        ax1 = fig1.add_subplot(122)
        sns.regplot(np.arange(tth.shape[0]), tth, label='time to hit')
        ax1.set_xlabel('Reward Trial')
        ax1.set_ylabel('Number of Frames')
        ax1.yaxis.set_label_position('right')
        fig1.savefig(folder_path + 'hpm.png', bbox_inches="tight")
    return hpm, tth, percentage_correct

def learning_params(folder, animal, day, sec_var='', bin_size=1, to_plot=False):
    '''
    Obtain the learning rate over time, including the fitted linear regression
    model. This function also allows for longer bin sizes.
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in YYMMDD format
        BIN_SIZE: The number of minutes to bin over. Default is one minute
        TO_PLOT: Bool; whether to generate hpm plots or not.
    Outputs:
        HPM: Numpy array; hits per minute
        PERCENTAGE_CORRECT: float; the proportion of hits out of all trials
        REG: The fitted linear regression model
    '''
    
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        ) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fr = f.attrs['fr']
    blen = f.attrs['blen']
    hits = np.asarray(f['hits'])
    miss = np.asarray(f['miss'])
    array_t1 = np.asarray(f['array_t1'])
    array_miss = np.asarray(f['array_miss'])
    trial_end = np.asarray(f['trial_end'])
    trial_start = np.asarray(f['trial_start'])
    percentage_correct = hits.shape[0]/trial_end.shape[0]
    bins = np.arange(0, trial_end[-1]/fr, bin_size*60)
    [hpm, xx] = np.histogram(hits/fr, bins)
    tth = trial_end[array_t1] + 1 - trial_start[array_t1]
    
    if to_plot:
        fig1 = plt.figure()
        ax = fig1.add_subplot(121)
        sns.regplot(xx[1:]/(bin_size*60.0), hpm, label='hits per min')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Hit Rate (hit/min)')
        ax1 = fig1.add_subplot(122)
        sns.regplot(np.arange(tth.shape[0]), tth, label='time to hit')
        ax1.set_xlabel('Reward Trial')
        ax1.set_ylabel('Number of Frames')
        ax1.yaxis.set_label_position('right')
        fig1.savefig(
            folder_path + 'hpm' + str(bin_size) + '.png',
            bbox_inches="tight"
            )

    xx_axis = xx[1:]/(bin_size*60.0)
    xx_axis = np.expand_dims(xx_axis, axis=1)
    reg = LinearRegression().fit(xx_axis, hpm)
    return hpm, percentage_correct, reg

def activity_hits(folder, animal, day, sec_var=''):
    '''
    Function to obtain the activity of neurons time-locked to the trial end.
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in YYMMDD format 
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )
    C = np.asarray(f['C'])

#def neuron_activity(folder, animal, day, sec_var='', to_plot=True):
    # Function to calculate and plot the CAsignals of each neuron
    # separate ensamble/indirect neurons, hits/miss, IT/PT/REST
    # in average (of all trials) + evolution over trials 


def frequency_tuning(folder, animal, day, to_plot=True):
    #function to check if there is any tuning to different frequencies
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_dest = folder +  'analysis/' + animal + '/'
    if not os.path.exists(folder_dest):
        os.makedirs(folder_dest)
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )
    frequency_data = np.asarray(f['frequency'])
    dff_data = np.asarray(f['dff'])
    blen = f.attrs['blen']
    end_trial = f['trial_end'][0] + 1
    
    f = h5py.File(
        folder_dest + 'tuning_' + animal + '_' + day + '_' +
        sec_var + '_freq.hdf5', 'w-'
        )
    
    dff = dff_data[:, blen:]
    frequencies = np.unique(frequency_data)
    num_neurons = dff.shape[0]
    num_frequencies = frequencies.shape[0]
    num_trials = end_trial.shape[0]
    tuning_session = np.ones((num_neurons, num_frequencies)) + np.nan
    tuning_session_er = np.ones((num_neurons, num_frequencies)) + np.nan
    tuning_trial = np.ones((num_neurons, num_trials, num_frequencies)) + np.nan
    ind_ts = np.ones(frequencies.shape[0]) + np.nan 
    ind_tt = np.ones((end_trial, frequencies.shape[0])) + np.nan
    
    # control check, frequency should be as long as dff
    if frequency.shape[0] != dff.shape[1]:
        print('Warning: arrays length mismatch')
        frequency = frequency[:dff.shape[1]]
        
    # to calculate average dff for different frequencies
    for ind,freq  in enumerate(frequencies):
        ind_freq = np.flatnonzero(frequency_data == freq)
        tuning_session[:, ind] = np.nanmean(dff[:, ind_freq], 1)
        tuning_session_er[:, ind] = \
            pd.DataFrame(dff[:, ind_freq]).sem(0).values # std of the mean
        ind_ts[ind] = ind_freq.shape[0]
    
    # to calculate dff for different frequencies over trials
    trials = np.concatenate(([[0], end_trial[:-1]]))
    for tt, trial in enumerate(trials):
        aux_dff = dff[:, trial:end_trial[tt]]
        aux_freq = frequency_data[trial:end_trial[tt]]
        for ind,freq in enumerate(frequencies):
            ind_freq = np.flatnonzero(aux_freq == freq)
            tuning_trial[:, tt, ind] = np.nanmean(aux_dff[:, ind_freq], 1)
            ind_ts[tt, ind] = ind_freq.shape[0]

    f.create_dataset('tuning_session', tuning_session)
    f.create_dataset('tuning_trial', tuning_trial)
    f.create_dataset('ind_ts', ind_ts)
    f.create_dataset('ind_tt', ind_tt)
    
    if to_plot:
        for n in np.arange(dff.shape[0]):
            sm_data = ut.sliding_mean(tuning_session[n,:], window=window_sld)
            sm_error = sliding_mean(tuning_session_er[n,:], window=window_sld)
            plt.fill_between(
                frequency_values, sm_data - sm_error, sm_data + sm_error,
                color="#3F5D7D"
                )
            plt.plot(frequency_values, sm_data, color="white", lw=1)
            plt.savefig(
                folder_dest + day + '_' + n + '_smtuning_curve.png',
                bbox_inches="tight"
                )

# def prediction(folder, animal, day, sec_var='', to_plot=True):
    # Can we predict the result of the hit by looking at the activity of neurons?
    # separate ensamble/indirect neurons, IT/PT/REST


# def tdmodel(folder, animal, day, sec_var='', to_plot=True):
    # Create TD model and compare V(t) and d(T) to activity of neurons
    # IT/PT/REST
