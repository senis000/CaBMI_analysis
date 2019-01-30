
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
from matplotlib import interactive
import utils_cabmi as ut

interactive(True)

def learning(folder, animal, day, sec_var='', to_plot=True):
    '''
    Function to plot the learning rate over time.
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in YYMMDD format
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

    if trial_end.shape[0] == trial_start.shape[0]:
        diff = trial_end - trial_start
    else:
        diff = trial_end - trial_start[:trial_end.shape[0]]
    tth = trial_end[array_t1] - trial_start[array_t1]
    
    if to_plot:
        fig1 = plt.figure()
        ax = fig1.add_subplot(121)
        sns.regplot(
                xx[1:-np.where(xx>blen)[0][0]]/60,
                hpm[np.where(xx>blen)[0][0]:],
                label='hits per min'
                )
        ax.set_xlabel('hpm')
        ax1 = fig1.add_subplot(122)
        sns.regplot(np.arange(tth.shape[0]), tth, label='time to hit')
        ax1.set_xlabel('tth')
        fig1.savefig(folder_path + 'hpm.png', bbox_inches="tight")
    return hpm, tth, percentage_correct


def activity_hits(folder, animal, day, sec_var=''):
    '''
    Function to obtain the activity of neurons time-locked to the trial end.
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    f = h5py.File(folder_path + 'full_' + animal + '_' + day + '_data.hdf5', 'r')
    C = np.asarray(f['C'])

def plot_trial_end_all(folder, animal, day,
        trial_type=0, sec_var=''):
    '''
    Plot calcium activity of each neuron from the last 5 seconds before the end
    of a trial to 3 seconds after the trial. The user can choose whether to plot
    all trials, hit trials, or miss trials
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in TTMMDD format
        TRIAL_TYPE: an integer from [0,1,2]. 0 indicates all trials,
            1 indicates hit trials, 2 indicates miss trials.
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )

    t_size = [30,3]
    tbin = 10
    time_lock_data = time_lock_activity(f, t_size, tbin)
    end_frame = time_lock_data.shape[2] - tbin*t_size[1]
    time_lock_data = time_lock_data[:,:,end_frame - tbin*5:]
    num_trials, num_neurons, num_frames = time_lock_data.shape
    end_frame = num_frames - tbin*t_size[1]

    # Sliding plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.subplots_adjust(bottom=0.25)
    ax.plot(time_lock_data[0,0,:])
    ax.axvline(end_frame, color='r', lw=1.25)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Calcium Activity")
    trial_type_names = ["All Trials", "Hit Trials", "Miss Trials"]
    plt.title(
        'Trial End Activity of Neurons:\n'+trial_type_names[trial_type],
        fontsize='large'
        )

    axcolor = 'lightgoldenrodyellow'
    axtrials = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
    trial_slider = Slider(axtrials, 'Trial', 0, num_trials-1, valinit=0)
    axneurons = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    neurons_slider = Slider(axneurons, 'Neuron', 0, num_neurons-1,valinit=0)
    def update(val):
        trial = int(trial_slider.val)
        neuron_idx = int(neurons_slider.val)
        trial_data = time_lock_data[trial,neuron_idx,:]
        for l in ax.get_lines():
            ax.lines.remove(l)
        ax.plot(trial_data)
        ax.axvline(end_frame, color='r', lw=1.25)
        ax.set_ylim((np.min(trial_data), np.max(trial_data)))
        fig.canvas.draw_idle()
    trial_slider.on_changed(update)
    neurons_slider.on_changed(update)
    plt.show()
    
def plot_trial_end_ens(folder, animal, day,
        trial_type=0, sec_var=''):
    '''
    Plot calcium activity of ensemble neurons from the last 5 seconds before the
    end of a trial to 3 seconds after the trial.The slider allows the user to
    view different trials. The user can choose whether to plot all trials,
    hit trials, or miss trials.
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in TTMMDD format
        TRIAL_TYPE: an integer from [0,1,2]. 0 indicates all trials,
            1 indicates hit trials, 2 indicates miss trials.
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )

    t_size = [30,3]
    tbin = 10
    time_lock_data = time_lock_activity(f, t_size, tbin)
    end_frame = time_lock_data.shape[2] - tbin*t_size[1]
    time_lock_data = time_lock_data[:,:,end_frame - tbin*5:]
    num_trials, num_neurons, num_frames = time_lock_data.shape
    end_frame = num_frames - tbin*t_size[1]
    ens_neurons = np.array(f['ens_neur'])

    # Sliding plot
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    plt.subplots_adjust(bottom=0.225, top=0.825)
    for icol in range(2):
        for irow in range(2):
            neuron_idx = int(ens_neurons[icol+2*irow])
            axs[irow, icol].plot(time_lock_data[0,neuron_idx,:])
            axs[irow, icol].set_title('Neuron ' + str(neuron_idx))
            axs[irow, icol].axvline(end_frame, color='r', lw=1.25)
        axs[1, icol].set_xlabel("Frame Number")
    for irow in range(2):
        axs[irow, 0].set_ylabel("Calcium Activity")
    trial_type_names = ["All Trials", "Hit Trials", "Miss Trials"]
    fig.suptitle(
        'Trial End Activity of Ensemble Neurons:\n'+trial_type_names[trial_type],
        fontsize='large'
        )

    axcolor = 'lightgoldenrodyellow'
    axtrials = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
    trial_slider = Slider(axtrials, 'Trial', 0, num_trials-1, valinit=0)
    def update(val):
        trial = int(trial_slider.val)
        for icol in range(2):
            for irow in range(2):
                neuron_idx = int(ens_neurons[icol+2*irow])
                trial_data = time_lock_data[trial,neuron_idx,:]
                for l in axs[irow, icol].get_lines():
                    axs[irow, icol].lines.remove(l)
                axs[irow, icol].plot(trial_data)
                axs[irow, icol].axvline(end_frame, color='r', lw=1.25)
        fig.canvas.draw_idle()
    trial_slider.on_changed(update)
    plt.show()

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
    f = h5py.File(folder_path + 'full_' + animal + '_' + day + '_data.hdf5', 'r')
    frequency_data = np.asarray(f['frequency'])
    dff_data = np.asarray(f['dff'])
    blen = f.attrs['blen']
    end_trial = f['trial_end'][0]
    
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

def gte_whole_experiment(folder, animal, day,
        exp_name, frame_size, frame_step, parameters,
        sec_var='', to_plot=True, pickle_results=True):
    '''
    Function to run GTE over all neurons, over the whole experiment.
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in YYMMDD format
        EXP_NAME: String; directory name within the GTE experiments directory.
        FRAME_SIZE: Integer; number of frames to process in GTE
        FRAME_STEP: Integer; number of frames for each step through the signal.
        PARAMETERS: Dictionary; parameters for GTE
        TO_PLOT: Boolean; whether or not to call the visualization script
        PICKLE_RESULTS: Boolean; whether or not to save the results matrix
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )
    exp_data = np.array(f['C'])
    neuron_locations = np.array(f['com_cm'])
    num_neurons = exp_data.shape[0]
    num_frames = exp_data.shape[1]

    control_file_names, output_file_names = create_gte_input_files(\
        exp_name, exp_data, parameters,
        frame_size, frame_step
        )
    results = run_gte(control_file_names, output_file_names, pickle_results)
    if to_plot:
        visualize_gte_results(results, neuron_locations)
    delete_gte_files(exp_name, delete_output=False)

