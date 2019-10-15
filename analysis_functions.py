
# 2p to record the neuronal data
# Scanimage defines the format of data

__author__ = 'Nuria'

# C++ extension for data acquisition

import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sklearn
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
from utils_loading import get_PTIT_over_days, parse_group_dict, encode_to_filename
PALETTE = [sns.color_palette('Blues')[-1], sns.color_palette('Reds')[-1]] # Blue IT, Red PT
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


def learning_params(
    folder, animal, day, sec_var='', bin_size=1,
    to_plot=None, end_bin=None, reg=False, dropend=True):
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
    
    folder_path = folder +  'CaBMI_analysis/processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        ) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fr = f.attrs['fr']
    blen = f.attrs['blen']
    blen_min = blen//600
    hits = np.asarray(f['hits'])
    miss = np.asarray(f['miss'])
    array_t1 = np.asarray(f['array_t1'])
    array_miss = np.asarray(f['array_miss'])
    trial_end = np.asarray(f['trial_end'])
    trial_start = np.asarray(f['trial_start'])
    #percentage_correct = hits.shape[0]/trial_end.shape[0]
    bigbin = bin_size*60
    if dropend:
        ebin = trial_end[-1]/fr + 1
    else:
        ebin = int(np.ceil(trial_end[-1]/fr / bigbin)) * bigbin + 1
    bins = np.arange(0, ebin, bigbin)
    [hpm, xx] = np.histogram(hits/fr, bins)
    [mpm, _] = np.histogram(miss/fr, bins)
    hpm = hpm[blen_min//bin_size:]
    mpm = mpm[blen_min//bin_size:]
    percentage_correct = hpm / (hpm+mpm)
    # TODO: CONSIDER SETTING THRESHOLD USING ONLY THE WHOLE WINDOWS
    if not dropend:
        last_binsize = (trial_end[-1] / fr - bins[-2])
        hpm[-1] *= bigbin / last_binsize
    xx = -1*(xx[blen_min//bin_size]) + xx[blen_min//bin_size:]
    xx = xx[1:]
    hpm = hpm / bin_size
    if end_bin is not None:
        end_frame = end_bin//bin_size + 1
        hpm = hpm[:end_frame]
        xx = xx[:end_frame]
    tth = trial_end[array_t1] + 1 - trial_start[array_t1]
    
    if to_plot is not None:
        maxHit, hitIT_salient, hitPT_salient, hit_all_salient, hit_all_average, pcIT_salient, pcPT_salient, pc_all_salient, pc_all_average= \
            to_plot
        out = os.path.join(folder, 'learning/plots/evolution_{}/'.format(bin_size))
        if not os.path.exists(out):
            os.makedirs(out)
        fig1 = plt.figure(figsize=(15, 5))
        ax = fig1.add_subplot(131)
        ax.axhline(hitIT_salient, color=PALETTE[0], lw=1.25, label='IT salience')
        ax.axhline(hitPT_salient, color=PALETTE[1], lw=1.25, label='PT salience')
        ax.axhline(hit_all_salient, color='yellow', lw=1.25, label='All salience')
        ax.legend()
        sns.regplot(xx/60, hpm, label='hits per min')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Hit Rate (hit/min)')
        ax.set_title('Hit Rate Evolution')
        if maxHit is not None:
            ax.set_ylim((-maxHit / 20, maxHit * 21 / 20))


        # TODO: ADD CHANCE LEVEL
        ax1 = fig1.add_subplot(132)
        sns.regplot(np.arange(tth.shape[0]), tth / fr, label='time to hit')
        ax1.set_xlabel('Reward Trial')
        ax1.set_ylabel('Hit Time (second)')
        ax1.set_title('Hit Time Evolution')
        ax1.set_ylim((-1.5, 31.5))
        ax1.legend()
        ax2 = fig1.add_subplot(133)
        ax2.axhline(pcIT_salient*100, color=PALETTE[0], lw=1.25, label='IT salience')
        ax2.axhline(pcPT_salient*100, color=PALETTE[1], lw=1.25, label='PT salience')
        ax2.axhline(pc_all_salient*100, color='yellow', lw=1.25, label='All salience')
        sns.regplot(xx/60, percentage_correct * 100, label='percentage correct')
        ax2.set_xlabel('Minutes')
        ax2.set_ylabel('Percentage Correct (%)')
        ax2.set_title('Percentage Correct Evolution')
        ax2.set_ylim((-5, 105))
        ax2.legend()
        learner_pc = -1 # Good learner: 2, Average Learner: 1, Bad Learner: 0, Non Learner: -1
        learner_hpm = -1
        pcmax = np.nanmax(percentage_correct)
        nmax = np.nanmax(hpm)
        if pcmax >= pc_all_salient:
            learner_pc = 2
        elif pcmax >= pc_all_average:
            learner_pc = 1
        elif pcmax >= 0.3:
            learner_pc = 0

        if nmax >= hit_all_salient:
            learner_hpm = 2
        elif nmax >= hit_all_average:
            learner_hpm = 1
        elif nmax >= 1: # 1 as learning criteria
            learner_hpm = 0
        # HERE DEFINE LEARNING VALUE [0.7, 0.3]
        fig1.savefig(
            out + "L_{}_evolution_{}.png".format(int(pcmax>=0.7)+int(pcmax>=0.3),animal, day, bin_size),
            bbox_inches="tight"
            )
        plt.close('all')

    xx_axis = xx/bigbin
    xx_axis = np.expand_dims(xx_axis, axis=1)
    return xx_axis, hpm, percentage_correct, LinearRegression().fit(xx_axis, hpm) if reg else None

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


def feature_select(folder, animal, day, sec_var='', sec_bin=[30, 0], step=5,
    score_min=0.9, toplot=True):
    """Function to select neurons that are relevant to the task, it goes iteratively through a
    temporal vector defined by sec_bin with bins of step
    folder (str): folder where the input/output is/will be stored 
    animal (str): animal to be analyzed
    day (str): day to be analyzed
    sec_var (str): secondary variable to identify type of experiment
    sec_bin (tuple): frames before and after exp.
    score_min: minimum value to consider the feature selection
    return 
    sel_neurs: an array of the fitted RFECV models
    neur : index of neurons to consider
    and number of neurons selected
    """
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    f = h5py.File(folder_path + 'full_' + animal + '_' + day + '__data.hdf5','r')
 
    # obtain C divided by trial
    C_ord = ut.time_lock_activity(f, sec_bin)
    array_t1 = np.asarray(f['array_t1'])
    
    # trial label 
    classif = np.zeros(C_ord.shape[0])
    classif[array_t1] = 1
    
    # steps to run throuhg C_ord
    steps = np.arange(0, np.nansum(sec_bin) + step, step)
        
    # prepare models
    lr = sklearn.linear_model.LogisticRegression()
    selector = sklearn.feature_selection.RFECV(
        lr, step=1, cv=5, scoring='balanced_accuracy'
        )
    
    # init neur
    neur = np.zeros(C_ord.shape[1]).astype('bool')
    succesful_steps = np.zeros(steps.shape[0])
    sel_neurs = []
    
    # run models through each step
    for ind, s in enumerate(steps[1::]):
        data = np.nansum(C_ord[:, :, steps[ind]:s], 2)
        sel_neur = selector.fit(data, classif)
        # if the info is good keep the neurons
        if max(sel_neur.grid_scores_) > score_min:
            neur = np.logical_or(sel_neur.support_, neur)
        succesful_steps[ind] = max(sel_neur.grid_scores_)
        if toplot:
            plt.plot(sel_neur.grid_scores_, label=str(ind))
        sel_neurs.append(sel_neur)
    
    if toplot:
        plt.legend()
    
    return sel_neurs, neur, np.sum(neur)


# def tdmodel(folder, animal, day, sec_var='', to_plot=True):
    # Create TD model and compare V(t) and d(T) to activity of neurons
    # IT/PT/REST


def C_activity_tuning(folder, groups, window=3000, zcap=None, test=True):
    # TODO: ADD OPTION TO PASS IN A LIST OF METHODS FOR COMPARING THE PLOTS!
    """Calculates Peak Timing and Stores them in csvs for all animal sessions in groups located in folder."""
    processed = os.path.join(folder, 'CaBMI_analysis/processed')

    if groups == '*':
        all_files = get_PTIT_over_days(processed)
    else:
        all_files = {g: parse_group_dict(processed, groups[g], g) for g in groups.keys()}
    print(all_files)
    hp = 'window_{}_zcap'.format(window, zcap)
    resW = {n: [] for n in ['window', 'roi_type', 'N', 'mZC', 'group', 'animal',
       'date', 'session']}
    for group in all_files:
        group_dict = all_files[group]
        for animal in group_dict:
            for i, day in sorted(group_dict[animal]):

                hf = encode_to_filename(processed, animal, day)
                if not os.path.exists(hf):
                    print("Not found:, ", hf)
                print(animal, day)
                with h5py.File(hf, 'r') as fp:
                    C = np.array(fp['C'])
                    # TODO: maybe include trial activity tuning
                    #array_hit, array_miss = np.array(fp['array_t1']), np.array(fp['array_miss'])
                    ens_neur = np.array(fp['ens_neur'])
                    e2_neur = ens_neur[fp['e2_neur']] if 'e2_neur' in fp else None
                    redlabel, nerden = np.array(fp['redlabel']), np.array(fp['nerden'])

                zscoreC = zscore(C, axis=1)
                if zcap is not None:
                    zscoreC = np.minimum(zscoreC, np.full_like(C, zcap))
                N = C.shape[0]
                nsessions = C.shape[1] // window
                remainder = C.shape[1] - nsessions * window
                Cfirst = (zscoreC[:, :nsessions * window]).reshape((C.shape[0], nsessions, window), order='C')
                avg_C = np.nanmean(Cfirst, axis=1)
                if remainder > 0:
                    Cremain = zscoreC[:, nsessions * window:]
                    avg_C = np.concatenate((avg_C, np.nanmean(Cremain, axis=1)), axis=1)

                # N, sw, st = probeW.shape[0], probeW.shape[1], probeT.shape[1]
                # ROI_type
                sw = nsessions + remainder
                rois = np.full(N, "D", dtype="U2")
                rois[nerden & ~redlabel] = 'IG'
                rois[nerden & redlabel] = 'IR'
                if e2_neur is not None:
                    rois[ens_neur] = 'E1'
                    rois[e2_neur] = 'E2'
                else:
                    rois[ens_neur] = 'E'
                # DF Window

                resW['window'].append(np.tile(np.arange(sw), N))
                resW['roi_type'].append(np.repeat(rois, sw))
                resW['N'].append(np.repeat(np.arange(N), sw))
                resW['mZC'].append(avg_C.ravel(order=C))
                resW['group'] = np.full(N * sw, animal[:2])
                resW['animal'] = np.full(N * sw, animal)
                resW['day'] = np.full(N * sw, animal)
                resW['session'] = np.full(N*sw, i+1)

                # # DF TRIAL
                # trials = np.arange(1, st + 1)
                # tempm = trials[array_miss]
                # temph = trials[array_hit]
                # misses = np.empty_like(tempm)
                # hits = np.empty_like(temph)
                # sortedm = np.argsort(tempm)
                # sortedh = np.argsort(temph)
                # for i in range(len(sortedm)):
                #     misses[sortedm[i]] = -i - 1
                # for i in range(len(sortedh)):
                #     hits[sortedh[i]] = i + 1
                # hm_trial = np.empty_like(trials)
                # hm_trial[array_hit] = hits
                # hm_trial[array_miss] = misses
                # # trials[array_miss] = -trials[array_miss]
                # # awhere = np.where(trials < 0)[0]
                # # assert np.array_equal(awhere, array_miss), "NOt alligned {} {}".format(awhere, array_miss)
                # resT['trial'] = np.tile(trials, N)  # 1-indexed
                # resT['HM_trial'] = np.tile(hm_trial, N)  # 1-indexed
                # resT['roi_type'] = np.repeat(rois, st)
                # resT['N'] = np.repeat(np.arange(N), st)
                # for k in mets_window:
                #     resW[k] = mets_window[k].ravel(order='C')
                #     resT[k] = mets_trial[k].ravel(order='C')

                # print(N, sw, st)
                # def debug_print(res):
                #     for k in res.keys():
                #         print(k, res[k].shape)
                # debug_print(resW)
                # debug_print(resT)
                # df_trial = pd.DataFrame(resT)
    for k in resW:
        resW[k] = np.concatenate(resW[k])
    df_window = pd.DataFrame.from_dict(resW)
    if test:
        # testing = os.path.join(path, 'test.csv')
        # if os.path.exists(testing):
        #     print('Deleting', testing)
        #     os.remove(testing)
        df_window.to_csv(os.path.join(processed, 'C_activity_tuning_window_{}.csv'.format(hp)),
                         index=False)
        # df_trial.to_csv(os.path.join(path, '{}_{}_trial_test.csv'.format(animal, day)),
        #                 index=False)
    # else:
    #     df_window.to_hdf(fname, 'df_window')
    #     # df_trial.to_hdf(fname, 'df_trial')
    return df_window

if __name__ == '__main__':
    home = "/home/user/"
    processed = os.path.join(home, "CaBMI_analysis/processed/")
    C_activity_tuning(home, '*')

