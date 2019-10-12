import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import os
import math
import random
import copy
from scipy.stats import zscore, iqr
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import MultipleLocator
from matplotlib import interactive
from matplotlib.widgets import Slider
from utils_cabmi import *
from utils_loading import encode_to_filename, decode_method_ibi
from utils_bursting import IBI_cv_matrix
from matplotlib import gridspec
from preprocessing import get_roi_type, get_peak_times_over_thres
import seaborn as sns

PALETTE = [sns.color_palette('Blues')[-1], sns.color_palette('Reds')[-1]] # Blue IT, Red PT


def plot_trial_end_all(folder, animal, day,
        trial_type=0, sec_var=''):
    '''
    Plot calcium activity of each neuron from the last 50 frames before the end
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
        sec_var + '_data.hdf5', 'r')

    t_size = [50,30] # 50 frames before and 30 frames after trial end
    time_lock_data = time_lock_activity(f, t_size=t_size)
    time_lock_data = time_lock_data[:,np.array(f['nerden']),:]
    if trial_type == 1:
        array_t1 = np.array(f['array_t1'])
        time_lock_data = time_lock_data[array_t1,:,:]
    elif trial_type == 2:
        array_miss = np.array(f['array_miss'])
        time_lock_data = time_lock_data[array_miss,:,:]
    num_trials, num_neurons, num_frames = time_lock_data.shape
    end_frame = num_frames - t_size[1]

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
        ax.set_ylim((np.min(trial_data)*.9, np.max(trial_data)*1.1))
        fig.canvas.draw_idle()
    trial_slider.on_changed(update)
    neurons_slider.on_changed(update)
    plt.show(block=True)
    
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
    t_size = [30,5]
    time_lock_data = time_lock_activity(f, t_size=t_size)
    if trial_type == 1:
        array_t1 = np.array(f['array_t1'])
        time_lock_data = time_lock_data[array_t1,:,:]
    elif trial_type == 2:
        array_miss = np.array(f['array_miss'])
        time_lock_data = time_lock_data[array_miss,:,:]
    num_trials = time_lock_data.shape[0]
    end_frame = time_lock_data.shape[2] - t_size[1]
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
                axs[irow, icol].set_ylim(min(trial_data)*.9, max(trial_data)*1.1)
                axs[irow, icol].axvline(end_frame, color='r', lw=1.25)
        fig.canvas.draw_idle()
    trial_slider.on_changed(update)
    plt.show(block=True)

def plot_avg_trial_end_ens(folder, animal, day,
        trial_type=0, sec_var=''):
    '''
    Plot the average calcium activity of ensemble neurons from the last
    second before the end of a trial to one second after the trial.
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

    t_size = [10,10]
    time_lock_data = time_lock_activity(f, t_size=t_size)
    if trial_type == 1:
        array_t1 = np.array(f['array_t1'])
        time_lock_data = time_lock_data[array_t1,:,:]
    elif trial_type == 2:
        array_miss = np.array(f['array_miss'])
        time_lock_data = time_lock_data[array_miss,:,:]
    num_trials, num_neurons, num_frames = time_lock_data.shape
    end_frame = num_frames - t_size[1]
    ens_neurons = np.array(f['ens_neur'])

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    plt.subplots_adjust(bottom=0.225, top=0.825)
    for icol in range(2):
        for irow in range(2):
            neuron_idx = int(ens_neurons[icol+2*irow])
            axs[irow, icol].plot(
                np.nanmean(time_lock_data[:,neuron_idx,:], axis=0)
                )
            axs[irow, icol].set_title('Neuron ' + str(neuron_idx))
            axs[irow, icol].axvline(end_frame, color='r', lw=1.25)
        axs[1, icol].set_xlabel("Frame Number")
    for irow in range(2):
        axs[irow, 0].set_ylabel("Calcium Activity")
    trial_type_names = ["All Trials", "Hit Trials", "Miss Trials"]
    fig.suptitle(
        'Average Trial End Activity of Ensemble Neurons:\n' + \
        trial_type_names[trial_type], fontsize='large'
        )
    plt.show(block=True)


def plot_zscore_activity(folder, animal, day, sec_var=''):
    '''
    Plot z-scored calcium activity of each neuron, truncated at 10.0
    at arbitrary values
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in TTMMDD format
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )
    exp_data = np.array(f['C'])
    exp_data = exp_data[np.array(f['nerden']),:]
    num_neurons, num_frames = exp_data.shape
    
    # Z Score (truncated)
    exp_data = zscore(exp_data, axis=1)
    exp_data = np.minimum(exp_data, np.ones(exp_data.shape)*10)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.subplots_adjust(bottom=0.25)
    ax.plot(exp_data[0,:])
    ax.axhline(10, color='r', lw=1.25)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Z-Scored Calcium Activity")
    ax.set_ylim((-3,4.5))
    plt.title("Neuron Activity Over Experiment " + day)
    axcolor = 'lightgoldenrodyellow'
    axneurons = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    neurons_slider = Slider(axneurons, 'Neuron', 0, num_neurons-1, valinit=0)
    def update(val):
        neuron_idx = int(neurons_slider.val)
        trial_data = exp_data[neuron_idx,:]
        for l in ax.get_lines():
            ax.lines.remove(l)
        ax.plot(trial_data) 
        ax.set_ylim((-3,4.5))
        fig.canvas.draw_idle()
    neurons_slider.on_changed(update)
    pdb.set_trace()
    plt.show()


def plot_zscore_rewards(folder, animal, day, sec_var=''):
    '''
    Plot z-scored calcium activity of each neuron, as well as a truncating line
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in TTMMDD format
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )
    t_size = [300,0]
    time_lock_data = time_lock_activity(f, t_size=t_size)
    time_lock_data = time_lock_data[:,np.array(f['nerden']),:]
    array_t1 = np.array(f['array_t1'])
    time_lock_data = time_lock_data[array_t1,:,:]
    num_rewards, num_neurons, num_frames = time_lock_data.shape

    # Z Score (untruncated)
    fluor_vals = zscore(time_lock_data, axis=2)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.subplots_adjust(bottom=0.25)
    ax.plot(fluor_vals[0,0,:])
    ax.set_ylabel("Z-Scored Calcium Activity")
    ax.set_xlabel("Number of Frames")
    ax.axhline(4, color='r', lw=1.25)
    plt.title("Fluorescent Value Histogram for Day " + day)
    axcolor = 'lightgoldenrodyellow'
    axtrials = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
    trial_slider = Slider(axtrials, 'Trial', 0, num_rewards-1, valinit=0)
    axneurons = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    neurons_slider = Slider(axneurons, 'Neuron', 0, num_neurons-1,valinit=0)
    def update(val):
        trial_idx = int(trial_slider.val)
        neuron_idx = int(neurons_slider.val)
        trial_data = fluor_vals[trial_idx,neuron_idx,:]
        for l in ax.get_lines():
            ax.lines.remove(l)
        ax.plot(trial_data)
        ax.axhline(4, color='r', lw=1.25)
        ax.set_ylim((np.min(trial_data)*.9, np.max(trial_data)*1.1))
        fig.canvas.draw_idle()
    trial_slider.on_changed(update)
    neurons_slider.on_changed(update)
    pdb.set_trace()
    plt.show()

def plot_reward_histograms(folder, animal, day, sec_var=''):
    '''
    Plot the histogram of the calcium activity over each reward trial.
    Z-score values are truncated at 4.
    Inputs:
        FOLDER: String; path to folder containing data files
        ANIMAL: String; ID of the animal
        DAY: String; date of the experiment in TTMMDD format
    '''
    folder_path = folder +  'processed/' + animal + '/' + day + '/'
    folder_anal = folder +  'analysis/learning/' + animal + '/' + day + '/'
    f = h5py.File(
        folder_path + 'full_' + animal + '_' + day + '_' +
        sec_var + '_data.hdf5', 'r'
        )
    t_size = [300,0]
    time_lock_data = time_lock_activity(f, t_size=t_size)
    time_lock_data = time_lock_data[:,np.array(f['nerden']),:]
    array_t1 = np.array(f['array_t1'])
    time_lock_data = time_lock_data[array_t1,:,:]
    num_rewards, num_neurons, num_frames = time_lock_data.shape
    
    # Z Score (untruncated)
    fluor_vals = zscore(time_lock_data, axis=2)
    fluor_vals = fluor_vals.reshape(num_rewards, num_neurons*num_frames) 
    fluor_vals = np.minimum(fluor_vals, 4.0)
    trial_data = fluor_vals[0,:]
    trial_data = trial_data[~np.isnan(trial_data)]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.subplots_adjust(bottom=0.25)
    ax.hist(trial_data,100)
    ax.set_xlabel("Z-Scored Calcium Activity")
    ax.set_ylabel("Number of Instances")
    plt.title("Fluorescent Value Histogram for Day " + day)
    axcolor = 'lightgoldenrodyellow'
    axtrial = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    trial_slider = Slider(axtrial, 'Reward Trial', 0, num_rewards-1, valinit=0)
    def update(val):
        trial_idx = int(trial_slider.val)
        trial_data = fluor_vals[trial_idx,:]
        trial_data = trial_data[~np.isnan(trial_data)]
        for l in ax.get_lines():
            ax.lines.remove(l)
        patches = ax.patches
        _ = [p.remove() for p in patches]
        ax.hist(trial_data,100)
        #ax.axhline(10, color='r', lw=1.25)
        #ax.set_ylim((np.min(trial_data)*.9, np.max(trial_data)*1.1))
        fig.canvas.draw_idle()
    trial_slider.on_changed(update)
    pdb.set_trace()
    plt.show()


def plot_peak_psth(folder, animal, day, method, window, tlock=30, eps=True, t=True, w=True):
    # TODO: ADD CV TUNING
    processed = os.path.join(folder, 'CaBMI_analysis/processed/')
    psth = os.path.join(folder, 'bursting/plots/PSTH/')
    windowplot = os.path.join(psth, 'window', animal, day)
    trialplot = os.path.join(psth, 'trial', animal, day)
    D_trial, D_window = get_peak_times_over_thres((processed, animal, day), window, method, tlock=tlock)
    # LABEL NEURON IN FRONT
    with h5py.File(encode_to_filename(processed, animal, day), 'r') as f:
        C = np.array(f['C'])
        hits = np.array(f['array_t1'])
        misses = np.array(f['array_miss'])
        array_start = np.array(f['trial_start'])
        array_end = np.array(f['trial_end'])
    roi_types = get_roi_type(processed, animal, day)
    hp = "psth_window_{}_theta_{}".format(window, decode_method_ibi(method)[1])
    metrics = ('cv', 'cv_ub', 'serr_pc')
    for i in range(C.shape[0]):
        nstr = roi_types[i] + "/" + roi_types[i] + "_" + str(i)
        ntfolder = os.path.join(trialplot, nstr)
        nwfolder = os.path.join(windowplot, nstr)
        fnamet = os.path.join(ntfolder, animal + "_" + day+"_"+hp)
        fnamew = os.path.join(nwfolder, animal + "_" + day+"_"+hp)
        if not os.path.exists(ntfolder):
            os.makedirs(ntfolder)
        if not os.path.exists((nwfolder)):
            os.makedirs(nwfolder)
        # TRIAL
        if t and not os.path.exists(fnamet):
            ibis_hit = [np.diff(D_trial[i][j]) for j in hits]
            ibis_hit_mat = np.full((len(hits), len(max(ibis_hit, key=len))), np.nan)
            for j in range(len(hits)):
                ibis_hit_mat[j, :len(ibis_hit[j])] = ibis_hit[j]
            ibis_miss = [np.diff(D_trial[i][j]) for j in misses]
            ibis_miss_mat = np.full((len(misses), len(max(ibis_miss, key=len))), np.nan)
            for j in range(len(misses)):
                ibis_miss_mat[j, :len(ibis_miss[j])] = ibis_miss[j]
            fig = plt.figure(figsize=(20, 10))
            hitsx = np.concatenate([np.array(D_trial[i][j]) - array_end[j] for j in hits])
            hitsy = np.concatenate([np.full(len(D_trial[i][j]), j + 1) for j in hits])
            missx = np.concatenate([np.array(D_trial[i][j]) - array_end[j] for j in misses])
            missy = np.concatenate([np.full(len(D_trial[i][j]), j + 1) for j in misses])
            gs = gridspec.GridSpec(3, 1, height_ratios=[5, 1, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            plt.subplots_adjust(hspace=0.5)
            ax0.plot(hitsx, hitsy, 'b.', markersize=3)
            ax0.plot(missx, missy, 'k.', markersize=3)
            ax0.plot(array_start-array_end, np.arange(1, len(array_start) + 1), 'r.', markersize=3)
            ax0.legend(['hits', 'miss', 'trial start'])
            ax0.axvline(0, color='r')
            ax0.set_title("PSTH trial")
            ax0.set_xlabel('Time(fr)')
            ax0.set_ylabel('Trial Number')
            blues, reds = sns.color_palette("Blues", 3), sns.color_palette('Reds', 3)
            for j, m in enumerate(metrics):
                ax1.plot(hits, IBI_cv_matrix(ibis_hit_mat, m), c=blues[j], label='hit '+ m)
                ax1.plot(misses, IBI_cv_matrix(ibis_miss_mat, m), c = reds[j], label = 'miss ' + m)
            ax1.set_title("HM cv evolution")
            ax1.set_xlabel("Trial#")

            ax2.plot(hits, [np.nanmean(h) for h in ibis_hit])
            ax2.plot(misses, [np.nanmean(mi) for mi in ibis_miss])
            ax2.legend(['hit', 'miss'])
            ax2.set_xlabel("Trial#")
            ax2.set_ylabel("No. of Frames")
            #TODO: MAYBE ADD A PSTH here
            fig.savefig(fnamet + '.png')
            if eps:
                fig.savefig(fnamet + ".eps")
            plt.close('all')
        # WINDOW
        if w and not os.path.exists(fnamew):
            fig = plt.figure(figsize=(20, 10))
            wlen = len(D_window[i])
            print(wlen)
            slidex = np.concatenate([np.array(D_window[i][j]) - window * j for j in range(wlen)])
            slidey = np.concatenate([np.full(len(D_window[i][j]), j + 1) for j in range(wlen)])
            ibis_slide = [np.diff(D_window[i][j]) for j in range(wlen)]
            ibis_slide_mat = np.full((wlen, len(max(ibis_slide, key=len))), np.nan)
            for j in range(wlen):
                ibis_slide_mat[j, :len(ibis_slide[j])] = ibis_slide[j]
            gs = gridspec.GridSpec(3, 1, height_ratios=[5, 1, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            plt.subplots_adjust(hspace=0.5)
            ax0.plot(slidex, slidey, 'k.', markersize=3)
            ax0.axvline(0, color='r')
            ax0.set_title("PSTH window")
            ax0.set_xlabel('Time(fr)')
            ax0.set_ylabel('Slide')
            for m in metrics:
                ax1.plot(np.arange(wlen), IBI_cv_matrix(ibis_slide_mat, m))
            ax1.legend(metrics)
            ax1.set_title("CV evolution")
            ax1.set_xlabel("Window")
            print(len(ibis_slide))
            ax2.hist(ibis_slide, density=True, color=sns.color_palette("Blues", wlen))
            ax2.legend(np.arange(wlen))
            ax2.set_title("IBI dist evolution")
            fig.savefig(fnamew + '.png')
            if eps:
                fig.savefig(fnamew + ".eps")
            plt.close("all")



def best_nbins(data):
    try:
        binsize = 2 * iqr(data, nan_policy='omit') * len(data) ** (-1 / 3)
    except:
        binsize = 3.49 * np.nanstd(data) * len(data) ** (-1 / 3)
    if binsize == 0:
        return max(len(data) // 100, len(data) // 10, 1)
    nbins = min(int((np.max(data) - np.min(data)) / binsize) + 1, 1000)
    return nbins if nbins < len(data) / 4 else max(len(data) // 20, 1)

if __name__ == '__main__':
    home = "/home/user/"
    processed = os.path.join(home, "CaBMI_analysis/processed/")
    #inputs = [(home, "IT2", "181002"), (home, "IT2", "190115"), (home, "IT5", "190206"), (home, "PT7", "190114")]
    #inputs = [(home, "IT2", "181002"), (home, "PT7", "190114")]
    for m in (2, 11):
        for window in (9000, 6000):
            # for inp in inputs:
                # plot_peak_psth(*inp, m, window)
            #for animal in os.listdir(processed):
            for animal in ['IT2', 'PT7']:
                if animal.find('IT') == -1 and animal.find('PT') == -1:
                    continue
                animal_path = os.path.join(processed, animal)
                sdays = sorted(os.listdir(animal_path))
                for day in sdays:
                    plot_peak_psth(home, animal, day, m, window)
