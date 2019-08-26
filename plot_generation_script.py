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
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import ttest_ind
from sklearn import preprocessing
import networkx as nx
from networkx.algorithms import community
from ExpGTE import ExpGTE
from utils_cabmi import *
from plotting_functions import *
from analysis_functions import *
from utils_gte import *
from utils_clustering import *
from clustering_functions import *
from plot_rewardend import *
from plot_base_end import *
from sklearn.linear_model import LinearRegression

sns.palplot(sns.color_palette("Set2"))


def plot_all_sessions_hpm(sharey=False):
    folder = '/home/user/'
    processed = os.path.join(folder, 'CaBMI_analysis/processed/')
    binsizes = [1, 3, 5]
    print("PRELIM")
    maxHit = 0
    IT_hit, PT_hit = OnlineNormalEstimator(algor='moment'), OnlineNormalEstimator(algor='moment')
    IT_pc, PT_pc = OnlineNormalEstimator(algor='moment'), OnlineNormalEstimator(algor='moment')
    for b in binsizes:
        print("BIN {}".format(b))
        for animal in os.listdir(processed):
            animal_path = processed + animal + '/'
            if not os.path.isdir(animal_path):
                continue
            if not (animal.startswith('IT') or animal.startswith('PT')):
                continue
            days = os.listdir(animal_path)
            days.sort()
            for day in days:
                if day.isnumeric():
                    print(animal, day)
                    _, hpm, pc, _, = learning_params(folder, animal, day, bin_size=b)
                    if animal.startswith('IT'):
                        IT_hit.handle(hpm)
                        IT_pc.handle(pc)
                    else:
                        PT_hit.handle(hpm)
                        PT_pc.handle(pc)
                    maxHit = max(maxHit, np.nanmax(hpm))


    allhitm, allhits = OnlineNormalEstimator.join(IT_hit, PT_hit)
    tHitIT, tHitPT, tHitAll = IT_hit.mean() + IT_hit.std(), PT_hit.mean() + PT_hit.std(), allhitm + allhits
    allPCm, allPCs = OnlineNormalEstimator.join(IT_pc, PT_pc)
    tPCIT, tPCPT, tPCAll = IT_pc.mean() + IT_pc.std(), PT_pc.mean() + PT_pc.std(), allPCm + allPCs

    if not sharey:
        opt = (None, tHitIT, tHitPT, tHitAll, tPCIT, tPCPT, tPCAll)
    else:
        opt = (maxHit, tHitIT, tHitPT, tHitAll, tPCIT, tPCPT, tPCAll)

    print("PLOT", maxHit)
    for b in binsizes:
        print("BIN {}".format(b))
        for animal in os.listdir(processed):
            animal_path = processed + animal + '/'
            if not os.path.isdir(animal_path):
                continue
            if not (animal.startswith('IT') or animal.startswith('PT')):
                continue
            days = os.listdir(animal_path)
            days.sort()
            for day in days:
                if day.isnumeric():
                    print(animal, day)
                    learning_params(folder, animal, day, bin_size=b, to_plot=opt)




def plot_itpt_hpm(bin_size=1, plotting_bin_size=10, num_minutes=200,
    first_N_experiments=100):
    """
    Aggregates hits per minute across all IT and PT animals. Performs regression
    on the resulting data, and returns the p-value of how different linear
    regression between the two animals are.
    """

    # Getting all hits per minute arrays
    IT_train = []
    IT_target = []
    PT_train = []
    PT_target = []
    num_it = 0
    num_pt = 0
    pattern = 'full_(IT|PT)\d+_(\d+)_.*\.hdf5'
    folder = './processed/'

    for animal in os.listdir(folder):
        animal_path = folder + animal + '/'
        if not os.path.isdir(animal_path):
            continue
        if not (animal.startswith('IT') or animal.startswith('PT')):
            continue
        hpm_arrays = []
        days = os.listdir(animal_path)
        days.sort()
        days = days[:first_N_experiments]
        for day in days:
            day_path = animal_path + day + '/'
            if not os.path.isdir(day_path):
                continue
            for file_name in os.listdir(day_path):
                if file_name.endswith(".hdf5"):
                    result = re.search(pattern, file_name)
                    if not result:
                        continue
                    try:
                        xs, hpm, _, _ =\
                            learning_params(
                                './', animal, day,
                                bin_size=bin_size
                                )
                        xs = xs*bin_size
                    except:
                        continue
                    if animal.startswith('IT'):
                        for idx, x_val in enumerate(xs):
                            if x_val <= num_minutes:
                                IT_train.append(x_val)
                                IT_target.append(hpm[idx])
                        num_it += 1
                    else:
                        for idx, x_val in enumerate(xs):
                            if x_val <= num_minutes:
                                PT_train.append(x_val)
                                PT_target.append(hpm[idx])
                        num_pt += 1

    # Collect data
    IT_train = np.array(IT_train).squeeze()
    IT_target = np.array(IT_target)
    PT_train = np.array(PT_train).squeeze()
    PT_target = np.array(PT_target)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # p-val for linear regression slope similarity
    p_val = linreg_pval(IT_train, IT_target, PT_train, PT_target)
    print("Comparing linear regression slopes of IT and PT:")
    print("p-val = " + str(p_val))

    # Some options:
    # Order 1, Order 2, Logx True
    sns.regplot(
        IT_train, IT_target,
        x_bins=plotting_bin_size,
        color='lightseagreen', label='IT (%d Experiments)'%num_it
        )
    sns.regplot(
        PT_train, PT_target,
        x_bins=plotting_bin_size,
        color='coral', label='PT (%d Experiments)'%num_pt
        )
    ax.set_ylabel('Number of Hits')
    ax.set_xlabel('Minutes into the Experiment')
    plt.title('Hits/%d-min of All Experiments'%bin_size)
    plt.legend()
    plt.show(block=True)

def plot_itpt_hpm_depth(bin_size=1, plotting_bin_size=10, num_minutes=200,
    first_N_experiments=100):
    """
    Aggregates hits per minute across all IT and PT animals. Performs regression
    on the resulting data, and returns the p-value of how different linear
    regression between the two animals are.
    """

    # Getting all hits per minute arrays
    ITshallow_train = []
    ITshallow_target = []
    ITdeep_train = []
    ITdeep_target = []
    PT_train = []
    PT_target = []
    num_itshallow = 0
    num_itdeep = 0
    num_pt = 0
    pattern = 'full_(IT|PT)\d+_(\d+)_.*\.hdf5'
    folder = './processed/'

    for animal in os.listdir(folder):
        animal_path = folder + animal + '/'
        if not os.path.isdir(animal_path):
            continue
        if not (animal.startswith('IT') or animal.startswith('PT')):
            continue
        hpm_arrays = []
        days = os.listdir(animal_path)
        days.sort()
        days = days[:first_N_experiments]
        for day in days:
            day_path = animal_path + day + '/'
            if not os.path.isdir(day_path):
                continue
            for file_name in os.listdir(day_path):
                if file_name.endswith(".hdf5"):
                    result = re.search(pattern, file_name)
                    if not result:
                        continue
                    try:
                        xs, hpm, _, _ =\
                            learning_params(
                                './', animal, day,
                                bin_size=bin_size
                                )
                        f = h5py.File(day_path + file_name, 'r')
                        com_cm = np.array(f['com_cm'])
                        e2_indices = np.array(f['e2_neur'])
                        ens_neur = np.array(f['ens_neur'])
                        e2_neur = ens_neur[e2_indices]
                        e2_depths = np.mean(com_cm[e2_neur,2])
                        xs = xs*bin_size
                    except:
                        continue
                    if animal.startswith('IT'):
                        shallow_thresh = 250
                        deep_thresh = 350
                        for idx, x_val in enumerate(xs):
                            if x_val <= num_minutes:
                                if e2_depths < shallow_thresh:
                                    ITshallow_train.append(x_val)
                                    ITshallow_target.append(hpm[idx])
                                elif e2_depths > deep_thresh:
                                    ITdeep_train.append(x_val)
                                    ITdeep_target.append(hpm[idx])
                        if e2_depths < shallow_thresh:
                            num_itshallow += 1
                        elif e2_depths > deep_thresh:
                            num_itdeep += 1
                    else:
                        for idx, x_val in enumerate(xs):
                            if x_val <= num_minutes:
                                PT_train.append(x_val)
                                PT_target.append(hpm[idx])
                        num_pt += 1

    # Collect data
    ITshallow_train = np.array(ITshallow_train).squeeze()
    ITshallow_target = np.array(ITshallow_target)
    ITdeep_train = np.array(ITdeep_train).squeeze()
    ITdeep_target = np.array(ITdeep_target)
    PT_train = np.array(PT_train).squeeze()
    PT_target = np.array(PT_target)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # p-val for linear regression slope similarity
    p_val = linreg_pval(
        ITshallow_train, ITshallow_target,
        ITdeep_train, ITdeep_target
    )
    print("Comparing linear regression slopes of IT and PT:")
    print("p-val = " + str(p_val))

    # Some options:
    # Order 1, Order 2, Logx True
    sns.regplot(
        ITshallow_train, ITshallow_target,
        x_bins=plotting_bin_size,
        color='forestgreen', label='IT shallow (%d Experiments)'%num_itshallow
        )
    sns.regplot(
        ITdeep_train, ITdeep_target,
        x_bins=plotting_bin_size,
        color='cornflowerblue', label='IT deep (%d Experiments)'%num_itdeep
        )
    sns.regplot(
        PT_train, PT_target,
        x_bins=plotting_bin_size,
        color='coral', label='PT (%d Experiments)'%num_pt
        )
    ax.set_ylabel('Number of Hits')
    ax.set_xlabel('Minutes into the Experiment')
    plt.title('Hits/%d-min of All Experiments'%bin_size)
    plt.legend()
    plt.show(block=True)

def linreg_pval(train1, target1, train2, target2):
    """
    Runs linear regression over both sets of data and returns a p-value
    describing how different the slopes of the two regressions are.
    Follows the procedure from:
    https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/
    PASS/Tests_for_the_Difference_Between_Two_Linear_Regression_Slopes.pdf
    """

    train1 = train1.reshape(-1, 1)
    target1 = target1.reshape(-1, 1)
    train2 = train2.reshape(-1, 1)
    taraget2 = target2.reshape(-1, 1)
    reg1 = LinearRegression().fit(train1, target1) # Regression models
    reg2 = LinearRegression().fit(train2, target2)
    pred1 = reg1.predict(train1) # Predictions of training data
    pred2 = reg2.predict(train2)
    sse1 = np.sum(np.square(pred1 - target1)) # Sum of squared errors
    sse2 = np.sum(np.square(pred2 - target2))
    n1 = target1.size # Number of samples
    n2 = target2.size
    m = n1/n2
    v = n1 + n2 - 4 # Degrees of freedom
    mean_sse = (1/v)*(sse1+sse2) # Mean total sum of squared errors
    var1 = np.var(train1) # Variance of training data
    var2 = np.var(train2)
    Sr_sqd = mean_sse*((1/(m*var1))+(1/var2)) # Idk what this is
    slope1 = reg1.coef_[0] # The predicted slopes of each line
    slope2 = reg2.coef_[0]
    D = (slope1 - slope2)*(np.sqrt(n2))/np.sqrt(Sr_sqd) # The t-statistic
    pval = stats.t.sf(np.abs(D), v)*2 # Two-sided p value
    return pval

def analyze_feature_selection(rfecv_accuracy_threshold=.7):
    '''
    Analyzes the results of running feature selection. We may want to see how
    the depth of E2 neurons influences the depth of feature-selected neurons
    or (in the case of IT animals) feature-selected dendrites.
    '''

    pattern = 'full_(IT|PT)\d+_(\d+)_.*\.hdf5'
    folder = './processed/'

    # We initialize arrays for plotting. See the lines where REGPLOT is used
    # to get a better idea of how these arrays are used.
    it_sigdepths = [] # Depths of significant neurons/dendrites
    it_e2location = [] # Mean depth of E2 neurons
    it_den_sigdepths = [] # Depths of significant dendrites
    it_den_e2location = [] # Mean depth of E2 neurons
    pt_sigdepths = []
    pt_e2location = []
    it_exp_counter = 0 # Number of experiments we looked at
    pt_exp_counter = 0

    for animal in os.listdir(folder):
        animal_path = folder + animal + '/'
        if not os.path.isdir(animal_path):
            continue
        if not (animal.startswith('IT') or animal.startswith('PT')):
            continue
        for day in os.listdir(animal_path):
            day_path = animal_path + day + '/'
            if not os.path.isdir(day_path):
                continue
            path_to_models = './processed/' + animal + '/' + day + '/rfecv_model.p'
            try:
                with open(path_to_models, 'rb') as f:
                    rfecv_models = pickle.load(f)
                f = h5py.File(
                    day_path + "full_" + animal + "_" + day + "__data.hdf5"
                    )
            except:
                continue

            # This is a boolean mask over neurons/dendrites considered significant
            selected_features = np.zeros(
                rfecv_models[0].support_.size
                ).astype('bool')

            # For one experiment, we iterate over each time-shifted RFECV model
            for rfecv_model in rfecv_models:
                # We collect the results of RFECV models that exceeded some
                # accuracy threshold
                if max(rfecv_model.grid_scores_) > rfecv_accuracy_threshold:
                    selected_features = np.logical_or(
                        rfecv_model.support_, selected_features 
                    )


            # Now, we extract the depth information of selected_features.
            # We put all this information into the arrays declared at the
            # beginning of this function. These arrays can be fed into
            # plotting functions directly.
            # Selected_features will refer to neurons and dendrites
            # Selected_neurons will only refer to neurons. 
            # Selected_dendrites will only refer to dendrites.
            exp_sigdepths = [] 
            exp_e2location = []
            com_cm = np.array(f['com_cm'])
            ens_neur = np.array(f['ens_neur'])
            nerden = np.array(f['nerden'])
            den_mask = np.logical_not(nerden)

            # Process E2 neurons to get their depth infomration
            e2_neur = np.array(f['e2_neur'])
            e2_neur = ens_neur[e2_neur]
            selected_features[e2_neur] = False
            e2_depths = np.mean(com_cm[e2_neur, 2])

            # Create boolean masks for selected dendrites and selected neurons
            # For neurons, remember to only use red-labeled neurons.
            redlabel = np.array(f['redlabel'])
            selected_features = np.logical_and(redlabel, selected_features)
            selected_dendrites = np.logical_and(selected_features, den_mask)
            selected_neurons = np.logical_and(selected_features, nerden)

            # Get the depths of these selected dendrites and neurons
            den_sig_depths = com_cm[selected_dendrites, 2] # for dendrites
            sig_depths = com_cm[selected_neurons, 2] # for neurons
            for val in sig_depths:
                exp_sigdepths.append(val)
                exp_e2location.append(e2_depths)
            if animal.startswith('IT'):
                it_sigdepths += exp_sigdepths
                it_e2location += exp_e2location
                for val in den_sig_depths: # We care about dendrites if IT animal
                    it_den_sigdepths.append(val)
                    it_den_e2location.append(e2_depths)
                it_exp_counter += 1
            else:
                pt_sigdepths += exp_sigdepths
                pt_e2location += exp_e2location
                pt_exp_counter += 1

    # Plotting the depth distribution of all selected neurons 
    print(len(it_sigdepths))
    print(len(pt_sigdepths))
    fig, ax = plt.subplots(
        1,1, sharex=True, sharey=True, figsize=(9,4)
        )
    sns.scatterplot(x=it_e2location, y=it_sigdepths, ax=ax, label="IT",
        color="lightseagreen")
    sns.scatterplot(x=pt_e2location, y=pt_sigdepths, ax=ax, label="PT",
        color="coral")
    ax.set_xlabel('E2 Neuron Depth (in microns)')
    ax.set_ylabel('Depth of Feature-selected Neurons (in microns)')
    plt.suptitle('Results of feature selection over different experimental depths')
    plt.subplots_adjust(top=0.85)
    plt.legend()
    plt.show(block=True)

    # Plotting the depth distribution of all selected neurons AND dendrites
    fig, ax = plt.subplots(1,1)
    sns.regplot(
        x=it_e2location, y=it_sigdepths, ax=ax,color='r',label='Neurons'
        )
    sns.regplot(
        x=it_den_e2location, y=it_den_sigdepths, ax=ax,color='b',
        label='Dendrites'
        )
    ax.set_title('Results of feature selection in IT animals')
    ax.set_ylabel('Depth of Feature-selected Neuron/Dendrites (in microns)')
    ax.set_xlabel('E2 Neuron Depth (in microns)')
    ax.legend()
    plt.show(block=True)

def plot_rfecv_thresholds():
    '''
    Plots various accuracy thresholds to use on the RFECV models to select
    the most significant neurons. Ideally, only one hyperparameter is chosen
    for each type of animal (IT or PT) to keep consistency. A tradeoff will need
    to be made betweeen quantity of selected neurons and accuracy. Here, chance 
    performance is at 50%.
    '''

    folder = './processed/'
    IT_rfecv_models = []
    IT_experiment_files = []
    PT_rfecv_models = []
    PT_experiment_files = []
    for animal in os.listdir(folder):
        animal_path = folder + animal + '/'
        if not os.path.isdir(animal_path):
            continue
        if not (animal.startswith('IT') or animal.startswith('PT')):
            continue
        for day in os.listdir(animal_path):
            day_path = animal_path + day + '/'
            if not os.path.isdir(day_path):
                continue
            path_to_models = './processed/' + animal + '/' + day + '/rfecv_model.p'
            try:
                with open(path_to_models, 'rb') as f:
                    rfecv_models = pickle.load(f)
                f = h5py.File(
                    day_path + "full_" + animal + "_" + day + "__data.hdf5"
                    )
            except:
                continue
            if animal.startswith('IT'):
                IT_rfecv_models.append(rfecv_models)
                IT_experiment_files.append(f)
            else:
                PT_rfecv_models.append(rfecv_models)
                PT_experiment_files.append(f)

    # We will iterate through accuracy scores from 50% to 95% in 2.5% increments
    score_min_range = np.arange(.50, .95, .025)
    IT_ratio_neur_selected = []
    IT_num_neur_selected = []
    PT_ratio_neur_selected = []
    PT_num_neur_selected = []

    for score_min in score_min_range:
        IT_num_neur = 0
        IT_total_neurs = 0
        # For accuracy threshold SCORE_MIN, how many IT neurons are retained
        # over all experiments?
        for index, experiment_models in enumerate(IT_rfecv_models):
            exp_file = IT_experiment_files[index]
            neur = np.zeros(experiment_models[0].support_.size).astype('bool')
            for rfecv_model in experiment_models: # Iterates over time shifts
                if max(rfecv_model.grid_scores_) > score_min:
                    neur = np.logical_or(rfecv_model.support_, neur)
            redlabel = np.array(exp_file['redlabel'])
            neur = np.logical_and(redlabel, neur)
            IT_num_neur += np.sum(neur)
            IT_total_neurs += neur.size
        IT_ratio_neur_selected.append(IT_num_neur/IT_total_neurs)
        IT_num_neur_selected.append(IT_num_neur)

        # For accuracy threshold SCORE_MIN, how many PT neurons are retained
        # over all experiments?
        PT_num_neur = 0
        PT_total_neurs = 0
        for index, experiment_models in enumerate(PT_rfecv_models):
            exp_file = PT_experiment_files[index]
            neur = np.zeros(experiment_models[0].support_.size).astype('bool')
            for rfecv_model in experiment_models: # Iterates over time shifts
                if max(rfecv_model.grid_scores_) > score_min:
                    neur = np.logical_or(rfecv_model.support_, neur)
            redlabel = np.array(exp_file['redlabel'])
            neur = np.logical_and(redlabel, neur)
            PT_num_neur += np.sum(neur)
            PT_total_neurs += neur.size
        PT_ratio_neur_selected.append(PT_num_neur/PT_total_neurs)
        PT_num_neur_selected.append(PT_num_neur)

    plt.figure()
    plt.plot(score_min_range, IT_ratio_neur_selected, linewidth=2.5, label="IT")
    plt.plot(score_min_range, PT_ratio_neur_selected, linewidth=2.5, label="PT")
    plt.title('Number of chosen Neurons through Logistic Regression')
    plt.xlabel('Classification accuracy threshold applied on RFECV models')
    plt.ylabel('Proportion of all neurons that are chosen to be significant')
    plt.legend()
    plt.show(block=True)

    plt.figure()
    plt.plot(score_min_range, IT_num_neur_selected, linewidth=2.5, label="IT")
    plt.plot(score_min_range, PT_num_neur_selected, linewidth=2.5, label="PT")
    plt.title('Number of chosen Neurons through Logistic Regression')
    plt.xlabel('Classification accuracy threshold applied on RFECV models')
    plt.ylabel('Number of neurons that are chosen to be significant')
    plt.legend()
    plt.show(block=True)
    
if __name__=='__main__':
    plot_all_sessions_hpm()
    # analyze_feature_selection(rfecv_accuracy_threshold=.7)
    # sys.exit(0)
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('animal', help='Animal name')
    # parser.add_argument('day', help='Day of experiment')
    # args = parser.parse_args()
    # folder = "./"
    # animal = args.animal
    # day = args.day
