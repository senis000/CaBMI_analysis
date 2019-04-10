import h5py as h5
import numpy as np
import argparse
import os
import pdb
from itertools import combinations
from plotting_functions import *

def label_E2(animal):
    animal_folder = './processed/' + animal + '/'
    days = os.listdir(animal_folder)
    for day in days:
        # Load all the necessary variables
        day_folder = animal_folder + day + '/'
        f = h5.File(day_folder + 'full_' + animal + '_' + day + '__data.hdf5')
        dff = np.array(f['dff'])
        C = np.array(f['C'])
        blen = f.attrs['blen']
        trial_start = np.array(f['trial_start'])
        trial_end = np.array(f['trial_end'])
        cursor = np.array(f['cursor'])
        ens_neur = np.array(f['ens_neur'])
        exp_data = dff
        cursor = np.concatenate((np.zeros(blen), cursor)) # Pad cursor

        # Generate all possible E2 combinations. We will find the combination
        # with the maximal correlation value.
        e2_possibilities = combinations(np.arange(ens_neur.size), 2)
        best_e2_combo = None
        best_e2_combo_val = 0.0

        # Loop over each possible E2 combination and assign a score to it
        for e2 in e2_possibilities:
            mask = np.zeros(ens_neur.shape,dtype=bool)
            mask[e2[0]] = True
            mask[e2[1]] = True
            e2_neur = ens_neur[mask]
            e1_neur = ens_neur[~mask]
            correlation = 0
            for i in range(trial_start.size):
                start_idx = trial_start[i]
                end_idx = trial_end[i]
                simulated_cursor = \
                    np.sum(exp_data[e2_neur,start_idx:end_idx], axis=0) - \
                    np.sum(exp_data[e1_neur,start_idx:end_idx], axis=0)
                trial_corr = np.nansum(
                    cursor[start_idx:end_idx]*simulated_cursor
                    )
                correlation += trial_corr
            # If this is the best E2 combo so far, record it
            if correlation > best_e2_combo_val:
                best_e2_combo_val = correlation
                best_e2_combo = e2

        # Write the most probable E2 combination to the H5 file
        if 'e2_neur' in f:
            data = f['e2_neur']
            data[...] = np.array([best_e2_combo[0], best_e2_combo[1]])
        else:
            f['e2_neur'] = np.array([best_e2_combo[0], best_e2_combo[1]])
        f.close()

label_E2('IT4')
