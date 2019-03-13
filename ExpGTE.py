import warnings
import h5py
import numpy as np
from utils_gte import *
from utils_cabmi import *

class ExpGTE:
    """A class that wraps around an experiment and runs GTE in various ways"""

    whole_exp_results = None
    reward_end_results = None
    reward_early_results = None
    reward_late_results = None
    whole_exp_results_shuffled = None
    reward_shuffled = None

    def __init__(self, folder, animal, day, sec_var=''):
        self.folder = folder
        self.animal = animal
        self.day = day
        self.parameters = {
            'bins':5, 'SourceMarkovOrder':2, 'TargetMarkovOrder':2
            }
        folder_path = folder +  'processed/' + animal + '/' + day + '/'
        self.exp_file = h5py.File(
            folder_path + 'full_' + animal + '_' + day + '_' +
            sec_var + '_data.hdf5', 'r'
            )

    def whole_experiment(self, frame_size, frame_step, parameters=None,
        to_plot=True, pickle_results = True):
        '''
        Run GTE over all neurons, over the whole experiment.
        Inputs:
            FRAME_SIZE: Integer; number of frames to process in GTE
            FRAME_STEP: Integer; number of frames for each step through the signal.
            PARAMETERS: Dictionary; parameters for GTE
            TO_PLOT: Boolean; whether or not to call the visualization script
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix
        Outputs:
            RESULTS: An array of numpy matrices (GTE connectivity matrices)
        '''
        exp_name = self.animal + '_' + self.day + '_' + 'whole'
        exp_data = np.array(self.exp_file['C']) # (neurons x frames)
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(
                exp_name, exp_data, parameters, frame_size, frame_step
            )
        results = run_gte(control_file_names, exclude_file_names,
            output_file_names, pickle_results)
        if to_plot:
            visualize_gte_results(results, neuron_locations)
        delete_gte_files(exp_name, delete_output=False)
        self.whole_exp_results = results
        return results
    
    def reward_end(self, frame_size, parameters=None,
        to_plot=True, pickle_results=True):
        '''
        Run general transfer of entropy in the last FRAME_SIZE frames before
        a hit, over all reward trials. Return an array of connectivity matrices
        over time.
        Inputs:
            FRAME_SIZE: Integer; number of frames before the hit to consider.
            PARAMETERS: Dictionary; parameters for GTE
            TO_PLOT: Boolean; whether or not to call the visualization script
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix 
        Outputs:
            GROUPED_RESULTS: An array of numpy matrices (GTE connectivity matrices) 
        '''
        exp_name = self.animal + '_' + self.day + '_' + 'rewardend'
        tbin = 10.0
        num_secs = frame_size/tbin
        exp_data = time_lock_activity(
            self.exp_file, [num_secs,0], tbin=10, trial_type=1
            )
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters

        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(exp_name, exp_data, parameters)
        results = run_gte(control_file_names, exclude_file_names,
            output_file_names, pickle_results)
        if to_plot:
            visualize_gte_results(results, neuron_locations)
        delete_gte_files(exp_name, delete_output=False)
        self.reward_end_results = results
        return results

    def reward_learning(self, frame_size, trials_to_compare=10, parameters=None,
        to_plot=True, pickle_results=True):
        '''
        Run general transfer of entropy in the last FRAME_SIZE frames before
        a hit. Compares the first TRIALS_TO_COMPARE reward trials to the last
        TRIALS_TO_COMPARE reward trials.
        Inputs:
            FRAME_SIZE: Integer; number of frames before the hit to consider.
            TRIALS_TO_COMPARE: Integer; number of trials in the early or later
                reward trials to consider.
            PARAMETERS: Dictionary; parameters for GTE
            TO_PLOT: Boolean; whether or not to call the visualization script
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix 
        Outputs:
            RESULTS_EARLY: An array of numpy matrices
            RESULTS_LATE: An array of numpy matrices
        '''
        exp_name_early = self.animal + '_' + self.day + '_' + 'rewardearly'
        exp_name_late = self.animal + '_' + self.day + '_' + 'rewardlate'
        tbin = 10.0
        num_secs = frame_size/tbin
        exp_data = time_lock_activity(
            self.exp_file, [num_secs,0], tbin=10, trial_type=1
            )   # (trials x neurons x frames)
        early_trials = exp_data[:trials_to_compare,:,:]
        late_trials = exp_data[-trials_to_compare:,:,:]
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        ctrl_file_early, excl_file_early, out_file_early = \
            create_gte_input_files(exp_name_early, early_trials, parameters)
        results_early = run_gte(
            ctrl_file_early, excl_file_early, out_file_early, pickle_results
            )
        ctrl_file_late, excl_file_late, out_file_late = \
            create_gte_input_files(exp_name_late, late_trials, parameters)
        results_late = run_gte(
            ctrl_file_late, excl_file_late, out_file_late, pickle_results
            )
        if to_plot:
            print("Plotting early reward trials.")
            visualize_gte_results(results_early, neuron_locations)
            print("Plotting late reward trials.")
            visualize_gte_results(results_late, neuron_locations)
        delete_gte_files(exp_name_early, delete_output=False)
        delete_gte_files(exp_name_late, delete_output=False)
        self.reward_early_results = results_early
        self.reward_late_results = results_late
        return results_early, results_late

    def group_results(self, results_type, grouping, to_plot=True):
        '''
        Groups an existing GTE connectivity matrix by averaging scores
        over user-defined groupings of neurons. Here it is assumed that GTE
        is already run; otherwise, an exception is thrown.
        Inputs:
            RESULTS_TYPE: a String; indicates which matrix to use for grouping.
            GROUPING: Numpy array of NUM_NEURONS size; an integer ID is given
                to each neuron, defining their group. This array is 0-indexed.
        Outputs:
            GROUPED_RESULTS: An array of numpy matrices (GTE connectivity matrices) 
        '''
        results = None
        if results_type == "whole":
            results = self.whole_exp_results
        elif results_type == "reward_end":
            results = self.reward_end_results
        elif results_type == "reward_early":
            results = self.reward_early_results
        elif results_type == "reward_late":
            results = self.reward_late_results
        if results is None:
            raise RuntimeError('No results for the function to load. ' +
                'Please check that GTE is already run and check the correct ' +
                'key word was used in function GROUP_RESULTS.')
        num_neurons = results[0].shape[0]
        num_groups = np.unique(grouping).size
        if grouping.size != num_neurons:
            raise RuntimeError('Wrong dimensions for GROUPING')

        grouped_results = []
        for result in results:
            grouped_result = np.zeros((num_groups, num_groups))
            for i in range(num_groups):
                for j in range(num_groups):
                    if i == j:
                        continue
                    relevant_vals = []
                    for k in range(num_neurons):
                        if grouping[k] != i:
                            continue
                        for l in range(num_neurons):
                            if grouping[l] != j:
                                continue
                            relevant_vals.append(result[k, l])
                    grouped_result[i, j] = np.mean(relevant_vals)
            grouped_results.append(grouped_result)
        if to_plot:
            visualize_gte_matrices(grouped_results)
        return grouped_results

    def shuffled_whole_exp(self, frame_size, frame_step, parameters=None,
        iters=100):
        '''
        Runs GTE over 'shuffled' instances of neurons over the whole experiment.
        Returns the average over many of these results.
        Inputs:
            FRAME_SIZE: Integer; number of frames to process in GTE
            FRAME_STEP: Integer; number of frames for each step through the signal.
            PARAMETERS: Dictionary; parameters for GTE.
            ITERS: Number of 'shuffled' samples to take and average over.
        Outputs:
            RESULT: A GTE connectivity matrix
        '''

        exp_name = self.animal + '_' + self.day + '_' + 'wholeshuffled'
        exp_data = np.array(self.exp_file['C']) # (neurons x frames)
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        control_file_names, exclude_file_names, output_file_names = \
            create_shuffled_input_files(
                exp_name, exp_data, parameters, frame_size, iters
            )
        results = run_gte(control_file_names, exclude_file_names,
            output_file_names, pickle_results)
        delete_gte_files(exp_name, delete_output=False)
        self.whole_exp_results_shuffled = np.mean(results) #TODO: fix
        return self.whole_exp_results_shuffled

    def shuffled_results(self, frame_size, parameters=None,
        iters=100): #TODO: adapt for results
        '''
        Runs GTE over 'shuffled' instances of neurons over the whole experiment.
        Returns the average over many of these results.
        Inputs:
            FRAME_SIZE: Integer; number of frames to process in GTE
            PARAMETERS: Dictionary; parameters for GTE.
            ITERS: Number of 'shuffled' samples to take and average over.
        Outputs:
            RESULT: A GTE connectivity matrix
        '''

        exp_name = self.animal + '_' + self.day + '_' + 'rewardshuffled'
        tbin = 10.0
        num_secs = int(frame_size/tbin)
        exp_data = time_lock_activity(
            self.exp_file, [num_secs,0], tbin=10, trial_type=1
            )
        num_rewards = exp_data.shape[0]
        num_neurons = exp_data.shape[1]
        num_frames = exp_data.shape[2]
        shuffled_data = np.zeros((iters, num_neurons, num_frames))
        for i in range(iters):
            for j in range(num_neurons):
                reward_idx = np.random.choice(num_rewards)
                # Sample the frame to start on, excluding Nans 
                pdb.set_trace()
                frame_idx = np.random.choice(num_frames) #TODO: correct   
                shuffled_data[i,j,:] = exp_data[reward_idx, j, frame_idx]
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(exp_name, shuffled_data, parameters)
        results = run_gte(control_file_names, exclude_file_names,
            output_file_names, pickle_results)
        delete_gte_files(exp_name, delete_output=False)
        self.reward_shuffled = np.mean(results) #TODO: fix
        return self.reward_shuffled
