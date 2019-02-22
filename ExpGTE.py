import warnings
import h5py
import numpy as np
from utils_gte import *

class ExpGTE:
    """A class that wraps around an experiment and runs GTE in various ways"""

    whole_exp_results = None
    grouped_results = None
    reward_end_results = None
    reward_early_results = None
    reward_late_results = None
    grouping = None

    def __init__(self, folder, animal, day, sec_var=''):
        self.folder = folder
        self.animal = animal
        self.day = day
        self.parameters = {'bins':5}
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

        control_file_names, output_file_names = create_gte_input_files(\
            exp_name, exp_data, parameters,
            frame_size, frame_step
            )
        results = run_gte(control_file_names, output_file_names, pickle_results)
        if to_plot:
            visualize_gte_results(results, neuron_locations)
        delete_gte_files(exp_name, delete_output=False)
        self.whole_exp_results = results
        return results

    def whole_experiment_grouped(self, grouping, to_plot=True):
        '''
        Run GTE over all neurons, over the whole experiment. Averages scores
        over user-defined groupings of neurons. Here it is assumed that the
        function WHOLE_EXPERIMENT is already run; otherwise, an exception is
        thrown.
        Inputs:
            GROUPING: Numpy array of NUM_NEURONS size; an integer ID is given
                to each neuron, defining their group. This array is 0-indexed.
        Outputs:
            GROUPED_RESULTS: An array of numpy matrices (GTE connectivity matrices) 
        '''
        if self.whole_exp_results is None:
            raise RuntimeError('No results for the function to load. ' +
                'Please run WHOLE_EXPERIMENT, or run the overloaded version ' +
                'of this function.')
        num_neurons = self.whole_exp_results[0].shape[0]
        num_groups = np.unique(grouping).size
        if grouping.size != num_neurons:
            raise RuntimeError('Wrong dimensions for GROUPING')

        grouped_results = []
        for result in self.whole_exp_results:
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
        self.grouped_results = grouped_results
        self.grouping = grouping
        if to_plot:
            visualize_gte_matrices(grouped_results)
        return grouped_results

    def whole_experiment_grouped(self, grouping, frame_size, frame_step,
        parameters=None, to_plot=True):
        '''
        Run GTE over all neurons, over the whole experiment. Averages scores
        over user-defined groupings of neurons.
        Inputs:
            GROUPING: Numpy array of NUM_NEURONS size; an integer ID is given
                to each neuron, defining their group.
            FRAME_SIZE: Integer; number of frames to process in GTE
            FRAME_STEP: Integer; number of frames for each step through the signal.
            PARAMETERS: Dictionary; parameters for GTE
        Outputs:
            GROUPED_RESULTS: An array of numpy matrices (GTE connectivity matrices)
        '''
        if self.whole_exp_results is not None:
            warnings.warn("There are already existing GTE results present " +
                "that will be overriden by this function.") 
        if parameters is None:
            parameters = self.parameters
        self.whole_experiment(frame_size, frame_step, parameters,
            to_plot=False, pickle_results=False)
        self.whole_experiment_grouped(grouping)
        if to_plot:
            visualize_gte_matrices(self.grouped_results)
        return grouped_results
    
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

        control_file_names, output_file_names = create_gte_input_files(\
            exp_name, exp_data, parameters)
        results = run_gte(control_file_names, output_file_names, pickle_results)
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
        ctrl_file_early, out_file_early = create_gte_input_files(\
            exp_name_early, early_trials, parameters)
        results_early = run_gte(
            ctrl_file_early, out_file_early, pickle_results
            )
        ctrl_file_late, out_file_late = create_gte_input_files(\
            exp_name_late, late_trials, parameters)
        results_late = run_gte(
            ctrl_file_late, out_file_late, pickle_results
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
