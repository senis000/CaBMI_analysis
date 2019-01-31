import warnings
import h5py
import numpy as np
from utils_gte import *

class ExpGTE:
    """A class that wraps around an experiment and runs GTE in various ways"""

    whole_exp_results = None
    grouped_results = None
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

    def whole_experiment(self, frame_size, frame_step, parameters=self.parameters,
        to_plot=True, pickle_results = True):
        '''
        Run GTE over all neurons, over the whole experiment.
        Inputs:
            FRAME_SIZE: Integer; number of frames to process in GTE
            FRAME_STEP: Integer; number of frames for each step through the signal.
            PARAMETERS: Dictionary; parameters for GTE
            TO_PLOT: Boolean; whether or not to call the visualization script
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix
        '''
        exp_name = self.animal + '_' + self.day + '_' + 'whole'
        exp_data = np.array(self.exp_file['C']) # (neurons x frames)
        neuron_locations = np.array(self.exp_file['com_cm'])

        control_file_names, output_file_names = create_gte_input_files(\
            exp_name, exp_data, parameters,
            frame_size, frame_step
            )
        results = run_gte(control_file_names, output_file_names, pickle_results)
        if to_plot:
            visualize_gte_results(results, neuron_locations)
        delete_gte_files(exp_name, delete_output=False)
        self.whole_exp_results = results

    def whole_experiment_grouped(self, grouping): #TODO: Plot groups sensibly
        '''
        Run GTE over all neurons, over the whole experiment. Averages scores
        over user-defined groupings of neurons. Here it is assumed that the
        function WHOLE_EXPERIMENT is already run; otherwise, an exception is
        thrown.
        Inputs:
            GROUPING: Numpy array of NUM_NEURONS size; an integer ID is given to
                each neuron, defining their group.
        '''
        if self.whole_exp_results is None:
            raise RuntimeError('No results for the function to load. ' +
                'Please run WHOLE_EXPERIMENT, or run the overloaded version ' +
                'of this function.')
        num_neurons = self.whole_exp_results[0]
        num_groups = np.unique(grouping).size
        if grouping.size != num_neurons:
            raise RuntimeError('Wrong dimensions for GROUPING')

        grouped_results = [] #TODO: Test grouping
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

    def whole_experiment_grouped(self, grouping, frame_size, frame_step,
        parameters=self.parameters):
        '''
        Run GTE over all neurons, over the whole experiment. Averages scores
        over user-defined groupings of neurons.
        Inputs:
            GROUPING: Numpy array of NUM_NEURONS size; an integer ID is given to
                each neuron, defining their group.
            FRAME_SIZE: Integer; number of frames to process in GTE
            FRAME_STEP: Integer; number of frames for each step through the signal.
            PARAMETERS: Dictionary; parameters for GTE
        '''
        if self.whole_exp_results is not None:
            warnings.warn("There are already existing GTE results present " +
                "that will be overriden by this function.") 
        self.whole_experiment(frame_size, frame_step, parameters,
            to_plot=False, pickle_results=False)
        self.whole_experiment_grouped(grouping)
