import warnings
import h5py
import numpy as np
import pickle
import shutil
from scipy.stats import zscore
from utils_gte import *
from utils_cabmi import *
from utils_loading import encode_to_filename


class ExpGTE:
    """A class that wraps around an experiment and runs GTE in various ways"""
    # Z-score threshold values
    reward_threshold = 4.0
    whole_exp_threshold = 10.0

    def __init__(self, folder, animal, day, sec_var='', lag=2, method='te-extended', out=None):
        """
        method: str
            xc, by finding the peak in the cross-correlogram between the two time series
            mi, by finding the lag with the largest Mutual Information
            gc, by computing the Granger Causality, based on: C.W.J. Granger, Investigating Causal Relations by Econometric Models and Cross-Spectral Methods , Econometrica, 1969
            te-extended, by computing GTE as defined above.
            TE and GTE without binning:
            te-binless-Leonenko, based on: L.F. Kozachenko and N.N. Leonenko, 1987
            te-binless-Kraskov, based on: A. Kraskov et al., 2004
            te-binless-Frenzel, based on: S. Frenzel and B. Pompe, 2007
            te-symbolic (experimental) based on: M. Staniek and K. Lehnertz, 2008.
        out: str
            path of the root output directory, if left None, default to {folder}/utils/FC/{method}/
        """
        if out is None:
            out = os.path.join(folder, 'utils/FC/')
        self.out_path = os.path.join(out, f'te-package_' + method, animal, day, '')
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.folder = folder
        self.animal = animal
        self.day = day
        self.parameters = {
            "AutoConditioningLevelQ": True,
            'AutoBinNumberQ': True, 'SourceMarkovOrder': lag, 'TargetMarkovOrder': lag,
            'StartSampleIndex': 2
        }  # update conditioning level for gte

        self.exp_file = h5py.File(encode_to_filename(os.path.join(folder, 'processed'), animal, day), 'r')
        self.blen = self.exp_file.attrs['blen']
        self.method = method

    def baseline(self, roi='red', input_type='dff', parameters=None, pickle_results=True,
                 zclean=False, clean=True):
        '''
        Run GTE over all neurons, over the baseline.
        Inputs:
            PARAMETERS: Dictionary; parameters for GTE
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix
        Outputs:
            RESULTS: An array of numpy matrices (GTE connectivity matrices)
        '''
        exp_name = self.animal + '_' + self.day + '_' + 'baseline'
        exp_data = np.array(self.exp_file[input_type][:, :self.blen])  # (neurons x frames)
        if roi == 'neuron':
            exp_data = exp_data[np.array(self.exp_file['nerden'])]
        elif roi == 'red':
            exp_data = exp_data[np.array(self.exp_file['redlabel'])]
        elif roi == 'ens':
            ens = np.array(self.exp_file['ens_neur'])
            ens = ens[~np.isnan(ens)].astype(np.int)
            exp_data = exp_data[ens]
        if zclean:
            exp_data = zscore(exp_data, axis=1)
            exp_data = np.nan_to_num(exp_data)
            exp_data = np.maximum(exp_data, -1 * self.whole_exp_threshold)
            exp_data = np.minimum(exp_data, self.whole_exp_threshold)
            exp_data = np.expand_dims(exp_data, axis=0)  # (1 x neurons x frames)
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(exp_name, exp_data, parameters)
        results = run_gte(control_file_names, exclude_file_names,
                          output_file_names, method=self.method)
        if pickle_results:
            order = self.parameters['SourceMarkovOrder']
            with open(self.out_path + f'baseline_{roi}_{input_type}_order_{order}.p', 'wb') as p_file:
                pickle.dump(results, p_file)
        if clean:
            exp_path = "./te-causality/transferentropy-sim/experiments/" + exp_name
            try:
                shutil.rmtree(exp_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        return results

    # TODO: fix input type for all following methods
    def whole_experiment(self, roi='ens', input_type='dff', parameters=None, pickle_results=True):
        '''
        Run GTE over all neurons, over the whole experiment.
        Inputs:
            PARAMETERS: Dictionary; parameters for GTE
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix
        Outputs:
            RESULTS: An array of numpy matrices (GTE connectivity matrices)
        '''
        exp_name = self.animal + '_' + self.day + '_' + 'whole'
        exp_data = np.array(self.exp_file[input_type])  # (neurons x frames)
        exp_data = exp_data[:, self.blen:]  # Isolate the experiment
        exp_data = exp_data[np.array(self.exp_file['nerden']), :]
        exp_data = zscore(exp_data, axis=1)
        exp_data = np.nan_to_num(exp_data)
        exp_data = np.maximum(exp_data, -1 * self.whole_exp_threshold)
        exp_data = np.minimum(exp_data, self.whole_exp_threshold)
        exp_data = np.expand_dims(exp_data, axis=0)  # (1 x neurons x frames)
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(exp_name, exp_data, parameters)
        results = run_gte(control_file_names, exclude_file_names,
                          output_file_names, method=self.method)
        if pickle_results:
            with open(self.out_path + 'whole_experiment.p', 'wb') as p_file:
                pickle.dump(results, p_file)
        return results

    def experiment_end(self, end_frame=0, length=0,
                       parameters=None, pickle_results=True):
        '''
        Run GTE over all neurons, over the end of the experiment.
        Inputs:
            END_FRAME: The frame to consider as the 'end' of the experiment.
                By default this is the last frame in the matrix C. However,
                you may wish to define another frame as the 'end' (for instance,
                if you are selecting for the optimal performance).
            LENGTH: This is the number of frames to process, up to END_FRAME.
                If not overwritten, SELF.BLEN will be used by default.
            PARAMETERS: Dictionary; parameters for GTE
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix
        Outputs:
            RESULTS: An array of numpy matrices (GTE connectivity matrices)
        '''
        if length == 0:
            length = self.blen
        exp_name = self.animal + '_' + self.day + '_' + 'expend'
        exp_data = np.array(self.exp_file['C'])  # (neurons x frames)
        exp_data = exp_data[:, end_frame - length:]  # Isolate the experiment
        exp_data = exp_data[np.array(self.exp_file['nerden']), :]
        exp_data = zscore(exp_data, axis=1)
        exp_data = np.nan_to_num(exp_data)
        exp_data = np.maximum(exp_data, -1 * self.whole_exp_threshold)
        exp_data = np.minimum(exp_data, self.whole_exp_threshold)
        exp_data = np.expand_dims(exp_data, axis=0)  # (1 x neurons x frames)
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(exp_name, exp_data, parameters)
        results = run_gte(control_file_names, exclude_file_names,
                          output_file_names, method=self.method)
        if pickle_results:
            with open(self.out_path + 'experiment_end.p', 'wb') as p_file:
                pickle.dump(results, p_file)
        return results

    def reward_end(self, frame_size, parameters=None, pickle_results=True):
        '''
        Run general transfer of entropy in the last FRAME_SIZE frames before
        a hit, over all reward trials. Return an array of connectivity matrices
        over each reward trial.
        Inputs:
            FRAME_SIZE: Integer; number of frames before the hit to consider.
            PARAMETERS: Dictionary; parameters for GTE
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix
        Outputs:
            RESULTS: An array of numpy matrices (GTE connectivity matrices)
        '''
        exp_name = self.animal + '_' + self.day + '_' + 'rewardend'
        exp_data = time_lock_activity(self.exp_file, t_size=[frame_size, 0])
        array_t1 = np.array(self.exp_file['array_t1'])
        exp_data = exp_data[array_t1, :, :]
        exp_data = exp_data[:, np.array(self.exp_file['nerden']), :]
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters

        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(
                exp_name, exp_data, parameters,
                to_zscore=True, zscore_threshold=self.reward_threshold
            )
        results = run_gte(control_file_names, exclude_file_names,
                          output_file_names, method=self.method)
        if pickle_results:
            with open(self.out_path + 'reward_end.p', 'wb') as p_file:
                pickle.dump(results, p_file)
        return results

    def reward_sliding(self, reward_idx, frame_size, frame_step, parameters=None,
                       pickle_results=True):
        '''
        Run general transfer of entropy over reward trial REWARD_IDX, with a
        sliding window of size FRAME_SIZE, from the last 300 frames of each
        trial.
        Inputs:
            FRAME_SIZE: Integer; number of frames to process in GTE.
            FRAME_STEP: Integer; number of frames for each step through signal.
            PARAMETERS: Dictionary; parameters for GTE.
            PICKLE_RESULTS: Boolean; whether or not to save the results matrix.
        Outputs:
            RESULTS: An array of numpy matrices (GTE connectivity matrices)
        '''
        exp_name = self.animal + '_' + self.day + '_' + \
                   'rewardsliding' + str(frame_size) + '_'
        exp_data = time_lock_activity(self.exp_file, t_size=[300, 0])
        array_t1 = np.array(self.exp_file['array_t1'])
        exp_data = exp_data[array_t1, :, :]
        exp_data = exp_data[:, np.array(self.exp_file['nerden']), :]
        num_rewards, num_neurons, num_frames = exp_data.shape
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters
        exp_name_idx = exp_name + str(reward_idx)

        # Chop off the NaN sections of the reward trial of interest.
        reward_data = exp_data[reward_idx, :, :]
        if np.sum(np.isnan(reward_data[0, :])) == 0:
            non_nan_idx = 0
        else:
            non_nan_idx = np.where(np.isnan(reward_data[0, :]))[0][-1] + 1
        reward_data = reward_data[:, non_nan_idx:]

        # If the reward trial is too short return an empty array.
        if reward_data.shape[1] < frame_size:
            return []

        # Otherwise, z-score the signal and GTE as normal
        reward_data = zscore(reward_data, axis=1)
        reward_data = np.nan_to_num(reward_data)
        reward_data = np.maximum(reward_data, -1 * self.reward_threshold)
        reward_data = np.minimum(reward_data, self.reward_threshold)
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files_sliding(
                exp_name_idx, reward_data, parameters,
                frame_size, frame_step=frame_step
            )
        results = run_gte(
            control_file_names, exclude_file_names, output_file_names, method=self.method
        )
        if pickle_results:
            with open(self.out_path + 'reward_sliding_' \
                      + str(reward_idx) + '.p', 'wb') as p_file:
                pickle.dump(results, p_file)
        return results

    def reward_sliding_full(self, frame_size, frame_step, parameters=None):
        '''
        Runs reward_sliding over each reward trial, with a sliding
        window of size FRAME_SIZE, from the last 300 frames of each trial.
        Due to memory constraints, this function does not return anything and
        will instead automatically pickle the results of each reward trial.
        Inputs:
            FRAME_SIZE: Integer; number of frames to process in GTE.
            FRAME_STEP: Integer; number of frames for each step through signal.
            PARAMETERS: Dictionary; parameters for GTE.
        '''
        exp_name = self.animal + '_' + self.day + '_' + 'rewardsliding'
        array_t1 = np.array(self.exp_file['array_t1'])
        num_rewards = array_t1.size

        # Run sliding GTE over each reward trial.
        for reward_idx in range(num_rewards):
            self.reward_sliding(
                reward_idx, frame_size, frame_step,
                parameters=parameters, pickle_results=True
            )

    def shuffled_results(self, frame_size, parameters=None,
                         iters=100, pickle_results=True):
        '''
        Runs GTE over 'shuffled' instances of neurons over reward trials.
        Returns the average over many of these results.
        Inputs:
            FRAME_SIZE: Integer; number of frames to process in GTE
            PARAMETERS: Dictionary; parameters for GTE.
            ITERS: Number of 'shuffled' samples to take and average over.
        Outputs:
            RESULT: A GTE connectivity matrix
        '''

        exp_name = self.animal + '_' + self.day + '_' + 'rewardshuffled'
        exp_data = time_lock_activity(self.exp_file, t_size=[300, 0])
        array_t1 = np.array(self.exp_file['array_t1'])
        exp_data = exp_data[array_t1, :, :]
        exp_data = exp_data[:, np.array(self.exp_file['nerden']), :]

        # Extract only reward trials that are long enough to sample from
        sufficient_trials = []
        for i in range(exp_data.shape[0]):
            if np.sum(np.isnan(exp_data[i, 0, :])) == 0:
                sufficient_trials.append(i)
                continue
            non_nan_idx = np.where(np.isnan(exp_data[i, 0, :]))[0][-1] + 1
            if (exp_data.shape[2] - non_nan_idx) >= frame_size:
                sufficient_trials.append(i)
        exp_data = exp_data[sufficient_trials, :, :]

        # Extract out 'shuffled' windows to use
        num_rewards = exp_data.shape[0]
        num_neurons = exp_data.shape[1]
        num_frames = exp_data.shape[2]
        shuffled_data = np.zeros((iters, num_neurons, frame_size))
        for i in range(iters):
            for j in range(num_neurons):
                # Sample the reward trial to use
                reward_idx = np.random.choice(num_rewards)
                # Sample the frame to start on, excluding Nans
                full_signal = exp_data[reward_idx, j, :]
                if np.sum(np.isnan(full_signal)) > 0:
                    non_nan_idx = np.where(np.isnan(full_signal))[0][-1] + 1
                else:
                    non_nan_idx = 0
                full_signal = full_signal[non_nan_idx:]
                frame_idx = np.random.choice(np.arange(0, full_signal.size - frame_size + 1))
                full_signal = zscore(full_signal)
                full_signal = np.nan_to_num(full_signal)
                full_signal = np.maximum(full_signal, -1 * self.reward_threshold)
                full_signal = np.minimum(full_signal, self.reward_threshold)
                shuffled_data[i, j, :] = \
                    full_signal[frame_idx:frame_idx + frame_size]
        neuron_locations = np.array(self.exp_file['com_cm'])
        if parameters is None:
            parameters = self.parameters

        # Run GTE on shuffled data
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(exp_name, shuffled_data, parameters)
        results = run_gte(control_file_names, exclude_file_names,
                          output_file_names, method=self.method)

        # Compute the average information transfer over shuffled instances.
        reward_shuffled_results = np.ones(results[0].shape) * np.nan
        num_results = len(results)
        for i in range(num_neurons):
            for j in range(num_neurons):
                num_samples = 0.0
                value_sum = 0.0
                for result_idx in range(num_results):  # Loop over all results
                    val = results[result_idx][i][j]
                    if np.isnan(val):
                        continue
                    else:
                        num_samples += 1.0
                        value_sum += val
                if num_samples > 0:  # Calculate the average
                    reward_shuffled_results[i][j] = value_sum / num_samples
        if pickle_results:
            with open(self.out_path + 'reward_shuffled.p', 'wb') as p_file:
                pickle.dump(reward_shuffled_results, p_file)
        return reward_shuffled_results

    def shuffled_whole(self, frame_size, parameters=None,
                       iters=100, pickle_results=True):
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

        exp_name = self.animal + '_' + self.day + '_' + 'wholeshuffled' + str(frame_size)
        exp_data = np.array(self.exp_file['C'])  # (neurons x frames)
        exp_data = exp_data[np.array(self.exp_file['nerden']), :]
        exp_data = zscore(exp_data, axis=1)
        exp_data = np.nan_to_num(exp_data)
        exp_data = np.maximum(exp_data, -1 * self.whole_exp_threshold)
        exp_data = np.minimum(exp_data, self.whole_exp_threshold)

        # Extract out 'shuffled' windows to use
        num_neurons = exp_data.shape[0]
        num_frames = exp_data.shape[1]
        shuffled_data = np.zeros((iters, num_neurons, frame_size))
        for i in range(iters):
            for j in range(num_neurons):
                # Sample the frame to start on
                full_signal = exp_data[j, :]
                frame_idx = np.random.choice(np.arange(0, full_signal.size - frame_size + 1))
                shuffled_data[i, j, :] = \
                    full_signal[frame_idx:frame_idx + frame_size]
        if parameters is None:
            parameters = self.parameters

        # Run GTE on shuffled data
        control_file_names, exclude_file_names, output_file_names = \
            create_gte_input_files(exp_name, shuffled_data, parameters)
        results = run_gte(control_file_names, exclude_file_names,
                          output_file_names, method=self.method)

        # Compute the average information transfer over shuffled instances.
        whole_shuffled_results = np.ones(results[0].shape) * np.nan
        num_results = len(results)
        for i in range(num_neurons):
            for j in range(num_neurons):
                num_samples = 0.0
                value_sum = 0.0
                for result_idx in range(num_results):  # Loop over all results
                    val = results[result_idx][i][j]
                    if np.isnan(val):
                        continue
                    else:
                        num_samples += 1.0
                        value_sum += val
                if num_samples > 0:  # Calculate the average
                    whole_shuffled_results[i][j] = value_sum / num_samples
        if pickle_results:
            with open(self.out_path + 'whole_shuffled.p', 'wb') as p_file:
                pickle.dump(whole_shuffled_results, p_file)
        return whole_shuffled_results


def fc_te_caulsaity(exp_name, exp_data, keywords, lag=2, method='te-extended',
                               pickle_path=None, clean=True):
    parameters = {
            "AutoConditioningLevelQ": True,
            'AutoBinNumberQ': True, 'SourceMarkovOrder': lag, 'TargetMarkovOrder': lag,
            'StartSampleIndex': 2}
    if len(exp_data.shape) == 2:
        exp_data = np.expand_dims(exp_data, axis=0)
    control_file_names, exclude_file_names, output_file_names = \
        create_gte_input_files(exp_name, exp_data, parameters)
    results = run_gte(control_file_names, exclude_file_names,
                      output_file_names, method=method)
    if pickle_path is not None:
        order = parameters['SourceMarkovOrder']
        with open(os.path.join(pickle_path, f'{exp_name}_{keywords}_order_{order}.p'), 'wb') as p_file:
            pickle.dump(results, p_file)
    if clean:
        exp_path = "./te-causality/transferentropy-sim/experiments/" + exp_name
        try:
            shutil.rmtree(exp_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
