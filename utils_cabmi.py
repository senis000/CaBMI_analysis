# 


#*************************************************************************
#************************ UTILS *****************
#*************************************************************************


__author__ = 'Nuria'


import numpy as np

       

def calc_pvalue(p_value):
    if p_value < 0.0005:
        p = '***'
    elif p_value < 0.005:
        p = '**'
    elif p_value < 0.05:
        p = '*'
    else:
        p = 'ns'
    return p


def sliding_mean(data_array, window=5):
    # program to smooth a graphic
    data_array = np.array(data_array)
    new_list = []
    for i in range(np.size(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, np.size(data_array)))
        avg = 0
        for j in indices:
            avg = np.nansum([avg, data_array[j]])
        avg /= float(np.size(indices))
        new_list.append(avg)
    return np.array(new_list)

def time_lock_activity(f, t_size=[30,3], tbin=10, trial_type=0):
    '''
    Creates a 3d matrix time-locking activity to trial end.
    Input:
        F: a File object; the experiment HDF5 file
        T_SIZE: an array; the first value is the number of
            seconds total to keep. The second value
            is the number of seconds after the trial end to keep.
        T_BIN: an integer; the number of frames per second
        TRIAL_TYPE: an integer from [0,1,2]. 0 indicates all trials,
            1 indicates hit trials, 2 indicates miss trials.
    Output:
        NEURON_ACTIVITY: a numpy matrix; (trials x neurons x frames)
            in size.
    '''
    trial_start = np.asarray(f['trial_start']).astype('int')
    trial_end = np.asarray(f['trial_end']).astype('int')
    assert(trial_start.shape[0] == trial_end.shape[0])
    if trial_type == 1: # Hit Trials
        hit_idxs = []
        hits = np.array(f['hits'])
        for idx in range(trial_end.size):
            if trial_end[idx] in hits:
                hit_idxs.append(idx)
        trial_start = trial_start(hit_idxs)
        trial_end = trial_end(hit_idxs)
    elif trial_type == 2: # Miss Trials
        miss_idxs = []
        misses = np.array(f['miss'])
        for idx in range(trial_end.size):
            if trial_end[idx] in misses:
                miss_idxs.append(idx)
        trial_start = trial_start(miss_idxs)
        trial_end = trial_end(miss_idxs)
    C = np.asarray(f['C'])
    neuron_activity = np.ones(
        (trial_end.shape[0], C.shape[0], np.sum(t_size)*tbin)
        )*np.nan # (num_trials x num_neurons x num_frames)
    for ind, trial in enumerate(trial_end):
        aux_act = C[:, trial_start[ind]:trial + t_size[1]*tbin]
        neuron_activity[ind, :, -aux_act.shape[1]:] = aux_act
    return neuron_activity
