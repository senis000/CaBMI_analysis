# 
#*************************************************************************
#************************ UTILS *****************
#*************************************************************************


__author__ = 'Nuria'


import numpy as np
import pdb       

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


def shuffle_peaks(data_array, sf, ef, ipi_proc, axis=0):
    """
    Takes in data_array of calcium peaks and shuffle with respect to axis.

    Input:
        data_array: ndarray
            Numpy array with peak data and inter-peak intervals
        sf: function
            start function that takes in (data_array, i) where i is the
            index that signifies the start of a peak region (inclusive)
        ef: function
            end function that takes in (data_array, j) where j the index
            that signifies the end of a peak region (exclusive),
            therefore, data_array[i:j] would represent one peak region
        ipi_proc: function
            randomizing procedure for ipi data shuffling
        axis: int
            axis for shuffling

    Returns:
        shuffled: ndarray
            shuffled version of peak array with respect to axis
    """
    peak_regions = []
    prev_end = 0
    IPI = np.array(np.empty(data_array.shape[:axis]+(0,)+data_array.shape[
        axis+1:]))
    pk_start = None
    # TODO: PROCEDURE ONLY APPLIES TO 1D NOW
    for i in range(len(data_array)):
        if pk_start is None:
            # Outside of peak regions, wait for criterion sf to be met
            if sf(data_array, i):
                pk_start = i
                IPI = np.concatenate((IPI,
                    np.take(data_array, range(prev_end, pk_start),
                            axis=axis)), axis=axis)
        else:
            # Within peak_regions, record the peak if ef criterion is met
            if ef(data_array, i):
                peak_regions.append((pk_start, i))
                prev_end = i
                pk_start = None

    IPI = np.concatenate((IPI,
        np.take(data_array, range(prev_end, data_array.shape[axis]),
                axis=axis)), axis=axis)

    # TODO: USE NUMPY Parallelization later to expedite the process, so far
    #  use naive method
    np.random.shuffle(peak_regions)
    newIPI = ipi_proc(IPI)
    s_inds = np.random.choice(len(IPI)+1, len(peak_regions))
    # TODO: RETURN SEQUENCE SUCH THAT all peak region sequences will be
    #  appended in accordance









def time_lock_activity(f, t_size=[300,30]):
    '''
    Creates a 3d matrix time-locking activity to trial end.
    Input:
        F: a File object; the experiment HDF5 file
        T_SIZE: an array; the first value is the number of
            frames before the hit we want to keep. The second value
            is the number of frames after the trial end to keep.
    Output:
        NEURON_ACTIVITY: a numpy matrix; (trials x neurons x frames)
            in size.
    '''
    trial_start = np.asarray(f['trial_start']).astype('int')
    trial_end = np.asarray(f['trial_end']).astype('int')

    C = np.asarray(f['C'])
    assert(np.sum(np.isnan(C)) == 0)
    neuron_activity = np.ones(
        (trial_end.shape[0], C.shape[0], np.sum(t_size) + 1)
        )*np.nan # (num_trials x num_neurons x num_frames)
    for ind, trial in enumerate(trial_end):
        start_idx = max(trial - t_size[0], trial_start[ind])
        aux_act = C[:, start_idx:trial + 1 + t_size[1]]
        neuron_activity[ind, :, -aux_act.shape[1]:] = aux_act
    return neuron_activity
