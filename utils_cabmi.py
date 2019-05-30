# 
#*************************************************************************
#************************ UTILS *****************
#*************************************************************************


__author__ = 'Nuria & Ching & Albert'


import numpy as np
import pdb
from math import sqrt   

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

class OnlineNormalEstimator(object):
    """
    A class to allow rolling calculation of mean and standard deviation.
    Useful especially when processing many GTE matrices. Thanks to:
    http://alias-i.com/lingpipe/docs/api/com/aliasi/stats/
    """

    def __init__(self):
        # Constructs an instance that has seen no data
        self.mN = 0 # Number of samples
        self.mM = 0.0 # Mean
        self.mS = 0.0 # Sum of squared differences from mean

    def handle(self, x):
        # Adds X to the collection of samples for this estimator
        self.mN += 1
        nextM = self.mM + (x - self.mM)/self.mN
        self.mS += (x - self.mM)*(x - nextM)
        self.mM = nextM

    def unHandle(self, x):
        # Removes the specified value from the sample set
        assert(self.mN != 0)
        if (self.mN ==1):
            self.mN = 0
            self.mM = 0.0
            self.mS = 0.0
        mOld = (self.mN*self.mM - x)/(self.mN - 1)
        self.mS -= (x - self.mM)*(x - mOld)
        self.mM = mOld
        self.mN -= 1

    def mean(self):
        return self.mM

    def std(self):
        if self.mN > 1:
            return sqrt(self.mS/self.mN)
        else:
            return 0.0