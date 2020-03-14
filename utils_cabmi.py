# 
#*************************************************************************
#************************ UTILS *****************
#*************************************************************************


__author__ = 'Nuria & Ching & Albert'


import numpy as np
import pdb, os, h5py
from math import sqrt
from collections import deque

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


def median_absolute_deviation(a, axis=None):
    med = np.nanmedian(a, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(a - med), axis=axis)


def time_lock_activity_old(f, t_size=(300,30)):
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
        print(trial, t_size[1], start_idx, trial + 1 + t_size[1])
        aux_act = C[:, start_idx:trial + 1 + t_size[1]]
        neuron_activity[ind, :, -aux_act.shape[1]:] = aux_act
    return neuron_activity


def time_lock_activity(f, t_size=(300,30), order='T'):
    """
    Creates a 3d matrix time-locking activity to trial end.
    Input:
        F: a File object; the experiment HDF5 file
        T_SIZE: an array; the first value is the number of
            frames before the hit we want to keep. The second value
            is the number of frames after the trial end to keep.
        order: char
            order of returned matrix
    Output:
        NEURON_ACTIVITY: a numpy matrix; (neurons x trials x frames)
        in size if order == 'N' else (trials x neurons x frames) .
    """
    trial_start = np.asarray(f['trial_start']).astype('int')
    trial_end = np.asarray(f['trial_end']).astype('int')
    C = f['C']
    assert(np.sum(np.isnan(C)) == 0)
    if order == 'T':
        neuron_activity = np.full(
            (trial_end.shape[0], C.shape[0], np.sum(t_size) + 1),
            np.nan)
    else:
        neuron_activity = np.full(
            (C.shape[0], trial_end.shape[0], np.sum(t_size) + 1),
            np.nan)
    for ind, trial in enumerate(trial_end):
        start_idx = max(trial - t_size[0], trial_start[ind])
        aux_act = C[:, start_idx:trial + 1 + t_size[1]]
        if order == 'T':
            neuron_activity[ind, :, np.sum(t_size) + 1-aux_act.shape[1]:] = aux_act
        else:
            neuron_activity[:, ind, np.sum(t_size) + 1-aux_act.shape[1]:] = aux_act
    return neuron_activity


class OnlineNormalEstimator(object):
    """
    A class to allow rolling calculation of mean and standard deviation.
    Useful especially when processing many GTE matrices. Thanks to:
    http://alias-i.com/lingpipe/docs/api/com/aliasi/stats/
    """

    def __init__(self, algor='welford'):
        # Constructs an instance that has seen no data
        self.mN = 0 # Number of samples
        self.mM = 0.0 # Mean
        self.mS = 0.0 # Sum of squared differences from mean
        if algor == 'welford':
            self.handle = self.handle_welford
            self.unHandle = self.unHandle_welford
            self.mean = self.mean_welford
            self.std = self.std_welford
        elif algor == 'moment':
            self.handle = self.handle_moment
            self.unHandle = self.unHandle_moment
            self.mean = self.mean_moment
            self.std = self.std_moment
        else:
            raise ValueError("Unknown Algorithm: {}".format(algor))

    def handle_welford(self, x):
        # Adds X to the collection of samples for this estimator
        self.mN += 1
        nextM = self.mM + (x - self.mM)/self.mN
        self.mS += (x - self.mM)*(x - nextM)
        self.mM = nextM

    def unHandle_welford(self, x):
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

    def mean_welford(self):
        return self.mM

    def std_welford(self):
        if self.mN > 1:
            return sqrt(self.mS/self.mN)
        else:
            return 0.0

    def handle_moment(self, x):
        # Adds X to the collection of samples for this estimator
        if isinstance(x, np.ndarray):
            self.mN += len(x[~np.isnan(x)])
            self.mS += np.nansum(np.square(x))
            self.mM += np.nansum(x)
        else:
            self.mN += 1
            self.mS += x ** 2
            self.mM += x

    def unHandle_moment(self, x):
        # TODO: FIX THIS
        raise NotImplementedError("Not Implemented Yet")

    def mean_moment(self):
        return self.mM / self.mN

    def std_moment(self):
        if self.mN > 1:
            return sqrt(self.mS / self.mN - self.mean_moment() ** 2)
        else:
            return 0.0

    @staticmethod
    def join(o1, o2):
        # Return joint mean, standard deviation
        mN = o1.mN + o2.mN
        mS = o1.mS + o2.mS
        mM = o1.mM + o2.mM
        m = mM / mN
        return m, sqrt(mS / mN - m ** 2)


class DCache:
    # TODO: AUGMENT IT SUCH THAT IT WORKS FOR MULTIPLE

    def __init__(self, size=20, thres=2, buffer=False, ftype='mean'):
        """
        :param size: int, size of the dampening cache
        :param thres: float, threshold for valid data caching, ignore signal if |x - mu_x| > thres * var
        :param buffer: boolean, for whether keeping a dynamic buffer
        so far cache buffer only accepts 1d input
        """
        self.size = size
        self.thres = thres
        self.counter = 0
        self.bandwidth = None
        self.ftype = ftype
        if ftype == 'median':
            assert buffer, 'median filter requires buffer'
        else:
            assert ftype == 'mean', 'filter type undefined'

        if buffer:
            self.cache = deque()
            self.avg = 0
            self.dev = 0
        else:
            self.cache = None
            self.avg = 0
            self.m2 = 0
            self.dev = 0

    def __len__(self):
        return self.size

    def update_model(self):
        if self.ftype == 'median':
            self.avg = np.nanmedian(self.cache)
            self.dev = np.median(np.abs(np.array(self.cache) - self.avg))
        elif self.cache is not None:
            self.avg = np.nanmean(self.cache)
            self.dev = np.std(self.cache)
        else:
            self.dev = np.sqrt(self.m2 - self.avg ** 2)

    def add(self, signal):
        # handle nans:
        if self.cache is not None:
            assert np.prod(np.array(signal).shape) == 1, 'cache buffer only supports scalar so far'
            if not np.isnan(signal):
                if self.counter < self.size:
                    self.cache.append(signal)
                else:
                    if (signal - self.avg) < self.get_dev() * self.thres:
                        self.cache.append(signal)
                        self.cache.popleft()
                self.counter += 1
        else:
            if self.bandwidth is None:
                self.bandwidth = signal.shape[0]
            if self.counter < self.size:
                if np.sum(np.isnan(signal)) > 0:
                    #print(self.avg, self.avg * (self.counter - 1), (self.avg * self.counter + signal) / (self.counter + 1))
                    self.avg = (self.avg * self.counter + signal) / (self.counter + 1)
                    self.m2 = (signal ** 2 + self.m2 * self.counter) / (self.counter+1)
                    self.counter += 1
            else:
                targets = (~np.isnan(signal)) & ((signal - self.avg) < self.get_dev() * self.thres)
                #print(self.avg, self.avg * (self.size - 1), (self.avg * (self.size - 1) + signal) / self.size)
                self.avg[targets] = (self.avg[targets] * (self.size - 1) + signal[targets]) / self.size
                self.m2[targets] = (signal[targets] ** 2 + self.m2[targets] * (self.size - 1)) / self.size
                self.counter += 1
        self.update_model()

    def get_val(self):
        return self.avg

    def get_dev(self):
        return self.dev


def std_filter(width=20, s=2, buffer=False):
    dc = DCache(width, s, buffer=buffer)

    def fil(sigs, i):
        dc.add(sigs[i])
        #print(sigs[i], dc.get_val())
        return dc.get_val()
    return fil, dc


def median_filter(width=20, s=2):
    dc = DCache(width, s, buffer=True, ftype='median')

    def fil(sigs, i):
        dc.add(sigs[i])
        # print(sigs[i], dc.get_val())
        return dc.get_val()
    return fil, dc
