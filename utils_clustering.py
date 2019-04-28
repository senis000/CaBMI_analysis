import numpy as np
import h5py
import time
import pdb
import matplotlib.pyplot as plt
import scipy.signal

def sort_matrix_by_clusters(matrix, labels):
    """
    Given a square MATRIX, this function will sort the matrix by LABELS.
    Inputs:
        MATRIX: A numpy matrix; a (neurons x neurons) distance matrix.
        LABELS: A numpy array of size (neurons x 1). Contains numerical,
            zero-indexed labels that indicates what cluster each neuron is in.
    Outputs:
        SORTED_MAT: The sorted version of MATRIX
        SORTING: A numpy array of size(neurons x 1), indicating the new indices
            of each neuron when sorted by LABELS. Specifically, neuron i in the
            original matrix will be at index SORTING_i in the sorted matrix.
    """

    sorting = np.zeros(labels.shape)
    sorted_mat = np.zeros(matrix.shape)
    grouped_arrays = []
    num_neurons = labels.size
    num_clusters = np.max(labels) + 1
    for i in range(num_clusters):
        grouped_arrays.append(np.argwhere(labels==i))
    new_idx = 0
    for group in grouped_arrays:
        for old_idx in group:
            sorting[old_idx] = new_idx
            new_idx += 1
    for i in range(num_neurons):
        for j in range(num_neurons):
            new_i = sorting[i]
            new_j = sorting[j]
            sorted_mat[new_i, new_j] = matrix[i,j]
    return sorted_mat, sorting

def normalized_cc(x, y): # The affinity function
    num_frames = x.size
    padding = 1 # Allow at most 4 frames for time-shifted correlations
    min_frames = 100
    frame_size = num_frames - padding
    if num_frames < min_frames:
        raise ValueError("Signal length is not long enough.")
    max_cc = 0
    for i in range(padding): # For different time-shifts of x 
        X = x[i:i + frame_size] # Time-shifted window of x; the template signal
        Y = y[:frame_size + padding] # The template will be slid along Y
        corr_coefs = correlate_template(Y, X)
        cc = np.nanmax(corr_coefs)
        if abs(cc) > abs(max_cc):
            max_cc = cc
    return max_cc

def normalized_cc_mat(X): # The affinity function
    num_neurons, _ = X.shape
    distance_mat = np.zeros((num_neurons, num_neurons))
    for i in range(num_neurons):
        for j in range(num_neurons):
            max_cc = normalized_cc(X[i,:], X[j,:])
            distance_mat[i,j] = max_cc
    return distance_mat


# Below are cross-correlation functions, written by the Github user trichter,
# and included within this file for sake of a cleaner repo.

def _pad_zeros(a, num, num2=None):
    """Pad num zeros at both sides of array a"""
    if num2 is None:
        num2 = num
    hstack = [np.zeros(num, dtype=a.dtype), a, np.zeros(num2, dtype=a.dtype)]
    return np.hstack(hstack)


def _xcorr_padzeros(a, b, shift, method):
    """
    Cross-correlation using SciPy with mode='valid' and precedent zero padding
    """
    if shift is None:
        shift = (len(a) + len(b) - 1) // 2
    dif = len(a) - len(b) - 2 * shift
    if dif > 0:
        b = _pad_zeros(b, dif // 2)
    else:
        a = _pad_zeros(a, -dif // 2)
    return scipy.signal.correlate(a, b, 'valid', method)


def _xcorr_slice(a, b, shift, method):
    """
    Cross-correlation using SciPy with mode='full' and subsequent slicing
    """
    mid = (len(a) + len(b) - 1) // 2
    if shift is None:
        shift = mid
    if shift > mid:
        # Such a large shift is not possible without zero padding
        return _xcorr_padzeros(a, b, shift, method)
    cc = scipy.signal.correlate(a, b, 'full', method)
    return cc[mid - shift:mid + shift + len(cc) % 2]


def get_lags(cc):
    """
    Return array with lags
    :param cc: Cross-correlation returned by correlate_maxlag.
    :return: lags
    """
    mid = (len(cc) - 1) / 2
    if len(cc) % 2 == 1:
        mid = int(mid)
    return np.arange(len(cc)) - mid


def correlate_maxlag(a, b, maxlag, demean=True, normalize='naive',
                     method='auto'):
    """
    Cross-correlation of two signals up to a specified maximal lag.
    This function only allows 'naive' normalization with the overall
    standard deviations. This is a reasonable approximation for signals of
    similar length and a relatively small maxlag parameter.
    :func:`correlate_template` provides correct normalization.
    :param a,b: signals to correlate
    :param int maxlag: Number of samples to shift for cross correlation.
        The cross-correlation will consist of ``2*maxlag+1`` or
        ``2*maxlag`` samples. The sample with zero shift will be in the middle.
    :param bool demean: Demean data beforehand.
    :param normalize: Method for normalization of cross-correlation.
        One of ``'naive'`` or ``None``
        ``'naive'`` normalizes by the overall standard deviation.
        ``None`` does not normalize.
    :param method: correlation method to use.
        See :func:`scipy.signal.correlate`.
    :return: cross-correlation function.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if demean:
        a = a - np.mean(a)
        b = b - np.mean(b)
    # choose the usually faster xcorr function for each method
    _xcorr = _xcorr_padzeros if method == 'direct' else _xcorr_slice
    cc = _xcorr(a, b, maxlag, method)
    if normalize == 'naive':
        norm = (np.sum(a ** 2) * np.sum(b ** 2)) ** 0.5
        if norm <= np.finfo(float).eps:
            # norm is zero
            # => cross-correlation function will have only zeros
            cc[:] = 0
        elif cc.dtype == float:
            cc /= norm
        else:
            cc = cc / norm
    elif normalize is not None:
        raise ValueError("normalize has to be one of (None, 'naive'))")
    return cc


def _window_sum(data, window_len):
    """Rolling sum of data"""
    window_sum = np.cumsum(data)
    # in-place equivalent of
    # window_sum = window_sum[window_len:] - window_sum[:-window_len]
    # return window_sum
    np.subtract(window_sum[window_len:], window_sum[:-window_len],
                out=window_sum[:-window_len])
    return window_sum[:-window_len]


def correlate_template(data, template, mode='valid', demean=True,
                       normalize='full', method='auto'):
    """
    Normalized cross-correlation of two signals with specified mode.
    If you are interested only in a part of the cross-correlation function
    around zero shift use :func:`correlate_maxlag` which allows to
    explicetly specify the maximum lag.
    :param data,template: signals to correlate. Template array must be shorter
        than data array.
    :param normalize:
        One of ``'naive'``, ``'full'`` or ``None``.
        ``'full'`` normalizes every correlation properly,
        whereas ``'naive'`` normalizes by the overall standard deviations.
        ``None`` does not normalize.
    :param mode: correlation mode to use.
        See :func:`scipy.signal.correlate`.
    :param bool demean: Demean data beforehand.
        For ``normalize='full'`` data is demeaned in different windows
        for each correlation value.
    :param method: correlation method to use.
        See :func:`scipy.signal.correlate`.
    :return: cross-correlation function.
    .. note::
        Calling the function with ``demean=True, normalize='full'`` (default)
        returns the zero-normalized cross-correlation function.
        Calling the function with ``demean=False, normalize='full'``
        returns the normalized cross-correlation function.
    """
    data = np.asarray(data)
    template = np.asarray(template)
    lent = len(template)
    if len(data) < lent:
        raise ValueError('Data must not be shorter than template.')
    if demean:
        template = template - np.mean(template)
        if normalize != 'full':
            data = data - np.mean(data)
    cc = scipy.signal.correlate(data, template, mode, method)
    if normalize is not None:
        tnorm = np.sum(template ** 2)
        if normalize == 'naive':
            norm = (tnorm * np.sum(data ** 2)) ** 0.5
            if norm <= np.finfo(float).eps:
                cc[:] = 0
            elif cc.dtype == float:
                cc /= norm
            else:
                cc = cc / norm
        elif normalize == 'full':
            pad = len(cc) - len(data) + lent
            if mode == 'same':
                pad1, pad2 = (pad + 2) // 2, (pad - 1) // 2
            else:
                pad1, pad2 = (pad + 1) // 2, pad // 2
            data = _pad_zeros(data, pad1, pad2)
            # in-place equivalent of
            # if demean:
            #     norm = ((_window_sum(data ** 2, lent) -
            #              _window_sum(data, lent) ** 2 / lent) * tnorm) ** 0.5
            # else:
            #      norm = (_window_sum(data ** 2, lent) * tnorm) ** 0.5
            # cc = cc / norm
            if demean:
                norm = _window_sum(data, lent) ** 2
                if norm.dtype == float:
                    norm /= lent
                else:
                    norm = norm / lent
                np.subtract(_window_sum(data ** 2, lent), norm, out=norm)
            else:
                norm = _window_sum(data ** 2, lent)
            norm *= tnorm
            if norm.dtype == float:
                np.sqrt(norm, out=norm)
            else:
                norm = np.sqrt(norm)
            mask = norm <= np.finfo(float).eps
            if cc.dtype == float:
                cc[~mask] /= norm[~mask]
            else:
                cc = cc / norm
            cc[mask] = 0
        else:
            msg = "normalize has to be one of (None, 'naive', 'full')"
            raise ValueError(msg)
    return cc
