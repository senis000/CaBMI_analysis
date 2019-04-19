import numpy as np
import h5py
import time
import pdb
import matplotlib.pyplot as plt

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
    padding = 10 # Allow at most 10 frames for time-shifted correlations
    min_frames = 100
    frame_size = num_frames - padding
    if num_frames < min_frames:
        raise ValueError("Signal length is not long enough.")
    Y = np.zeros((padding, frame_size))
    for i in range(padding):
        Y[i,:] = y[i:i + frame_size]
    max_cc = 0
    for i in range(padding):
        cc = np.nanmax(np.corrcoef(Y, x[i:i + frame_size]))
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
