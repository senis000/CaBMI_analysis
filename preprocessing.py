from scipy.signal import find_peaks_cwt
import numpy as np
import pandas as pd
import h5py, os
from utils_loading import path_prefix_free, file_folder_path, get_PTIT_over_days, \
    parse_group_dict, encode_to_filename
import csv


def calcium_to_peak_times(inputs, low=1, high=20):
    """Returns a pd.DataFrame with peak timing for calcium events
    Params:
        inputs: str, h5py.File, tuple, or np.ndarray
            if str/h5py.File: string that represents the filename of hdf5 file
            if tuple: (path, animal, day), that describes the file location
            if np.ndarray: array C of calcium traces
        out: str
            Output path for saving the metrics in a hdf5 file
            outfile: Animal_Day.csv
                columns: neuron number
    """

    if isinstance(inputs, np.ndarray):
        C = inputs
        animal, day = None, None
        path = './'
    else:
        if isinstance(inputs, str):
            opts = path_prefix_free(inputs, '/').split('_')
            path = file_folder_path(inputs)
            animal, day = opts[1], opts[2]
            f = h5py.File(inputs, 'r')
        elif isinstance(inputs, h5py.File):
            opts = path_prefix_free(inputs.filename, '/').split('_')
            path = file_folder_path(inputs.filename)
            animal, day = opts[1], opts[2]
            f = inputs
        elif isinstance(inputs, tuple):
            path, animal, day = inputs
            hfile = os.path.join(path, animal, day, "full_{}_{}__data.hdf5".format(animal, day))
            f = h5py.File(hfile, 'r')
        else:
            raise RuntimeError("Input Format Unknown!")
        C = np.array(f['C'])
        f.close()
    hyperparams = "low_{}_high_{}".format(low, high)
    if animal is None:
        savepath = os.path.join(path, 'sample_IBI_{}.csv'.format(hyperparams))
    else:
        savepath = os.path.join(path, '{}_{}_{}.csv'.format(animal, day, hyperparams))

    with open(savepath, 'w') as fh:
        cwriter = csv.writer(fh, delimiter=',')
        for i in range(C.shape[0]):
            cwriter.writerow(find_peaks_cwt(C[i, :], np.arange(low, high)))


def calcium_to_peak_times_all(folder, groups, low=1, high=20):
    # TODO: ADD OPTION TO PASS IN A LIST OF METHODS FOR COMPARING THE PLOTS!
    """Calculates Peak Timing and Stores them in csvs for all animal sessions in groups located in folder."""
    processed = os.path.join(folder, 'CaBMI_analysis/processed')

    if groups == '*':
        all_files = get_PTIT_over_days(processed)
    else:
        all_files = {g: parse_group_dict(processed, groups[g], g) for g in groups.keys()}
    print(all_files)

    for group in all_files:
        group_dict = all_files[group]
        for animal in group_dict:
            for day in (group_dict[animal]):
                hf = encode_to_filename(processed, animal, day)
                calcium_to_peak_times(hf, low, high)
