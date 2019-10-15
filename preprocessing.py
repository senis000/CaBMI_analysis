from scipy.signal import find_peaks_cwt
import numpy as np
import pandas as pd
import h5py, os
from utils_loading import path_prefix_free, file_folder_path, get_PTIT_over_days, \
    parse_group_dict, encode_to_filename, find_file_regex
from utils_cabmi import median_absolute_deviation
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
    hyperparams = "low_{}_high_{}".format(low, high)
    if isinstance(inputs, np.ndarray):
        C = inputs
        animal, day = None, None
        path = './'
        savepath = os.path.join(path, 'sample_IBI_{}.csv'.format(hyperparams))
    else:
        if isinstance(inputs, str):
            opts = path_prefix_free(inputs, '/').split('_')
            path = file_folder_path(inputs)
            animal, day = opts[1], opts[2]
            f = None
            hfile = inputs
        elif isinstance(inputs, tuple):
            path, animal, day = inputs
            hfile = os.path.join(path, animal, day, "full_{}_{}__data.hdf5".format(animal, day))
            f = None
        elif isinstance(inputs, h5py.File):
            opts = path_prefix_free(inputs.filename, '/').split('_')
            path = file_folder_path(inputs.filename)
            animal, day = opts[1], opts[2]
            f = inputs
        else:
            raise RuntimeError("Input Format Unknown!")
        savepath = os.path.join(path, '{}_{}_rawcwt_{}.csv'.format(animal, day, hyperparams))
        cwt = os.path.join(path, 'cwt.txt')
        if os.path.exists(savepath):
            return
        if f is None:
            f = h5py.File(hfile, 'r')
        C = np.array(f['C'])
        f.close()

    with open(savepath, 'w') as fh:
        cwriter = csv.writer(fh, delimiter=',')
        for i in range(C.shape[0]):
            print(i)
            cwriter.writerow(find_peaks_cwt(C[i, :], np.arange(low, high)))
    if animal is not None:
        with open(cwt, 'a') as cf:
            cf.write(hyperparams + "\n")
    return savepath


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
                print(animal, day)
                hf = encode_to_filename(processed, animal, day)
                calcium_to_peak_times(hf, low, high)



def get_roi_type(processed, animal, day):
    rois = None
    with h5py.File(os.path.join(processed, animal, day, "full_{}_{}__data.hdf5".format(animal, day)),
                   'r') as hfile:
        N = hfile['C'].shape[0]
        rois = np.full(N, "D", dtype="U2")
        nerden = np.array(hfile['nerden'])
        redlabel = np.array(hfile['redlabel'])
        ens_neur = np.array(hfile['ens_neur'])
        e2_neur = ens_neur[hfile['e2_neur']] if 'e2_neur' in hfile else None
        rois[nerden & ~redlabel] = 'IG'
        rois[nerden & redlabel] = 'IR'
        if e2_neur is not None:
            rois[ens_neur] = 'E1'
            rois[e2_neur] = 'E2'
        else:
            rois[ens_neur] = 'E'
    return rois


def get_peak_times_over_thres(inputs, window, method, tlock=30):
    """ Returns Peak Times, organized by window bins and trial bins respectively, that Passes a specific
    threshold specified by method. """
    if isinstance(inputs, str):
        opts = path_prefix_free(inputs, '/').split('_')
        path = file_folder_path(inputs)
        session_path = path
        animal, day = opts[1], opts[2]
        f = None
        hfile = inputs
    elif isinstance(inputs, tuple):
        path, animal, day = inputs
        session_path = os.path.join(path, animal, day)
        hfile = os.path.join(session_path, "full_{}_{}__data.hdf5".format(animal, day))
        f = None
    elif isinstance(inputs, h5py.File):
        opts = path_prefix_free(inputs.filename, '/').split('_')
        path = file_folder_path(inputs.filename)
        session_path = path
        animal, day = opts[1], opts[2]
        f = inputs
    else:
        raise RuntimeError("Input Format Unknown!")
    cwt_pattern = '{}_{}_rawcwt_low_(\d+)_high_(\d+).csv'.format(animal, day)
    cwtfile = find_file_regex(session_path, cwt_pattern)
    if cwtfile is None:
        print("({}, {}) requires preprocessing!".format(animal, day))
        cwtfile = calcium_to_peak_times((path, animal, day))
    if f is None:
        f = h5py.File(hfile, 'r')
    C = np.array(f['C'])
    trial_start = np.array(f['trial_start'])
    trial_end = np.array(f['trial_end'])
    array_hit = np.array(f['array_t1'])
    array_miss = np.array(f['array_miss'])
    blen = f.attrs['blen']
    f.close()
    print(animal, day)

    opt, th = method // 10, method % 10
    dispersion = median_absolute_deviation if opt else np.nanstd
    T = C.shape[1]
    slides = int(np.ceil(T / window))
    with open(cwtfile) as cwtstream:
        creader = csv.reader(cwtstream)
        D_trial = {}
        D_window = {}
        for i, row in enumerate(creader):
            c = C[i]
            D_window[i] = {s: [] for s in range(slides)}
            D_trial[i] = {t: [] for t in range(len(trial_start))}
            t = 0
            s = 0
            s_end = min(window, T)
            thres = np.nanmean(c) + dispersion(c) * th # Use the entire signal as a criteria for evaluating
            # large events
            for j in range(len(row)):
                p = int(row[j])
                if p >= s_end:
                    s += 1
                    D_window[i][s] = []
                    s_end = min(s_end + window, T)
                elif c[p] >= thres:
                    D_window[i][s].append(p)

                if p <= blen:
                    pass
                elif t >= len(trial_start):
                    pass
                #     # if i == 0:
                #     #     print("Reaches End, dropping future frames ({}/{})".format(p, trial_end[-1] + tlock))
                else:
                    # if t > 0 and trial_start[t] - trial_end[t-1] > tlock and i == 0:
                    #     print("trial {}, out of ({}, {}, prev {}), diff:{}, {}".format(t, trial_start[t], trial_end[t], trial_end[t-1], trial_start[t]-trial_end[t-1], HM))
                    if p > trial_end[t] + tlock:
                        # if t < len(trial_start) -1 and p > trial_start[t+1]:
                        #     print("Frame overflow into next trial bin {}, (end: {}, start: {})"
                        #           .format(t, trial_end[t], trial_start[t+1]))`
                        t+=1
                        if t < len(trial_start):
                            if p >= trial_start[t] and c[p] >= thres:
                                D_trial[i][t].append(p)
                    elif p >= trial_start[t] and c[p] >= thres:
                            D_trial[i][t].append(p)
                    # elif i == 0:
                    #     HM = "hit" if t in array_hit else "miss"
                    #     print("trial {}, Out of bin frame: {}".format(t, p))

    return D_trial, D_window


if __name__ == '__main__':
    home = "/home/user/"
    calcium_to_peak_times_all(home, "*")
