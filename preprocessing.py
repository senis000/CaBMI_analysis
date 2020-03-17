from scipy.signal import find_peaks_cwt
import numpy as np
import pandas as pd
import h5py, os
from utils_loading import path_prefix_free, file_folder_path, get_PTIT_over_days, \
    parse_group_dict, encode_to_filename, find_file_regex, get_all_animals, decode_from_filename
from utils_cabmi import median_absolute_deviation
import csv
import multiprocessing as mp


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
    if isinstance(processed, str):
        hfile = h5py.File(os.path.join(processed, animal, day, "full_{}_{}__data.hdf5".format(animal, day)),
                   'r')
    else:
        hfile = processed
    N = hfile['C'].shape[0]
    rois = np.full(N, "D", dtype="U2")
    nerden = np.array(hfile['nerden'])
    redlabel = np.array(hfile['redlabel'])
    ens_neur = np.array(hfile['ens_neur'])
    e2_neur = ens_neur[hfile['e2_neur']] if 'e2_neur' in hfile else None
    if isinstance(processed, str):
        hfile.close()
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



def digitize_signal(sigs, ns, axis=None, minbias=True):
    mins = np.nanmin(sigs, axis=axis, keepdims=True)
    maxes = np.nanmax(sigs, axis=axis, keepdims=True)
    ranges = maxes - mins
    def segment(n):
        steps = ranges / n
        if minbias:
            res = np.ceil((sigs - mins) / steps).astype(np.int)
            res[res > 0] -= 1
        else:
            res = np.floor((sigs - mins) / steps).astype(np.int)
            res[res == n] = n-1
        assert np.max(res) == n-1
        return res

    if hasattr(ns, '__iter__'):
        return [segment(n) for n in ns]
    else:
        return segment(ns)


def digitize_calcium(inputs, source, n, out):
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
        S = inputs
        animal, day = None, None
        path = './'
        savepath = os.path.join(path, 'sample_IBI_{}.csv')
    else:
        if isinstance(inputs, str):
            opts = path_prefix_free(inputs, '/').split('_')
            path = file_folder_path(inputs)
            animal, day = opts[1], opts[2]
            f = None
            hfile = inputs
        elif isinstance(inputs, tuple):
            path, animal, day = inputs
            f1 = os.path.join(path, animal, "full_{}_{}__data.hdf5".format(animal, day))
            f2 = encode_to_filename(path, animal, day)
            if os.path.exists(f1):
                hfile = f1
            elif os.path.exists(f2):
                hfile = f2
            else:
                raise FileNotFoundError("File {} or {} not found".format(f1, f2))
            f = None
        elif isinstance(inputs, h5py.File):
            opts = path_prefix_free(inputs.filename, '/').split('_')
            path = file_folder_path(inputs.filename)
            animal, day = opts[1], opts[2]
            f = inputs
        else:
            raise RuntimeError("Input Format Unknown!")
        savepath = os.path.join(path, '%s_%s_rawcwt_{}.csv' % (animal, day))
        if os.path.exists(savepath):
            return
        if f is None:
            f = h5py.File(hfile, 'r')
        S = np.array(f[source])
        f.close()
    dgs = digitize_signal(S, n)
    hyperparams = "n_{}".format(n)
    savepath = savepath.format(hyperparams)
    with open(savepath, 'w') as fh:
        cwriter = csv.writer(fh, delimiter=',')
        for i in range(S.shape[0]):
            print(i)
            cwriter.writerow(dgs[i])
    return savepath


def digitize_calcium_by_animal(folder, animal, days, source, n):
    for day in days:
        if day.isnumeric():
            hf = encode_to_filename(folder, animal, day)
            digitize_calcium(hf, source, n)


def digitize_calcium_all(folder, groups, source, ns, nproc=1):
    # TODO: ADD OPTION TO PASS IN A LIST OF METHODS FOR COMPARING THE PLOTS!
    """Calculates Peak Timing and Stores them in csvs for all animal sessions in groups located in folder."""
    processed = os.path.join(folder, 'CaBMI_analysis/processed')
    logfolder = os.path.join(processed, 'log')
    if not os.path.exists(logfolder):
        os.makedirs(logfolder)
    all_files = parse_group_dict(processed, groups, 'all')
    print(all_files)

    for n in ns:
        if nproc == 0:
            nproc = mp.cpu_count()
        if nproc == 1:
            for animal in all_files:
                for day in (all_files[animal]):
                    print(animal, day)
                    hf = encode_to_filename(processed, animal, day)
                    digitize_calcium(hf, source, n)
        else:
            p = mp.Pool(nproc)
            p.starmap_async(digitize_calcium_by_animal,
                            [(processed, animal, all_files[animal], source, n) for animal in all_files])
        with open("dCalcium_n_{}.txt".format(n)) as f:
            f.write("done")


def move_typhos(folder):
    # check date in micelog
    for animal in get_all_animals(folder):
        animal_path = os.path.join(folder, animal)
        for day in os.listdir(animal_path):
            if day[-5:] == '.hdf5':
                _, d = decode_from_filename(day)
                daydir = os.path.join(animal_path, d)
                if not os.path.exists(daydir):
                    os.makedirs(daydir)
                os.rename(os.path.join(animal_path, day), os.path.join(daydir, day))
            elif day.isnumeric():
                daypath = os.path.join(animal_path, day)
                for f in os.listdir(daypath):
                    if f == 'onlineSNR.hdf5':
                        os.rename(os.path.join(daypath, f), os.path.join(daypath,
                                                                         f'onlineSNR_{animal}_{day}.hdf5'))


def regularize_directory(folder):
    # TODO: rename ALL dffSNRs from snr_ens to snr_dff
    # check date in micelog
    processed = os.path.join(folder, 'processed')
    utils = os.path.join(folder, 'utils')

    def hdf5_to_utils(f, animal, fpath):
        subdir = f.split('_')[0]
        animal_subdir = os.path.join(utils, subdir, animal)
        if not os.path.exists(animal_subdir):
            os.makedirs(animal_subdir)
        os.rename(os.path.join(fpath, f), os.path.join(animal_subdir, f))
    for animal in get_all_animals(processed):
        animal_path = os.path.join(processed, animal)

        for day in os.listdir(animal_path):
            if day[-5:] == '.hdf5' and day[:4] != 'full':
                hdf5_to_utils(day, animal, animal_path)
            elif day.isnumeric():
                daypath = os.path.join(animal_path, day)
                hdf5only = True
                for f in os.listdir(daypath):
                    if f[-4:] != '.hdf5':
                        print(f)
                        hdf5only = False
                    if f[:4] == 'full':
                        os.rename(os.path.join(daypath, f), os.path.join(animal_path, f))
                    else:
                        hdf5_to_utils(f, animal, daypath)
                if hdf5only:
                    os.removedirs(daypath)


if __name__ == '__main__':
    home = "/home/user/"
    digitize_calcium_all(home, "*", 'dff', [2, 3, 4, 5, 6])
