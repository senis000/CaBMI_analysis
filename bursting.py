import numpy as np
import json
import seaborn as sns
import os, h5py
from utils_bursting import *
from plotting_functions import best_nbins
from utils_loading import get_PTIT_over_days, path_prefix_free, \
    decode_from_filename, encode_to_filename, get_redlabel, parse_group_dict, decode_method_ibi
from utils_cabmi import time_lock_activity
import matplotlib.pyplot as plt
from scipy import io
from matplotlib.widgets import Slider


def calcium_IBI_single_session_windows(inputs, out, window=None, perc=30, ptp=True):
    """Returns a metric matrix and meta data of IBI metric
    Params:
        inputs: str, h5py.File, tuple, or np.ndarray
            if str/h5py.File: string that represents the filename of hdf5 file
            if tuple: (path, animal, day), that describes the file location
            if np.ndarray: array C of calcium traces
        out: str
            Output path for saving the metrics in a hdf5 file
            outfile: h5py.File
                N: number of neurons
                s: number of sliding sessions
                K: maximum number of IBIs extracted
                'mean': N * s matrix, means of IBIs
                'stds': N * s matrix, stds of IBIs
                'CVs': N * s matrix, CVs of IBIs
                'IBIs': N * s * K, IBIs
        window: None or int
            sliding window for calculating IBIs.
            if None, use 'blen' in hdf5 file instead, but inputs have to be str/h5py.File
        perc: float
            hyperparameter for partitioning algorithm, correlated with tail length of splitted calcium trace
        ptp: boolean
            True if IBI is based on peak to peak measurement, otherwise tail to tail

    Alternatively, could store data in:
        mat_ibi: np.ndarray
            N * s * m matrix, , where N is the number of neurons, s is number of sliding sessions,
            m is the number of metrics
        meta: dictionary
            meta data of form {axis: labels}
    """
    if isinstance(inputs, np.ndarray):
        C = inputs
        window = C.shape[1]
        animal, day = None, None
    else:
        if isinstance(inputs, str):
            opts = path_prefix_free(inputs, '/').split('_')
            animal, day = opts[1], opts[2]
            f = h5py.File(inputs, 'r')
        elif isinstance(inputs, h5py.File):
            opts = path_prefix_free(inputs.filename, '/').split('_')
            animal, day = opts[1], opts[2]
            f = inputs
        elif isinstance(inputs, tuple):
            path, animal, day = inputs
            hfile = os.path.join(path, animal, day, "full_{}_{}__data.hdf5".format(animal, day))
            f = h5py.File(hfile, 'r')
        else:
            raise RuntimeError("Input Format Unknown!")
        C = np.array(f['C'])
        if window is None:
            window0 = window
            window = f.attrs['blen']
        f.close()
    if animal is None:
        savepath = os.path.join(out, 'sample_IBI.hdf5')
    else:
        hyperparams = 'theta_perc{}{}_window{}'.format(perc, '_ptp' if ptp else "", window0)
        savepath = os.path.join(out, animal, day)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        savepath = os.path.join(savepath, "IBI_{}_{}_{}.hdf5".format(animal, day, hyperparams))
    if os.path.exists(savepath):
        with h5py.File(savepath, 'r') as f:
            N, nsessions = f['mean'].shape[:2]
        return savepath, N, nsessions
    nsessions = int(np.ceil(C.shape[1] / window))
    rawibis = {}
    maxLen = -1
    for i in range(C.shape[0]):
        rawibis[i] = {}
        for s in range(nsessions):
            slide = C[i, s*window:min(C.shape[1], (s+1) * window)]
            ibis = neuron_calcium_ipri(slide, perc, ptp)
            rawibis[i][s] = ibis
            maxLen = max(len(ibis), maxLen)

    all_ibis = np.full((C.shape[0], nsessions, maxLen), np.nan)
    for i in range(C.shape[0]):
        for s in range(nsessions):
            all_ibis[i][s][:len(rawibis[i][s])] = rawibis[i][s]
    means = np.nanmean(all_ibis, axis=2)
    stds = np.nanstd(all_ibis, axis=2)
    cvs = stds / means
    outfile = h5py.File(savepath, 'w-')
    outfile['mean'], outfile['stds'], outfile['CVs'] = means, stds, cvs
    outfile['IBIs'] = all_ibis
    outname = outfile.filename
    outfile.close()
    return outname, C.shape[0], nsessions
    #return np.concatenate([means, stds, cvs], axis=2), {2: ['mean', 'stds', 'CVs']}


def calcium_IBI_all_sessions_windows(folder, window=None, perc=30, ptp=True, IBI_dist=False):
    """Returns a metric matrix across all sessions and meta data of IBI metric
        Params:
            folder: str
                root folder path where all the processed hdf5 will be stored
            out: str
                Output path for saving the metrics in a hdf5 file
                outfile: h5py.File
                    N: number of neurons
                    s: number of sliding sessions
                    K: maximum number of IBIs extracted
                    'mean': N * s matrix, means of IBIs
                    'stds': N * s matrix, stds of IBIs
                    'CVs': N * s matrix, CVs of IBIs
                    'IBIs': N * s * K, IBIs
                All stored by animal/day/IBI_animal_day_hyperparams.hdf5
            window: None or int
                sliding window for calculating IBIs.
                if None, use 'blen' in hdf5 file instead, but inputs have to be str/h5py.File
            perc: float
                hyperparameter for partitioning algorithm, correlated with tail length of splitted calcium trace
            ptp: boolean
                True if IBI is based on peak to peak measurement, otherwise tail to tail
            IBI_dist: boolean
                generate the IBI_distribution matrix if True

        Returns:
            mats: {group: {mat_ibi, (mat_ibi_dist,) meta}}
            mat_ibi: np.ndarray (first 4 ~ 8.93MB)
                A * D * N * s * m matrix,
                A: number of animals
                D: number of days
                N: number of neurons
                s: number of sliding sessions,
                m is the number of metrics

            meta: dictionary
                meta data of form {group: {axis: labels}}

        IO:
            summary.mat: dict
                {group: (A, D, N, s, nibis)}, first four the dimension of the ibi metric matrix,
                nibis is the maximum number of ibis
        """
    processed = os.path.join(folder, 'CaBMI_analysis/processed')
    out = os.path.join(folder, 'bursting/IBI')
    if 'navigation.json' in os.listdir(processed):
        with open(os.path.join(processed, 'navigation.json'), 'r') as jf:
            all_files = json.load(jf)
    else:
        all_files = get_PTIT_over_days(processed)
    calculate = True
    summary_file = os.path.join(out, 'summary.json')
    summary_mat = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as jf:
            summary_mat = json.load(jf)
        calculate = False
    mats = {}
    hyperparam = 'theta_perc{}{}_window{}'.format(perc, '_ptp' if ptp else "", window)
    for group in 'IT', 'PT':
        animal_map = all_files[group]['maps']
        mats[group] = {'meta': [''] * len(animal_map)}
        if calculate:
            summary_mat[group] = [len(animal_map), len(all_files[group]) - 1] + [0] * 3
            temp = {}
        else:
            mats[group]['mat_ibi'] = np.full(tuple(summary_mat[group][:4]) + (3,), np.nan)
            if IBI_dist:
                mats[group]['mat_ibi_dist'] = np.full(summary_mat[group], np.nan)
        for d in all_files[group]:
            if d == 'maps':
                continue
            animal_files = all_files[group][d]
            if calculate:
                temp[d] = {}
            for filename in animal_files:
                print(filename)
                animal, day = decode_from_filename(filename)
                if calculate:
                    burst_file = calcium_IBI_single_session((processed, animal, day),
                                            out, window=window, perc=perc, ptp=ptp)[0]
                else:
                    burst_file = os.path.join(out, animal, day,
                                              encode_to_filename(out, animal, day, hyperparam))
                #try:
                burst_data = h5py.File(burst_file, 'r')
                metrics = np.stack((burst_data['mean'], burst_data['stds'], burst_data['CVs']), axis=-1)
                animal_ind = animal_map[animal]
                mats[group]['redlabels'][animal_ind, int(d)-1] = get_redlabel(processed, animal, day)
                if calculate:
                    temp[d][animal] = {'mat_ibi': metrics}
                    if IBI_dist:
                        temp[d][animal]['mat_ibi_dist'] = burst_data['IBIs']
                    summary_mat[group][2] = max(metrics.shape[0], summary_mat[group][2])
                    summary_mat[group][3] = max(metrics.shape[1], summary_mat[group][3])
                    summary_mat[group][4] = max(burst_data['IBIs'].shape[-1], summary_mat[group][4])
                else:
                    temp = {'mat_ibi': metrics}
                    if IBI_dist:
                        temp['mat_ibi_dist'] = burst_data['IBIs']
                    for opt in mats[group]:
                        if opt == 'meta':
                            continue
                        animal_ind = animal_map[animal]
                        tN, ts, tm = temp[opt].shape
                        mats[group][opt][animal_ind, int(d) - 1, :tN, :ts, :tm] = temp[opt]
                        mats[group]['meta'][animal_ind] = animal
                # except Exception as e:
                #     skipped.append(e.args)
                #     print(e.args)
        #try:
        summary_mat[group] = tuple(summary_mat[group])
        if calculate:
            mats[group]['mat_ibi'] = np.full(summary_mat[group][:4] + (3,), np.nan)
            mats[group]['redlabels'] = np.full(summary_mat[group][:3], False)
            if IBI_dist:
                mats[group]['mat_ibi_dist'] = np.full(summary_mat[group], np.nan)
            for opt in mats[group]:
                if opt == 'meta':
                    continue
                for d in temp:
                    for animal in temp[d]:
                        animal_ind = animal_map[animal]
                        tN, ts, tm = temp[d][animal][opt].shape
                        mats[group][opt][animal_ind, int(d)-1, :tN, :ts, :tm] = temp[d][animal][opt]
                        mats[group]['meta'][animal_ind] = animal
        """except Exception as e:
            skipped.append(str(e.args))
            print(e.args)"""
    if calculate:
        with open(summary_file, 'w') as jf:
            json.dump(summary_mat, jf)
    """f = open(os.path.join(folder, 'errLOG.txt'), 'w')
    f.write("\n".join([str(s) for s in skipped]))
    f.close()"""
    return mats


def calcium_IBI_single_session(inputs, out, window=None, method=0):
    """Returns a metric matrix and meta data of IBI metric
    Params:
        inputs: str, h5py.File, tuple, or np.ndarray
            if str/h5py.File: string that represents the filename of hdf5 file
            if tuple: (path, animal, day), that describes the file location
            if np.ndarray: array C of calcium traces
        out: str
            Output path for saving the metrics in a hdf5 file
            outfile: h5py.File
                N: number of neurons
                s: number of sliding sessions
                t: number of trials
                K: maximum number of IBIs extracted
                K': maximum number of IBIs within each trial
                'IBIs_window': N * s * K, IBIs across window
                'IBIs_trial': N * t * K', IBIs across trial
        window: None or int
            sliding window for calculating IBIs.
            if None, use 'blen' in hdf5 file instead, but inputs have to be str/h5py.File
        method: int/float
            if negative:
                Use signal_partition algorithm in shuffling_functions.py, the absolute value is the perc
                parameter
                perc: float
                    hyperparameter for partitioning algorithm, correlated with tail length of splitted calcium trace
                if method < -100:
                    ptp = False
                    ptp: boolean
                        True if IBI is based on peak to peak measurement, otherwise tail to tail
            Else:
                opt, thres = method // 10, method % 10
                opt: 0: std
                     1: mad
                thres: number of std/mad
    Alternatively, could store data in:
        mat_ibi: np.ndarray
            N * s * m matrix, , where N is the number of neurons, s is number of sliding sessions,
            m is the number of metrics
        meta: dictionary
            meta data of form {axis: labels}
    """
    if method == 0:
        return [calcium_IBI_single_session(inputs, out, window, m) for m in (1, 2, 11, 12)]
    if isinstance(inputs, np.ndarray):
        C = inputs
        t_locks = None
        window = C.shape[1]
        animal, day = None, None
    else:
        if isinstance(inputs, str):
            opts = path_prefix_free(inputs, '/').split('_')
            animal, day = opts[1], opts[2]
            f = h5py.File(inputs, 'r')
        elif isinstance(inputs, h5py.File):
            opts = path_prefix_free(inputs.filename, '/').split('_')
            animal, day = opts[1], opts[2]
            f = inputs
        elif isinstance(inputs, tuple):
            path, animal, day = inputs
            hfile = os.path.join(path, animal, day, "full_{}_{}__data.hdf5".format(animal, day))
            f = h5py.File(hfile, 'r')
        else:
            raise RuntimeError("Input Format Unknown!")
        C = np.array(f['C'])
        t_locks = time_lock_activity(f, order='N')
        if window is None:
            window0 = window
            window = f.attrs['blen']
        f.close()
    ibi_func, hp = decode_method_ibi(method)
    if animal is None:
        savepath = os.path.join(out, 'sample_IBI.hdf5')
    else:
        hyperparams = 'theta_{}_window{}'.format(hp, window0)
        savepath = os.path.join(out, animal, day)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        savepath = os.path.join(savepath, "IBI_{}_{}_{}.hdf5".format(animal, day, hyperparams))
    if os.path.exists(savepath):
        with h5py.File(savepath, 'r') as f:
            N, nsessions = f['mean'].shape[:2]
        return savepath, N, nsessions
    nsessions = int(np.ceil(C.shape[1] / window))
    rawibis_windows = {}
    maxLenW = -1
    if t_locks is not None:
        rawibis_trials = {}
        maxLenT = -1
    for i in range(C.shape[0]):
        rawibis_windows[i] = {}
        for s in range(nsessions):
            slide = C[i, s*window:min(C.shape[1], (s+1) * window)]
            ibis = ibi_func(slide)
            rawibis_windows[i][s] = ibis
            maxLenW = max(len(ibis), maxLenW)
        if t_locks is not None:
            rawibis_trials[i] = {}  # TODO: Modify IBIs to handle empty trials
            for s in range(t_locks.shape[1]):
                slide = t_locks[i, s]
                ibis = ibi_func(slide)
                rawibis_trials[i][s] = ibis
                maxLenT = max(len(ibis), maxLenT)

    all_ibis_windows = np.full((C.shape[0], nsessions, maxLenW), np.nan)
    if t_locks is not None:
        all_ibis_trials = np.full((C.shape[0], nsessions, maxLenT), np.nan)
    for i in range(C.shape[0]):
        for s in range(nsessions):
            all_ibis_windows[i][s][:len(rawibis_windows[i][s])] = rawibis_windows[i][s]
        if t_locks is not None:
            for s in range(t_locks.shape[1]):
                all_ibis_trials[i][s][:len(rawibis_trials[i][s])] = rawibis_trials[i][s]
    outfile = h5py.File(savepath, 'w-')
    outfile['IBIs_window'] = all_ibis_windows
    outfile['IBIs_trial'] = all_ibis_trials
    outname = outfile.filename
    outfile.close()
    return outname, C.shape[0], nsessions


def calcium_IBI_all_sessions(folder, window=None, method=0, options=('window', 'trial')):
    # TODO: ADD OPTION TO PASS IN A LIST OF METHODS FOR COMPARING THE PLOTS!
    """Returns a metric matrix across all sessions and meta data of IBI metric
        Params:
            folder: str
                root folder path where all the processed hdf5 will be stored
            out: str
                Output path for saving the metrics in a hdf5 file
                outfile: h5py.File
                    N: number of neurons
                    s: number of sliding sessions
                    K: maximum number of IBIs extracted
                    'mean': N * s matrix, means of IBIs
                    'stds': N * s matrix, stds of IBIs
                    'CVs': N * s matrix, CVs of IBIs
                    'IBIs': N * s * K, IBIs
                All stored by animal/day/IBI_animal_day_hyperparams.hdf5
            window: None or int
                sliding window for calculating IBIs.
                if None, use 'blen' in hdf5 file instead, but inputs have to be str/h5py.File
            perc: float
                hyperparameter for partitioning algorithm, correlated with tail length of splitted calcium trace
            ptp: boolean
                True if IBI is based on peak to peak measurement, otherwise tail to tail
            IBI_dist: boolean
                generate the IBI_distribution matrix if True

        Returns:
            res_mat: dict
                IBIs_window
                IBIs_trial
                redlabel
                array_t1
                array_miss

            mats: {group: {mat_ibi, (mat_ibi_dist,) meta}}
            mat_ibi_window: np.ndarray (first 4 ~ 8.93MB)
                A * D * N * s * m matrix,
                A: number of animals
                D: number of days
                N: number of neurons
                s: number of windows,
                m: the number of metrics
            mat_ibi_trial: np.ndarray (first 4 ~ 8.93MB)
                A * D * N * s * m matrix,
                A: number of animals
                D: number of days
                N: number of neurons
                t: number of trials,
                m: the number of metrics

            meta: dictionary
                meta data of form {group: {axis: labels}}
        """
    if method == 0:
        return {m: calcium_IBI_single_session(folder, window, m) for m in (1, 2, 11, 12)}
    processed = os.path.join(folder, 'CaBMI_analysis/processed')
    out = os.path.join(folder, 'bursting/IBI')
    all_files = get_PTIT_over_days(processed)
    hyperparam = 'theta_{}_window{}'.format(decode_method_ibi(method)[1], window)
    mats = {'meta': hyperparam}
    for group in all_files:
        group_dict = all_files[group]
        maxA, maxD, maxN = len(group_dict), max([len(group_dict[a]) for a in group_dict]), 0
        temp = {}
        res_mat = {"IBIs_{}".format(o): [0, 0] for o in options} # maxW/T, maxK
        skipped = {}
        for animal in group_dict:
            for day in group_dict[animal]:
                hf = encode_to_filename(processed, animal, day)
                errorFile = False
                if not os.path.exists(hf):
                    try:
                        calcium_IBI_single_session(hf, window, method)
                    except Exception as e:
                        errorFile = True
                        if animal in skipped:
                            skipped[animal].append([day])
                        else:
                            skipped[animal] = [day]
                with h5py.File(hf, 'r') as f:
                    temp[animal][day]['redlabel'] = np.array(f['redlabel'])
                    if 'trial' in options:
                        array_t1, array_miss = np.array(f['array_t1']), np.array(f['array_miss'])
                        a_t1, a_miss = np.full_like(array_t1, False), np.full_like(array_miss, False)
                        a_t1[array_t1] = True
                        a_miss[array_miss] = True
                        temp[animal][day]['array_t1'] = a_t1
                        temp[animal][day]['array_miss'] = a_miss
                if not errorFile:
                    with h5py.File(encode_to_filename(out, animal, day, hyperparams=hyperparam), 'r') as f:
                        for i, o in enumerate(options):
                            arg = 'IBIs_{}'.format(o)
                            ibi = f[arg]
                            if i == 0:
                                maxN = max(ibi.shape[0], maxN)
                            if animal in temp:
                                temp[animal][day] = {o: np.array(ibi)}
                            else:
                                temp[animal] = {day: {o: np.array(ibi)}}
                            res_mat[arg][0] = max(ibi.shape[1], res_mat[arg][0])
                            res_mat[arg][1] = max(ibi.shape[-1], res_mat[arg][1])
        maxA, maxD = len(temp), len(temp[max(temp.keys(), key=lambda k: len(temp[k]))])
        animal_maps = {}
        for k in res_mat:
            maxS, maxK = res_mat[k][0], res_mat[k][1]
            res_mat[k] = np.full((maxA, maxD, maxN, maxS, maxK), np.nan)
        res_mat['redlabel'] = np.full((maxA, maxD, maxN), False)
        if 'trial' in options:
            res_mat['array_t1'] = np.full((maxA, maxD, maxN, res_mat['IBIs_trial'].shape[-2]), False)
            res_mat['array_miss'] = np.full((maxA, maxD, maxN, res_mat['IBIs_trial'].shape[-2]), False)
        for i, animal in enumerate(temp):
            animal_maps[i] = animal
            for j, d in enumerate(sorted([k for k in temp[animal].keys()])):
                res_mat['redlabel'][i, j,:len(temp[animal][d]['redlabel'])] = temp[animal][d]['redlabel']
                del temp[animal][d]['redlabel']
                if 'trial' in options:
                    at1 = temp[animal][d]['array_t1']
                    am1 = temp[animal][d]['array_miss']
                    res_mat['array_t1'][i, j, :, :len(at1)] = at1
                    del temp[animal][d]['array_t1']
                    del temp[animal][d]['array_miss']
                    res_mat['array_miss'][i, j, :, :len(am1)] = am1
                for o in options:
                    tIBI = temp[animal][d][o]
                    res_mat['IBIs_{}'.format(o)][i, j, :tIBI.shape[0], :tIBI.shape[1], :tIBI.shape[2]] = tIBI
                    del temp[animal][d][o]
        res_mat['animal_map'] = animal_maps
        mats[group] = res_mat
    return mats


def IBI_to_metric_window(ibi_mat, metric='cv', red=True):
    res = {}
    for group in ibi_mat:
        if group != 'meta':
            res[group] = {}
            for k in ibi_mat[group].keys():
                if k.find('IBI') != -1:
                    if red:
                        ibis = ibi_mat[group][k][ibi_mat[group]['redlabel']]
                    else:
                        ibis = ibi_mat[group][k]
                        res[group]['redlabel'] = ibi_mat[group]['redlabel']
                    ax = len(ibi_mat.shape)-1
                    m = np.nanmean(ibis, axis=ax)
                    if metric == 'cv':
                        s = np.nanstd(ibis, axis=ax)
                    elif metric == 'cv_ub':
                        nn = np.sum(~np.isnan(ibis), axis=ax)
                        s = (1 + 1 / (4 * nn)) * np.nanstd(ibis, axis=ax)
                    elif metric == 'serr_pc':
                        nn = np.sum(~np.isnan(ibis), axis=ax)
                        s = np.nanstd(ibis, axis=ax)/np.sqrt(nn)
                    else:
                        raise ValueError("wrong metric")
                    m[m==0] = 1e-16
                    res[group][k] = s / m
    res['meta'] = ibi_mat['meta'] + '_m_{}'.format(metric)
    return res


def plot_IBI_ITPT_contrast_all_sessions(metric_mats, out, bins=None, same_rank=True, eps=True):
    if not os.path.exists(out):
        os.makedirs(out)
    IT_metric = metric_mats['IT']['IBIs_window'].ravel()
    IT_metric = IT_metric[~np.isnan(IT_metric)]
    PT_metric = metric_mats['PT']['IBIs_window'].ravel()
    PT_metric = PT_metric[~np.isnan(PT_metric)]

    if same_rank:
        minSize = min(len(IT_metric), len(PT_metric))
        inds = np.arange(minSize)
        IT_metric = IT_metric[inds]
        PT_metric = PT_metric[inds]

    fig, axes = plt.subplots(nrows=1, ncols=2)
    if bins is not None:
        axes[0].hist([IT_metric, PT_metric], bins=bins, density=True)
    else:
        axes[0].hist([IT_metric, PT_metric], bins=best_nbins(IT_metric), density=True)
    axes[0].legend(['IT', 'PT'])
    axes[0].set_title('IBI Contrast IT & PT All Sessions Histogram')
    axes[0].set_xlabel('AU')
    sns.distplot(IT_metric, bins=bins, hist=False, color="dodgerblue", label="IT", ax=axes[1])
    sns.distplot(PT_metric, bins=bins, hist=False, color="deeppink", label="PT", ax=axes[1])
    axes[1].set_title("IBI Contrast IT & PT All Sessions Histogram ")
    axes[1].set_xlabel('AU')
    fname = os.path.join(out, "ITPT_all_IBI_{}".format(metric_mats['meta']))
    fig.savefig(fname+'.png')
    if eps:
        fig.savefig(fname+".eps")


def plot_IBI_ITPT_evolution_days_windows(metric_mats, out, eps=True):
    # TODO: handle redlabel more elegantly
    if not os.path.exists(out):
        os.makedirs(out)

    def get_sequence_over_days(metric_mats, group):
        metric = metric_mats[group]['IBIs_window']
        all_mean = np.empty(metric.shape[1])
        all_serr = np.empty(metric.shape[1])
        for d in range(metric.shape[1]):
            data = metric[:, d, :, :][metric_mats[group]['redlabel'][:, d, :]].ravel()
            all_mean[d] = np.nanmean(data)
            all_serr[d] = np.nanstd(data)/np.sum(~np.isnan(data))
        return all_mean, all_serr

    def get_sequence_over_windows(metric_mats, group):
        metric = metric_mats[group]['IBIs_window'][metric_mats[group]['redlabel']]

        all_mean = np.nanmean(metric, axis=0)
        all_serr = np.nanstd(metric, axis=0)/np.sum(~np.isnan(metric), axis=0)
        return all_mean, all_serr

    data = {'IT': {'day': get_sequence_over_days(metric_mats, 'IT'),
                   'window': get_sequence_over_windows(metric_mats, 'IT')},
            'PT': {'day': get_sequence_over_days(metric_mats, 'PT'),
                   'window': get_sequence_over_windows(metric_mats, 'PT')}}
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i, t in enumerate(data['IT']):
        IT_mean, IT_serr = data['IT'][t]
        PT_mean, PT_serr = data['PT'][t]
        axes[i].errorbar(np.arange(1, len(IT_mean) + 1), IT_mean, yerr=IT_serr)
        axes[i].errorbar(np.arange(1, len(PT_mean) + 1), PT_mean, yerr=PT_serr)
        axes[i].set_title('IT PT IBI over {}s'.format(t))
        axes[i].set_xlabel(t)
        axes[i].set_ylabel('AU')
    fname = os.path.join(out, "ITPT_IBI_evolution_{}".format(metric_mats['meta']))
    fig.savefig(fname + '.png')
    if eps:
        fig.savefig(fname + ".eps")




def plot_IBI_contrast_CVs_ITPTsubset(folder, ITs, PTs, window=None, perc=30, ptp=True, IBI_dist=False,
                                     eps=True):
    """
    Params:
        folder: str
            root folder
        ITs/PTs: dict {animal: [days]}
        window: None or int
                sliding window for calculating IBIs.
                if None, use 'blen' in hdf5 file instead, but inputs have to be str/h5py.File
        perc: float
            hyperparameter for partitioning algorithm, correlated with tail length of splitted calcium trace
        ptp: boolean
            True if IBI is based on peak to peak measurement, otherwise tail to tail
        IBI_dist: boolean
            generate the IBI_distribution matrix if True

    :return:
    """
    hyperparam = 'theta_perc{}{}_window{}'.format(perc, '_ptp' if ptp else "", window)
    processed = os.path.join(folder, 'CaBMI_analysis/processed')
    IBIs = os.path.join(folder, 'bursting/IBI')
    out = os.path.join(folder, 'bursting/plots')
    ITs = parse_group_dict(processed, ITs, 'IT')
    print(ITs)
    PTs = parse_group_dict(processed, PTs, 'PT')
    def get_matrix(group_dict):
        maxA, maxD, maxN, maxS, maxIBI = \
            len(group_dict), max([len(group_dict[a]) for a in group_dict]), 0, 0, 0
        temp = {}
        for animal in group_dict:
            for day in group_dict[animal]:
                with h5py.File(encode_to_filename(processed, animal, day), 'r') as f:
                    redlabel = np.copy(f['redlabel'])
                with h5py.File(encode_to_filename(IBIs, animal, day, hyperparams=hyperparam), 'r') as f:
                    CVs = f['CVs']
                    ibi_dist = f['IBIs']
                    maxN = max(CVs.shape[0], maxN)
                    maxS = max(CVs.shape[1], maxS)
                    maxIBI = max(ibi_dist.shape[-1], maxIBI)
                    if animal in temp:
                        temp[animal][day] = {k: np.copy(f[k]) for k in f.keys()}
                    else:
                        temp[animal] = {day: {k: np.copy(f[k]) for k in f.keys()}}
                    temp[animal][day]['redlabel'] = redlabel
        animal_maps = {}
        metric_mats = {k: np.full((maxA, maxD, maxN, maxS), np.nan) for k in ('CVs', 'mean', 'stds')}
        IBI_mat = np.full((maxA, maxD, maxN, maxS, maxIBI), np.nan)
        redlabels = np.full((maxA, maxD, maxN), False)
        for i, animal in enumerate(temp):
            animal_maps[i] = animal
            for j, d in enumerate(sorted([k for k in temp[animal].keys() if k != 'redlabel'])):
                # TODO: Add day Map, which could be better perfected using navigation.mat
                redlabels[i, j, :len(temp[animal][d]['redlabel'])] = temp[animal][d]['redlabel']
                del temp[animal][d]['redlabel']
                for k in metric_mats:
                    target = temp[animal][d][k]
                    metric_mats[k][i, j, :target.shape[0], :target.shape[1]] = target
                targetIBI = temp[animal][d]['IBIs']
                IBI_mat[i, j, :targetIBI.shape[0], :targetIBI.shape[1], :targetIBI.shape[2]] \
                    = targetIBI
        return metric_mats, IBI_mat, redlabels
    IT_metric, IT_IBI, IT_redlabels = get_matrix(ITs)
    PT_metric, PT_IBI, PT_redlabels = get_matrix(PTs)
    savepath = os.path.join(out, 'IBI')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    lims = {}
    for k in IT_metric:
        if k != 'IBIs':
            tmax = max(np.nanmax(IT_metric[k][IT_redlabels]), np.nanmax(PT_metric[k][PT_redlabels]))
            tmin = min(np.nanmin(IT_metric[k][IT_redlabels]), np.nanmin(PT_metric[k][PT_redlabels]))
            lims[k] = (tmin*0.9, tmax*1.1)
    tmax = max(np.nanmax(IT_IBI[IT_redlabels]), np.nanmax(PT_IBI[PT_redlabels]))
    tmin = min(np.nanmin(IT_IBI[IT_redlabels]), np.nanmin(PT_IBI[PT_redlabels]))
    lims['IBIs'] = (tmin*0.9, tmax*1.1)
    for s in range(IT_IBI.shape[-2]):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
        for i, k in enumerate(IT_metric):
            tp = IT_metric[k]
            dataIT = tp[IT_redlabels][:, s].reshape(-1)
            dataIT = dataIT[~np.isnan(dataIT)]
            r, c = i // 2, i % 2
            sns.distplot(dataIT, bins=10, kde=True, norm_hist=True, ax=axes[r][c])
            if s < PT_IBI.shape[-2]:
                dataPT = PT_metric[k][PT_redlabels][:, s].reshape(-1)
                dataPT = dataPT[~np.isnan(dataPT)]
                sns.distplot(dataPT, bins=10, kde=True, norm_hist=True,ax=axes[r][c])
            axes[r, c].legend(['IT', 'PT'])
            axes[r, c].set_xlim(lims[k])
            axes[r, c].set_title(k)
        dataIT = IT_IBI[IT_redlabels][:, s, :].reshape(-1)
        dataIT = dataIT[~np.isnan(dataIT)]
        sns.distplot(dataIT, bins=best_nbins(dataIT), kde=True, norm_hist=True, ax=axes[1][1])
        if s < PT_IBI.shape[-2]:
            dataPT = PT_IBI[PT_redlabels][:, s, :].reshape(-1)
            dataPT = dataPT[~np.isnan(dataPT)]
            sns.distplot(dataPT, bins=best_nbins(dataPT), kde=True, norm_hist=True, ax=axes[1][1])
        axes[1, 1].legend(['IT', 'PT'])
        axes[1, 1].set_title('IBI distribution')
        axes[1, 1].set_xlim(lims['IBIs'])
        imgname = "IT_PT_contrast_session_{}".format(s)
        fig.savefig(os.path.join(savepath, imgname+'.png'))
        if eps:
            fig.savefig(os.path.join(savepath, imgname + '.eps'))


def deconv_fano_contrast_single_pair(hIT, hPT, fano_opt='raw', density=True):
    nneg = True
    W = None
    step = 100
    OPT = 'IT VS PT'
    if fano_opt == 'raw':
        fano_metric = neuron_fano
    elif fano_opt == 'norm_pre':
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=True)
    else:
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=False)

    def get_datas(hfile, data_opt, expr_opt):
        redlabels = np.array(hfile['redlabel'])
        datas = np.array(hfile[data_opt])
        blen = hfile.attrs['blen']
        if nneg:
            datas = datas - np.min(datas, axis=1, keepdims=True)
        if expr_opt.find('IT') != -1:
            datas_it = datas[np.logical_and(redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(~redlabels, hfile['nerden'])]
        elif expr_opt.find('PT') != -1:
            datas_it = datas[np.logical_and(~redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(redlabels, hfile['nerden'])]
        else:
            raise RuntimeError('NOT PT OR IT')
        return {'IT': {'N': datas_it.shape[0], 'data': datas_it,
                       'base': datas_it[:, :blen], 'online': datas_it[:, blen:]},
                'PT': {'N': datas_pt.shape[0], 'data': datas_pt,
                       'base': datas_pt[:, :blen], 'online': datas_pt[:, blen:]}}

    def fano_series(all_data, W, step, out=None, label=None):
        datas, datas_base, datas_online, N = all_data['data'], all_data['base'], \
                                             all_data['online'], all_data['N']
        nfanos = np.empty(N)
        base_fanos = np.empty(N)
        online_fanos = np.empty(N)
        for j in range(N):
            fano = fano_metric(datas[j], W, step)
            fano_base = fano_metric(datas_base[j], W, step)
            fano_online = fano_metric(datas_online[j], W, step)
            print(j, fano)
            nfanos[j] = fano
            base_fanos[j] = fano_base
            online_fanos[j] = fano_online
        if out:
            out['nfanos'][label], out['base_fanos'][label], out['online_fanos'][label] = \
                nfanos, base_fanos, online_fanos
        else:
            return out

    datas_IT_expr = get_datas(hIT, 'neuron_act', 'IT')
    datas_PT_expr = get_datas(hPT, 'neuron_act', 'PT')

    def subroutine(W, step):
        vars = ['IT_expr_IT', 'IT_expr_PT', 'PT_expr_IT', 'PT_expr_PT']
        labels = ['nfanos', 'base_fanos', 'online_fanos']
        plot_datas = {'nfanos': {d: None for d in vars},
                      'base_fanos': {d: None for d in vars},
                      'online_fanos': {d: None for d in vars}}
        fano_series(datas_IT_expr['IT'], W, step, plot_datas, 'IT_expr_IT')
        fano_series(datas_IT_expr['PT'], W, step, plot_datas, 'IT_expr_PT')
        fano_series(datas_PT_expr['IT'], W, step, plot_datas, 'PT_expr_IT')
        fano_series(datas_PT_expr['PT'], W, step, plot_datas, 'PT_expr_PT')

        for v in vars:
            ax[0][0].plot(plot_datas['nfanos'][v])
        ax[0][0].legend(vars)
        ax[0][0].set_xlabel("Neuron")
        ax[0][0].set_title("Fano Factor for all neurons")
        all_stats = {l: {v: {} for v in vars} for l in labels}
        all_stats['meta'] = {'W': W if W else -1, 'T': step}
        for i, label in enumerate(labels):
            curr = i+1
            stat = [None] * 12
            r, c = curr // 2, curr % 2
            for j, v in enumerate(vars):
                fanos = plot_datas[label][v]
                # Choice of bin size: Ref: https://www.fmrib.ox.ac.uk/datasets/techrep/tr00mj2/tr00mj2/node24
                # .html
                miu, sigma, N = np.around(np.nanmean(fanos), 5), np.around(np.nanstd(fanos), 5), len(fanos)
                binsize = 3.49 * sigma * N ** (-1/3) # or 2 IQR N ^(-1/3)
                if density:
                    sns.distplot(fanos, bins=int((max(fanos)- min(fanos)) / binsize + 1), kde=True,
                                 norm_hist=True, ax=ax[r][c])
                else:
                    ax[r][c].hist(fanos, bins=int((max(fanos)- min(fanos)) / binsize + 1), density=True,
                              alpha=0.6)
                stat[j], stat[j+4], stat[j+8] = miu, sigma, N
                all_stats[label][v]['mean'] = stat[j]
                all_stats[label][v]['std'] = stat[j+4]
                all_stats[label][v]['N'] = stat[j + 8]
            ax[r][c].legend(vars)
            ax[r][c].set_title(
                "{}, Mean(ITIT, ITPT, PTIT, PTPT): {}|{}|{}|{}\nStd: {}|{}|{}|{}, N: {}|{}|{}|{}"
                               .format(label, *stat), fontsize=10)
        outpath = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/bursty_log"
        io.savemat(os.path.join(outpath, 'fano_{}_stats_{}.mat'.format(fano_opt, all_stats['meta'])),
                   all_stats)
    #plt.style.use('bmh')
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.5)
    fig.suptitle(OPT)
    subroutine(W, step)
    axcolor = 'lightgoldenrodyellow'
    axW = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
    W_slider = Slider(axW, 'Window', valmin=50, valmax=1000, valinit=W if W else -1, valstep=1)
    axstep = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    step_slider = Slider(axstep, 'step', valmin=1, valmax=1000, valinit=step, valstep=1)

    def update(val):
        W, step = int(W_slider.val), int(step_slider.val)
        if W == -1:
            W = None
        for cax in ax.ravel():
            cax.clear()
        subroutine(W, step)
        fig.canvas.draw_idle()

    step_slider.on_changed(update)
    W_slider.on_changed(update)
    plt.show()


def deconv_fano_contrast_avg_days(root, fano_opt='raw', W=None, step=100, eps=True):
    all_files = get_PTIT_over_days(root)
    print(all_files)
    nneg = True
    OPT = 'IT VS PT bursting {}'.format(fano_opt)
    if fano_opt == 'raw':
        fano_metric = neuron_fano
    elif fano_opt == 'norm_pre':
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=True)
    else:
        fano_metric = lambda *args: neuron_fano_norm(*args, pre=False)

    def get_datas(hfile, data_opt, expr_opt):
        redlabels = np.array(hfile['redlabel'])
        datas = np.array(hfile[data_opt])
        blen = hfile.attrs['blen']
        if nneg:
            datas = datas - np.min(datas, axis=1, keepdims=True)
        if expr_opt.find('IT') != -1:
            datas_it = datas[np.logical_and(redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(~redlabels, hfile['nerden'])]
        elif expr_opt.find('PT') != -1:
            datas_it = datas[np.logical_and(~redlabels, hfile['nerden'])]
            datas_pt = datas[np.logical_and(redlabels, hfile['nerden'])]
        else:
            raise RuntimeError('NOT PT OR IT')
        return {'IT': {'N': datas_it.shape[0], 'data': datas_it,
                       'base': datas_it[:, :blen], 'online': datas_it[:, blen:]},
                'PT': {'N': datas_pt.shape[0], 'data': datas_pt,
                       'base': datas_pt[:, :blen], 'online': datas_pt[:, blen:]}}

    def fano_series(all_data, W, step, day=None, out=None, label=None):
        datas, datas_base, datas_online, N = all_data['data'], all_data['base'], \
                                             all_data['online'], all_data['N']
        nfanos = np.empty(N)
        base_fanos = np.empty(N)
        online_fanos = np.empty(N)
        for j in range(N):
            fano = fano_metric(datas[j], W, step)
            fano_base = fano_metric(datas_base[j], W, step)
            fano_online = fano_metric(datas_online[j], W, step)
            print(j, fano)
            nfanos[j] = fano
            base_fanos[j] = fano_base
            online_fanos[j] = fano_online
        if out:
            if day:
                if out['nfanos'][day][label] is None:
                    out['nfanos'][day][label], out['base_fanos'][day][label], \
                    out['online_fanos'][day][label] = nfanos, base_fanos, online_fanos
                else:
                    out['nfanos'][day][label] = np.concatenate((out['nfanos'][day][label], nfanos))
                    out['base_fanos'][day][label] = np.concatenate((out['base_fanos'][day][label], base_fanos))
                    out['online_fanos'][day][label] = np.concatenate((out['online_fanos'][day][label], online_fanos))
            else:
                if out['nfanos'][label] is None:
                    out['nfanos'][label], out['base_fanos'][label], out['online_fanos'][label] = \
                        nfanos, base_fanos, online_fanos
                else:
                    out['nfanos'][label] = np.concatenate((out['nfanos'][label], nfanos))
                    out['base_fanos'][label] = np.concatenate((out['base_fanos'][label], base_fanos))
                    out['online_fanos'][label] = np.concatenate((out['online_fanos'][label], online_fanos))
        else:
            return nfanos, base_fanos, online_fanos

    vars = ['IT_expr_IT', 'IT_expr_PT', 'PT_expr_IT', 'PT_expr_PT']
    labels = ['nfanos', 'base_fanos', 'online_fanos']
    day_range = range(1, max(len(all_files['IT']), len(all_files['PT'])))
    plot_datas = {label: {i: {d: None for d in vars} for i in day_range} for label in labels}

    for group in 'IT', 'PT':
        for day in all_files[group]:
            for expr in all_files[group][day]:
                animal, day = decode_from_filename(expr)
                hfile = h5py.File(os.path.join(root, animal, day, expr), 'r')
                for celltype in 'IT', 'PT':
                    print(group, day, celltype)
                    data_expr = get_datas(hfile, 'neuron_act', group)
                    var = '{}_expr_{}'.format(group, celltype)
                    fano_series(data_expr[celltype], W, step, day, plot_datas, var)

    plt.style.use('bmh')
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.5)
    fig.suptitle(OPT)
    all_stats = {l: {d: {v: {} for v in vars} for d in day_range} for l in labels}
    all_stats['meta'] = {'W': W if W else -1, 'T': step}
    outpath = "/home/user/bursting/plots/ITPT_contrast"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for day in day_range:
        for v in vars:
            ax[0][0].plot(plot_datas['nfanos'][day][v])
        ax[0][0].legend(vars)
        ax[0][0].set_xlabel("Neuron")
        ax[0][0].set_title("Fano Factor for all neurons")

        for i, label in enumerate(labels):
            curr = i + 1
            stat = ['NA'] * 12
            r, c = curr // 2, curr % 2
            legs = []
            for j, v in enumerate(vars):
                fanos = plot_datas[label][day][v]
                # Choice of bin size: Ref: https://www.fmrib.ox.ac.uk/datasets/techrep/tr00mj2/tr00mj2/node24
                # .html
                if fanos is not None:
                    miu, sigma, N = np.around(np.nanmean(fanos), 5), np.around(np.nanstd(fanos), 5), len(fanos)
                    nbins = best_nbins(fanos)
                    ax[r][c].hist(fanos, bins=int((max(fanos) - min(fanos)) / nbins + 1), density=True, alpha=0.6)
                    stat[j], stat[j + 4], stat[j + 8] = miu, sigma, N
                    all_stats[label][day][v]['mean'] = stat[j]
                    all_stats[label][day][v]['std'] = stat[j + 4]
                    all_stats[label][day][v]['N'] = stat[j + 8]
                legs.append(v)

            ax[r][c].legend(legs)
            ax[r][c].set_title("{}".format(label), fontsize=10)
            fig.savefig(os.path.join(outpath, "d{}_ITPT_contrast_deconvFano_{}_{}_{}.png".format(day, fano_opt, W, step)))
            if eps:
                fig.savefig(os.path.join(outpath,"d{}_ITPT_contrast_deconvFano_{}_{}_{}.eps".format(day, fano_opt, W, step)))
            plt.close('all')
    io.savemat(os.path.join(outpath, 'fano_{}_stats_{}.mat'.format(fano_opt, all_stats['meta'])), all_stats)
    io.savemat(os.path.join(outpath, 'plot_data_fano_{}_{}.mat'.format(fano_opt, all_stats['meta'])), plot_datas)


def burstITPT_contrast_plot(file, fano_opt, W, step, eps=True):
    plot_datas = io.loadmat(file)
    OPT='ITPT_contrast'
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.5)
    fig.suptitle(OPT)
    outpath = "/home/user/bursting/plots/ITPT_contrast"
    vars = ['IT_expr_IT', 'IT_expr_PT', 'PT_expr_IT', 'PT_expr_PT']
    labels = ['nfanos', 'base_fanos', 'online_fanos']
    day_range = range(1, max(len(plot_datas['IT']), len(plot_datas['PT'])) + 1)
    for day in day_range:
        for v in vars:
            ax[0][0].plot(plot_datas['nfanos'][day][v])
        ax[0][0].legend(vars)
        ax[0][0].set_xlabel("Neuron")
        ax[0][0].set_title("Fano Factor for all neurons")

        for i, label in enumerate(labels):
            curr = i + 1
            r, c = curr // 2, curr % 2
            legs = []
            for j, v in enumerate(vars):
                fanos = plot_datas[label][day][v]
                # Choice of bin size: Ref: https://www.fmrib.ox.ac.uk/datasets/techrep/tr00mj2/tr00mj2/node24
                # .html
                if fanos:
                    miu, sigma, N = np.around(np.nanmean(fanos), 5), np.around(np.nanstd(fanos), 5), len(
                        fanos)
                    nbins = min(best_nbins(fanos), 200)
                    sns.distplot(fanos, bins=nbins, kde=True, ax=ax[r][c])
                legs.append(v)
            ax[r][c].legend(legs)
            ax[r][c].set_title("{}".format(label), fontsize=10)
            fig.savefig(
                os.path.join(outpath, "d{}_ITPT_contrast_deconvFano_{}_{}_{}.png".format(day, fano_opt, W, step)))
            if eps:
                fig.savefig(os.path.join(outpath,
                                         "d{}_ITPT_contrast_deconvFano_{}_{}_{}.eps".format(day, fano_opt,
                                                                                            W, step)))


def deconv_fano_run():
    root = "/home/user/CaBMI_analysis/processed"
    W, T = None, 100
    for opt in 'norm_pre', 'raw', 'norm_post':
        deconv_fano_contrast_avg_days(root, fano_opt=opt, W=W, step=T, eps=True)


if __name__ == '__main__':
    root = "/home/user/"
    #mat = calcium_IBI_all_sessions(root)
    plot_IBI_contrast_CVs_ITPTsubset(root, '*', '*')
