import os
import json
import h5py
import re
import tifffile
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from utils_bursting import neuron_calcium_ibi_cwt, neuron_calcium_ipri


def get_PTIT_over_days(root, order='A'):
    """
    Params:
        root: dataroot/processed/, where all the processed hdf5 will be stored.
        navigation.mat will be stored here
    Returns:
        group * [days | maps] * [animals]
    """
    if order == 'D':
        results = {'IT': {'maps': []}, 'PT': {'maps': []}}
        for animal in os.listdir(root):
            if animal.find('IT') == -1 and animal.find('PT') == -1:
                continue
            animal_path = os.path.join(root, animal)
            group = animal[:2]
            results[group]['maps'].append(animal)
            sdays = sorted(os.listdir(animal_path))
            for i, day in enumerate(sdays):
                if not day.isnumeric():
                    continue
                j = i+1
                daypath = os.path.join(animal_path, day)
                file = None
                for p in os.listdir(daypath):
                    if p.find('full') != -1:
                        file = p
                if j in results[group]:
                    results[group][j].append(file)
                else:
                    results[group][j] = [file]
        for group in results:
            sorted_animals = sorted(results[group]['maps'])
            maps = {}
            for i, animal in enumerate(sorted_animals):
                maps[animal] = i
            results[group]['maps'] = maps
        with open(os.path.join(root, 'navigation.json'), 'w') as jf:
            json.dump(results, jf)
        with open(os.path.join(root, 'navigation.json'), 'r') as jf:
            print(json.load(jf))

    elif order == 'A':
        results = {'IT': parse_group_dict(root, "*", 'IT'),
                'PT': parse_group_dict(root, "*", 'PT')}
    else:
        raise ValueError('Invalid Order Value {}'.format(order))
    return results


def get_redlabel(folder, animal, day):
    with h5py.File(encode_to_filename(folder, animal, day), 'r') as f:
        labels = np.copy(f['redlabel'])
    return labels


def path_prefix_free(path):
    symbol = os.path.sep
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol,0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol)+len(symbol):]


def file_folder_path(f):
    symbol = os.path.sep
    len_sym = len(symbol)
    if f[-len_sym:] == symbol:
        return f[:f.rfind(symbol, 0, -len_sym)]
    else:
        return f[:f.rfind(symbol)]


def parse_group_dict(folder, group_dict, opt):
    if "*" in group_dict:
        if opt == 'all':
            group_dict = {k: '*' for k in os.listdir(folder) if k.startswith('PT') or k.startswith('IT')}
        else:
            group_dict = {k: '*' for k in os.listdir(folder) if k.find(opt) != -1}
    for animal in group_dict:
        if group_dict[animal] == '*':
            group_dict[animal] = {v for v in os.listdir(os.path.join(folder, animal)) if v.isnumeric()}
    return group_dict


def get_all_animals(folder):
    return [d for d in os.listdir(folder) if d[1] == 'T'
            and os.path.isdir(os.path.join(folder, d))]


def get_animal_days(animal_path):
    subs = os.listdir(animal_path)
    hdf = lambda s: s[:4] == 'full' and s[-5:] == '.hdf5'
    for s in subs:
        if hdf(s):
            return sorted([decode_from_filename(ss)[1] for ss in subs if hdf(ss)])
    return sorted(filter(lambda s: s.isnumeric(), subs))


def encode_to_filename(path, animal, day, hyperparams=None):
    """
    :param path: str: root in which the target data file exists
    :param animal: str
    :param day: str, day of the animal or hdf5 file name (prefix free), e.g. 'full_IT2_181002.hdf5'
    :param hyperparams: str
    :return: file name
    """
    dirs = path.split(os.path.sep)
    if day[-5:] == '.hdf5':
        return os.path.join(path, animal, day)
    category = dirs[-1] if dirs[-1] else dirs[-2]
    if category == 'processed':
        template = "full_{}_{}__data.hdf5"
    elif category == 'IBI':
        template = "IBI_{}_{}_" + hyperparams + ".hdf5"
    elif category == 'utils':
        if 'granger' in hyperparams:
            hyperparams = hyperparams[7:]
            if hyperparams == "":
                hyperparams = "_baseline_red_ens-indirect_dff_order_auto"
            hyperparams = hyperparams[1:]
            path = os.path.join(path, "FC", "statsmodel")
            template = f"{hyperparams}.p"
        else:
            template = hyperparams+'_{}_{}.hdf5'
    else:
        template = category + '_{}_{}.hdf5'
        #raise ValueError("Category Undefined")
    temp = os.path.join(path, animal, day, template.format(animal, day))
    if os.path.exists(temp):
        return temp
    else:
        f = os.path.join(path, animal, template.format(animal, day))
        if not os.path.exists(f):
            raise FileNotFoundError("File {} or {} not found".format(temp, f))
        return f


def decode_from_filename(filename):
    fname = path_prefix_free(filename)
    if fname[-5:] == '.hdf5':
        fname = fname[:-5]
    opts = fname.split('_')
    return opts[1], opts[2]


def decode_method_ibi(method):
    """ Decode Method To IBI HOF
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
        0 for generating all 4 threshold: 1std, 2std, 1mad, 2mad
        opt, thres = method // 10, method % 10
        opt: 0: std
             1: mad
        thres: number of std/mad
    Returns:
        method: HOF that takes signal and return ibi
        hyperparams: string that encodes the hyperparams
    """
    if method < 0:
        ptp = (method >= -100)
        perc = np.around((-method) % 100, 2)
        hp = 'gp_perc{}{}'.format(perc, '_ptp' if ptp else "")
        return lambda sig: neuron_calcium_ipri(sig, perc, ptp), hp
    elif method == 0:
        raise ValueError("Invalid Method Option")
    else:
        opt, thres = "mad" if method // 10 else "std", method % 10
        hp = "cwt_{}_t{}".format(opt, thres)
        return lambda sig: neuron_calcium_ibi_cwt(sig, method), hp


def change_window_IBI(ibi):
    for k in os.listdir(ibi):
        for d in os.listdir(os.path.join(ibi, k)):
            fname = os.path.join(ibi, k, d, os.listdir(os.path.join(ibi, k, d))[0])
            w = fname.find('window') + 6
            d = fname.find('.h')
            os.rename(fname, fname[:w]+'None'+fname[d:])


def find_file_regex(folder, regex):
    for f in os.listdir(folder):
        if re.match(regex, f):
            return os.path.join(folder, f)


def get_learners(typhos=None):
    """
        2: LEARNER session, 1: Undefined, 0: Nonlearner Session
        Returns:
            learners, undefined, nonlearners: each is one different category of learning session.
    """

    if typhos is None:
        learning_file = "/Volumes/DATA_01/NL/layerproject/plots/learning/allDist_1max/hpm_stats_bin_5.csv"
    else:
        import os
        learning_file = os.path.join(typhos, "NL/layerproject/plots/learning/allDist_1max/hpm_stats_bin_5.csv")
    df0 = pd.read_csv(learning_file)
    df = df0.iloc[:-1]
    NL, L = 0.4, 0.6 # Please Adjust
    df['LT'] = (df['max_pc'].astype(np.float) >=NL).astype(np.int) + (df['max_pc'].astype(np.float) >=L).astype(np.int)
    df_old = df
    df = df_old[['animal', 'day', 'LT']]
    learners = df[df['LT'] == 2]
    undefined = df[df['LT'] == 1]
    nonlearners = df[df['LT'] == 1]
    return learners, undefined, nonlearners


def load_A(hf):
    if 'estimates' in hf:
        A = hf['estimates']['A']
    else:
        A = hf['Nsparse']
    data = A['data']
    indices = A['indices']
    indptr = A['indptr']
    if 'shape' in A:
        return csc_matrix((data, indices, indptr), A['shape'])
    else:
        return csc_matrix((data, indices, indptr))


def load_all(hf):
    # A, C, b, f, dff, snr
    ests = hf['estimates']
    return load_A(hf), np.array(ests['C']), np.array(ests['b']), np.array(ests['f']), np.array(
        hf['dff']), np.array(hf['snr'])


def load_Yr(tf, T, nplanes=1, used_planes=1, ret_shape=False, ORDER='F'):
    rf = tifffile.TiffFile(tf)
    shp = rf.pages[0].shape[0]
    if nplanes == 1:
        Yr = np.concatenate([p.asarray().ravel(order=ORDER)[:, np.newaxis] for p in rf.pages[:T]], axis=1)
        if ret_shape:
            return Yr, ret_shape
        return Yr
    else:
        plane_iter = used_planes if hasattr(used_planes, '__iter__') else range(used_planes)
        Y = {i: np.concatenate(
            [p.asarray().ravel(order=ORDER)[:, np.newaxis] for p in rf.pages[i:T * nplanes:nplanes]], axis=1)
             for i in plane_iter}
        Y_all = np.sum(np.concatenate([y[:, np.newaxis, :] for y in Y.values()], axis=1), axis=1)
        if ret_shape:
            return Y, Y_all, ret_shape
        return Y, Y_all

