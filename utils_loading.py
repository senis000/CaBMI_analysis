import os
import numpy as np
import json
import h5py
import re
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


def path_prefix_free(path, symbol='/'):
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol,0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol)+len(symbol):]


def file_folder_path(f, symbol='/'):
    len_sym = len(symbol)
    if f[-len_sym:] == symbol:
        return f[:f.rfind(symbol, 0, -len_sym)]
    else:
        return f[:f.rfind(symbol)]


def parse_group_dict(folder, group_dict, opt):
    if "*" in group_dict:
        group_dict = {k: '*' for k in os.listdir(folder) if k.find(opt) != -1}
    for animal in group_dict:
        if group_dict[animal] == '*':
            group_dict[animal] = [v for v in os.listdir(os.path.join(folder, animal)) if v.isnumeric()]
    return group_dict


def encode_to_filename(path, animal, day, hyperparams=None):
    dirs = path.split('/')
    k = -1
    category = None
    while True:
        curr = dirs[k]
        if curr == '':
            k -= 1
        else:
            category = curr
            break
    if category == 'processed':
        template = "full_{}_{}__data.hdf5"
    elif category == 'IBI':
        template = "IBI_{}_{}_" + hyperparams + ".hdf5"
    else:
        raise ValueError("Category Undefined")
    return os.path.join(path, animal, day, template.format(animal, day))


def decode_from_filename(filename):
    fname = path_prefix_free(filename)
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

