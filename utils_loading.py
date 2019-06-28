import os
from scipy import io


def get_PTIT_over_days(root):
    """
    Params:
        root: dataroot/processed/, where all the processed hdf5 will be stored.
        navigation.mat will be stored here
    Returns:
        group * [days | maps] * [animals]
    """
    results = {'IT': {'maps': []}, 'PT': {'maps': []}}
    for animal in os.listdir(root):
        if animal.find('IT') == -1 and animal.find('PT') == -1:
            continue
        animal_path = os.path.join(root, animal)
        group = animal[:2]
        results[group]['maps'].append(animal)
        for i, day in enumerate(os.listdir(animal_path)):
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
            maps[i] = animal
        results[group]['maps'] = maps
    io.savemat(os.path.join(root, 'navigation.mat'), results)
    return results


def path_prefix_free(path, symbol='/'):
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol,0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol)+len(symbol):]


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
