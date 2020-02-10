import os, csv, h5py
import pandas as pd
import multiprocessing as mp
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from utils_loading import encode_to_filename, parse_group_dict, get_all_animals, decode_from_filename


def dff_sanity_check_single_session(rawbase, processed, animal, day, out=None, PROBELEN=1000,
                                    number_planes_total=6, mproc=False):
    rawpath = os.path.join(rawbase, animal, day)
    end = 10
    onlinef = None
    for f in os.listdir(rawpath):
        if f.find('bmi_IntegrationRois') != -1:
            tend = int(f[-5])
            if tend < end:
                end = tend
                onlinef = f
    if onlinef is None:
        raise FileNotFoundError('bmi_IntegrationRois')

    online_data = pd.read_csv(os.path.join(rawpath, onlinef))
    hfname = encode_to_filename(processed, animal, day)
    with h5py.File(hfname, 'r') as hf:
        dff = np.array(hf['dff'])
        C = np.array(hf['C'])
        blen = hf.attrs['blen']
        ens_neur = np.array(hf['ens_neur'])



    dff[np.isnan(dff)] = 0
    dff_ens = dff[ens_neur]
    C_ens = C[ens_neur]
    units = len(ens_neur)
    N = 2 * units

    def helper(vars):
        R = np.corrcoef(vars)
        corrs_pair = np.diagonal(R, units)
        chance_corr = (np.nansum(R) / 2 - units - np.nansum(corrs_pair)) * 2 / (N ** 2 - 2 * N)
        return corrs_pair, chance_corr


    corrs_pair1, chance1 = helper(np.vstack([dff_ens, C_ens]))

    frames = online_data['frameNumber'].values // number_planes_total + blen
    online = online_data.iloc[:, 2:2 + units].values.T
    online[np.isnan(online)] = 0
    slice_stack = np.vstack([dff_ens[:, frames], online])
    corrs_pair2, chance2 = helper(slice_stack)
    b = [np.nan] * 4
    corrs_pair3, chance3 = helper(np.vstack([C[:, frames], online]))

    if out is not None:
        CAIMANONLY = False
        OFFSET = 0
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axflat = axes.ravel()
        for i, ens in enumerate(ens_neur):
            if CAIMANONLY:
                axflat[i].plot(zscore(dff[ens]))
                axflat[i].plot(zscore(C[ens]) + OFFSET)
                axflat[i].legend(['CaImAnDFF', 'C'])
                axflat[i].set_title('Ens #{}'.format(ens))
            else:
                # s = online_data.iloc[:, 2+i].values
                s = online[i]
                nmean = np.nanmean(s)
                auxonline = (s - nmean) / nmean
                onlinedff = auxonline
                onlineraw = s
                # TODO: get back to Nuria for the sample analysis plots in harddrive of the
                #  ensemble vs C plots
                axflat[i].plot(zscore(dff[ens, frames[-PROBELEN:]]))
                axflat[i].plot(zscore(C[ens, frames[-PROBELEN:]]) + OFFSET * 1)
                # axflat[i].plot(zscore(onlinedff[-PROBELEN:]) + OFFSET * 2)
                axflat[i].plot(zscore(onlineraw[-PROBELEN:]) + OFFSET * 2)
                # axflat[i].legend(['CaImAnDFF', 'C', 'greedyDFF(f0=mean)', 'online raw'])
                axflat[i].legend(['CaImAnDFF', 'C', 'online raw'])
                axflat[i].set_title('Ens #{}'.format(ens))
        fig.suptitle("CaImAn DFF Sanity Check {} {}{}".format(animal, day,
                                                              " With Offset {}".format(
                                                                  OFFSET) if OFFSET else ""))
        basename = "dff_check_{}_{}{}{}".format(animal, day,
                                                "" if CAIMANONLY else "_with_raw_online",
                                                "_offset_{}".format(OFFSET) if OFFSET else "")
        tpath = os.path.join(out, "OFFSET{}".format(OFFSET))
        if not os.path.exists(tpath):
            os.makedirs(tpath)
        outname = os.path.join(out, "OFFSET{}".format(OFFSET), basename)
        fig.savefig(outname + '.png')
        fig.savefig(outname + '.eps')
        if not mproc:
            plt.show()
    results = [animal, day, chance1] + b + [chance2] + b + [chance3] +b
    for i in range(units):
        results[i + 3] = corrs_pair1[i]
        results[i + 8] = corrs_pair2[i]
        results[i + 13] = corrs_pair3[i]
    return results


def dff_sanity_check(rawbase, processed, nproc=1, group='*', out=None, csvout=None,
                     nonstop=True, PROBELEN=1000):
    # TODO: SO FAR assume map_async does not have a callback, also assuming __main__ is not mandatory
    if nproc == 0:
        nproc = mp.cpu_count()

    opt = 'all' if group == '*' else None
    group = parse_group_dict(rawbase, group, 'all')
    animals = list(group.keys())
    if opt is None:
        opt = "_".join(animals)
    pastfiles = {}
    if csvout is not None:
        csvname = os.path.join(csvout, "corr_{}_plen{}.csv"
                                 .format(opt, PROBELEN))
        if os.path.exists(csvname):
            csvdf = pd.read_csv(csvname)
            for i in range(csvdf.shape[0]):
                a, d = csvdf.iloc[i, 0], str(csvdf.iloc[i, 1])
                if a in pastfiles:
                    pastfiles[a].add(d)
                else:
                    pastfiles[a] = {d}
            csvf = open(csvname, 'a')
            cwriter = csv.writer(csvf)
        else:
            csvf = open(csvname, 'w')
            cwriter = csv.writer(csvf)
            cwriter.writerow(['animal', 'day', 'chanceC'] + ['Cens' + str(i) for i in range(4)] + ['chanceO']
                             + ['online_ens' + str(i) for i in range(4)] + ['chanceCO']
                             + ['onlineC_ens' + str(i) for i in range(4)])

    if animals is None:
        animals = [a for a in os.listdir(processed) if (a.startswith('IT') or a.startswith('PT')) and
                   os.path.isdir(os.path.join(processed, a))]
    print(animals)
    print(pastfiles)
    try:

        # for animal in animals:
        def helper(animal):
            ds = [d for d in group[animal] if animal not in pastfiles or d not in pastfiles[animal]]
            results = []
            for day in ds: #TODO: fix this with dictionary
                try:
                    result = dff_sanity_check_single_session(rawbase, processed, animal, day, out, PROBELEN,
                                                             mproc=(nproc > 1))
                    print(animal, day, 'done')
                    results.append(result)
                except Exception as e:
                    print(e.args)
                    results.append([animal, day] + [np.nan] * 15)
            return results
        if nproc == 1:
            for animal in animals:
                results = helper(animal)
                if csvout is not None:
                    for r in results:
                        cwriter.writerow(r)
        else:
            p = mp.Pool(nproc)
            allresults = p.map_async(helper, animals).get()
            for rs in allresults:
                for r in rs:
                    cwriter.writerow(r)
        if csvout is not None:
            csvf.close()
    except (KeyboardInterrupt, FileNotFoundError) as e:
        if csvout is not None:
            csvf.close()


def caiman_dff_check(folder, out):
    if not os.path.exists(out):
        os.makedirs(out)
    allrows = None
    for animal in sorted(get_all_animals(folder)):
        animal_path = os.path.join(folder, animal)
        for day in sorted(os.listdir(animal_path)):
            if day[-5:] == '.hdf5':
                _, d = decode_from_filename(day)
            elif not day.isnumeric():
                continue
            else:
                d = day
            try:
                with h5py.File(encode_to_filename(folder, animal, d), 'r') as hf:
                    nans = np.sum(np.any(np.isnan(hf['dff']), axis=1))
            except OSError as e:
                nans = np.nan
            print(animal, d)
            if allrows is None:
                allrows = np.array([[animal, d, nans]])
            else:
                allrows = np.vstack((allrows, [animal, d, nans]))
    pdf = pd.DataFrame(allrows, columns=['animal', 'day', '#nans'])
    pdf.to_csv(os.path.join(out, 'caiman_dff_quality.csv'))


#############################################################
#################### caiman issue debug #####################
#############################################################
def query_nans_issue(fil, normal, wheres):
    plt.plot(fil['com_cm'][:, 2])
    plt.scatter(normal, np.zeros_like(normal), s=0.2)
    plt.scatter(wheres, np.zeros_like(wheres), s=0.2)
    plt.show()


if __name__ == '__main__':
    root = '/run/user/1000/gvfs/smb-share:server=typhos.local,share=data_01/NL/layerproject/'
    rawbase = os.path.join(root, 'raw')
    processed = "/home/user/CaBMI_analysis/processed/"
    csvout = '/home/user/caiman_test'
    out = os.path.join(csvout, 'dff_corr')
    if not os.path.exists(csvout):
        os.makedirs(csvout)
    if not os.path.exists(out):
        os.makedirs(out)

    # dff_sanity_check(rawbase, processed, nproc=4, group=GROUPS, out=out,
    #                  csvout=csvout)
    nproc = 4
    #group = {'IT2': ['181001'], 'IT6': ['190124'], 'PT6':['181128'], 'PT19':['190731']}
    group = '*'
    PROBELEN = 1000
    if nproc == 0:
        nproc = mp.cpu_count()

    opt = 'all' if group == '*' else None
    group = parse_group_dict(rawbase, group, 'all')
    animals = list(group.keys())
    if opt is None:
        opt = "_".join(animals)
    pastfiles = {}
    if csvout is not None:
        csvname = os.path.join(csvout, "corr_{}_plen{}.csv"
                                 .format(opt, PROBELEN))
        if os.path.exists(csvname):
            csvdf = pd.read_csv(csvname)
            for i in range(csvdf.shape[0]):
                a, d = csvdf.iloc[i, 0], str(csvdf.iloc[i, 1])
                if a in pastfiles:
                    pastfiles[a].add(d)
                else:
                    pastfiles[a] = {d}
            csvf = open(csvname, 'a')
            cwriter = csv.writer(csvf)
        else:
            csvf = open(csvname, 'w')
            cwriter = csv.writer(csvf)
            cwriter.writerow(['animal', 'day', 'chanceC'] + ['Cens' + str(i) for i in range(4)] + ['chanceO']
                             + ['online_ens' + str(i) for i in range(4)] + ['chanceCO']
                             + ['onlineC_ens' + str(i) for i in range(4)])

    if animals is None:
        animals = [a for a in os.listdir(processed) if (a.startswith('IT') or a.startswith('PT')) and
                   os.path.isdir(os.path.join(processed, a))]
    print(animals)
    print(pastfiles)
    try:

        # for animal in animals:
        def helper(animal):
            ds = [d for d in group[animal] if animal not in pastfiles or d not in pastfiles[animal]]
            results = []
            for day in ds: #TODO: fix this with dictionary
                try:
                    result = dff_sanity_check_single_session(rawbase, processed, animal, day, out, PROBELEN,
                                                             mproc=(nproc > 1))
                    print(animal, day, 'done')
                    results.append(result)
                except Exception as e:
                    print(e.args)
                    results.append([animal, day] + [np.nan] * 15)
            return results
        if nproc == 1:
            for animal in animals:
                results = helper(animal)
                if csvout is not None:
                    for r in results:
                        cwriter.writerow(r)
        else:
            p = mp.Pool(nproc)
            allresults = p.map_async(helper, animals).get()
            for rs in allresults:
                for r in rs:
                    cwriter.writerow(r)
        if csvout is not None:
            csvf.close()
    except (KeyboardInterrupt, FileNotFoundError) as e:
        if csvout is not None:
            csvf.close()

