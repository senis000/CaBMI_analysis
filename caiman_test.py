import os, csv, h5py
import pandas as pd
import multiprocessing as mp
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from utils_loading import encode_to_filename, parse_group_dict, get_all_animals, decode_from_filename
from pipeline import *


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
def query_nans_issue(folder, animal, day, out=None, dffnans=None):
    rawf = os.path.join(folder, 'raw')
    processedf = os.path.join(folder, 'processed')
    if dffnans is not None:
        print(animal, day, file=dffnans)
    else:
        print(animal, day)
    for i in range(4):
        with h5py.File(os.path.join(rawf, animal, day, f'bmi__{i}.hdf5'), 'r') as hf:
            if dffnans is not None:
                print(i, hf['dff'].shape[0], file=dffnans)
            else:
                print(i, hf['dff'].shape[0])

    with h5py.File(encode_to_filename(processedf, animal, day), 'r') as processed:
        nans = np.any(np.isnan(processed['dff']), axis=1)
        normal = ~nans
        nans, normal = np.where(nans)[0], np.where(normal)[0]
        if dffnans is not None:
            print('#nans:', len(nans), file=dffnans)
        else:
            print('#nans:', len(nans))
        plt.figure(figsize=(15, 15))
        plt.plot(processed['com_cm'][:, 2])
        plt.scatter(normal, np.zeros_like(normal), s=0.2)
        plt.scatter(nans, np.zeros_like(nans), s=0.2)
        if out is not None:
            plt.savefig(os.path.join(out, f'plane_depth2_nan_{animal}_{day}.png'))
        else:
            plt.show()
        plt.close()


def session_nan_test(folder, out=None):
    sessions = [('PT7', '181211'),
                ('IT5', '190129'),
                ('PT9', '181219'),
                ('PT6', '181128'),
                ('IT2', '181001'),
                ('PT6', '181126'),
                ('PT9', '181128')]
    rawf = os.path.join(folder, 'raw')
    processedf = os.path.join(folder, 'processed')
    dffnans = open(os.path.join(processedf, 'dffnans.txt'), 'w+')
    for a, d in sessions:
        query_nans_issue(folder, a, d, out=out, dffnans=dffnans)


def single_dff_nan_test():
    folder ="/media/user/Seagate Backup Plus Drive/Nuria_data/CaBMI/Layer_project/"
    animal = 'IT5'
    day = '190129'
    folder_path = folder + 'raw/' + animal + '/' + day + '/'
    folder_final = folder + 'processed/' + animal + '/' + day + '/'
    err_file = open(folder_path + "errlog.txt", 'a+')  # ERROR HANDLING
    if not os.path.exists(folder_final):
        os.makedirs(folder_final)

    finfo = folder_path + 'wmat.mat'  # file name of the mat
    matinfo = scipy.io.loadmat(finfo)
    ffull = [folder_path + matinfo['fname'][0]]  # filename to be processed
    fbase = [folder_path + matinfo['fbase'][0]]
    number_planes = 4
    number_planes_total = 6

    try:
        num_files, len_bmi = separate_planes(folder, animal, day, ffull, 'bmi', number_planes,
                                             number_planes_total)
        num_files_b, len_base = separate_planes(folder, animal, day, fbase, 'baseline', number_planes,
                                                number_planes_total)
    except Exception as e:
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit('Error in separate planes')
    

    dend=False; display_images=True
    folder_path = folder + 'raw/' + animal + '/' + day + '/separated/'
    finfo = folder + 'raw/' + animal + '/' + day + '/wmat.mat'  #file name of the mat 
    matinfo = scipy.io.loadmat(finfo)
    initialZ = int(matinfo['initialZ'][0][0])
    fr = matinfo['fr'][0][0]
    
    if dend:
        sec_var = 'Dend'
    else:
        sec_var = ''
    
    print('*************Starting with analysis*************')
    neuron_mats = []
    plane = 1
    dff_all = []
    neuron_act_all = []
    fnames = []
    for nf in np.arange(int(num_files_b)):
        fnames.append(folder_path + 'baseline' + '_plane_' + str(plane) + '_nf_' + str(nf) + '.tiff')
    print('performing plane: ' + str(plane))
    for nf in np.arange(int(num_files)):
        fnames.append(folder_path + 'bmi' + '_plane_' + str(plane) + '_nf_' + str(nf) + '.tiff')
        
    fpath = folder + 'raw/' + animal + '/' + day + '/analysis/' + str(plane) + '/'
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        
    try:
        f = h5py.File(folder + 'raw/' + animal + '/' + day + '/' + 'bmi_' + sec_var + '_' + str(plane) + '.hdf5', 'w-')
    except IOError:
        print(" OOPS!: The file already existed ease try with another file, new results will NOT be saved")
        
    zval = calculate_zvalues(folder, plane)
    print(fnames)
    z = zval

    decay_time = 0.4                    # length of a typical transient in seconds    

    # Look for the best parameters for this 2p system and never change them again :)
    # motion correction parameters
    niter_rig = 1               # number of iterations for rigid motion correction
    max_shifts = (3, 3)         # maximum allow rigid shift
    splits_rig = 10            # for parallelization split the movies in  num_splits chuncks across time
    strides = (96, 96)          # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (48, 48)         # overlap between pathes (size of patch strides+overlaps)
    splits_els = 10             # for parallelization split the movies in  num_splits chuncks across time
    upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
    max_deviation_rigid = 3     # maximum deviation allowed for patch with respect to rigid shifts
    
    # parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thresh = 0.8          # merging threshold, max correlation allowed
    rf = 25                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 10            # amount of overlap between the patches in pixels
    K = 25                       # number of components per patch
    
    if dend:
        gSig = [1, 1]               # expected half size of neurons
        init_method = 'sparse_nmf'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf =  1e-6          # sparsity penalty for dendritic data analysis through sparse NMF
    else:
        gSig = [3, 3]               # expected half size of neurons
        init_method = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf = None           # sparsity penalty for dendritic data analysis through sparse NMF
    
    # parameters for component evaluation
    min_SNR = 2.5               # signal to noise ratio for accepting a component
    rval_thr = 0.8              # space correlation threshold for accepting a component
    cnn_thr = 0.8               # threshold for CNN based classifier

    dview = None # parallel processing keeps crashing. 
    
    print('***************Starting motion correction*************')
    print('files:')
    print(fnames)
    
    
    # %% start a cluster for parallel processing
    #c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    
    #%%% MOTION CORRECTION
    # first we create a motion correction object with the parameters specified
    min_mov = cm.load(fnames[0]).min()
    # this will be subtracted from the movie to make it non-negative
    
    mc = MotionCorrect(fnames, min_mov,
                       dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
                       splits_rig=splits_rig,
                       strides=strides, overlaps=overlaps, splits_els=splits_els,
                       upsample_factor_grid=upsample_factor_grid,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=True, nonneg_movie=True)
    # note that the file is not loaded in memory
    
    #%% Run piecewise-rigid motion correction using NoRMCorre
    mc.motion_correct_pwrigid(save_movie=True)
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
                            
    totdes = [np.nansum(mc.x_shifts_els), np.nansum(mc.y_shifts_els)]
    print('***************Motion correction has ended*************')
    # maximum shift to be used for trimming against NaNs

    #%% MEMORY MAPPING
    # memory map the file in order 'C'
    fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C', border_to_0=bord_px_els)  # exclude borders
    
    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)
    
    
    # %% restart cluster to clean up memory
    #cm.stop_server(dview=dview)
    #c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        
    
    #%% RUN CNMF ON PATCHES
    print('***************Running CNMF...*************')
    
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    
    cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf,
                    only_init_patch=False, gnb=gnb, border_pix=bord_px_els)
    cnm = cnm.fit(images)
    
    
    #%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    
    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
        estimate_components_quality_auto(images, cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f,
                                         cnm.estimates.YrA, fr, decay_time, gSig, dims,
                                         dview=dview, min_SNR=min_SNR,
                                         r_values_min=rval_thr, use_cnn=False,
                                         thresh_cnn_min=cnn_thr)
    
     
    if display_images:
        plt.figure()
        plt.subplot(131)

        auxb = np.transpose(np.reshape(cnm.estimates.b[:,0], [int(np.sqrt(cnm.estimates.b.shape[0])), int(np.sqrt(cnm.estimates.b.shape[0]))]))
        plt.imshow(auxb)
        plt.title('Raw mean')
        plt.subplot(132)
        crd_good = cm.utils.visualization.plot_contours(
            cnm.estimates.A[:, idx_components], auxb, thr=.8)
        plt.title('Contour plots of accepted components')
        plt.subplot(133)
        crd_bad = cm.utils.visualization.plot_contours(
            cnm.estimates.A[:, idx_components_bad], auxb, thr=.8, vmax=0.2)
        plt.title('Contour plots of rejected components')
        plt.savefig(fpath + 'comp.png', bbox_inches="tight")
         
        plt.close('all')
    
    
    #%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    A_in, C_in, b_in, f_in = cnm.estimates.A[:, idx_components], cnm.estimates.C[idx_components], cnm.estimates.b, cnm.estimates.f
    cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                     merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                     f_in=f_in, rf=None, stride=None, gnb=gnb,
                     method_deconvolution='oasis', check_nan=True)
    
    print('***************Fit*************')
    cnm2 = cnm2.fit(images)
    F_dff = detrend_df_f(cnm2.estimates.A, cnm2.estimates.b, cnm2.estimates.C, cnm2.estimates.f, YrA=cnm2.estimates.YrA, quantileMin=8, frames_window=250)


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

