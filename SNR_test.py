import numpy as np
import tifffile, os, h5py, time
from skimage import io
from scipy.io import loadmat, savemat
from scipy.sparse import csc_matrix
from collections.abc import Iterable
from utils_loading import file_folder_path, get_all_animals, decode_from_filename
try:
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf as cnmf
    from caiman.source_extraction.cnmf.estimates import Estimates
    from caiman.source_extraction.cnmf.params import CNMFParams
    from caiman.source_extraction.cnmf import online_cnmf
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf.utilities import detrend_df_f
    from caiman.components_evaluation import estimate_components_quality_auto
except ModuleNotFoundError:
    print("please activate caiman environment first")
import matplotlib.pyplot as plt


def load_A(hf):
    if 'estimates' in hf:
        A = hf['estimates']['A']
    else:
        A = hf['A']
    data = A['data']
    indices = A['indices']
    indptr = A['indptr']
    return csc_matrix((data, indices, indptr), A['shape'])


def extract_planes(tfile, outpath, use_planes, nplanes=6, decay=1.0, fmm='bigmem',
                   tifn='plane', order='F', default_planes=4, del_mmap=True):
    tif = tifffile.TiffFile(tfile)
    dims = tif.pages[0].shape
    d3 = dims[2] if len(dims) == 3 else 1
    d1, d2 = dims[0], dims[1]
    totlen = int(np.ceil(len(tif.pages) / nplanes))

    if use_planes is None:
        use_planes = range(default_planes)
    elif not isinstance(use_planes, Iterable):
        use_planes = [use_planes]

    fnames = []
    for p in use_planes:
        # fnamemm = os.path.join(outpath, '{}{}_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_.mmap'
        #                        .format(fmm, p, d1, d2, d3, order, totlen))
        # bigmem =  np.memmap(fnamemm, mode='w+', dtype=np.float32, shape=(totlen, dims[0], dims[1]), order=order)

        # for i in range(totlen):
        #     print(i)
        #     img = tif.pages[nplanes * i + p].asarray()
        #     bigmem[i, :, :] = img * decay ** i if decay != 1.0 else img
        # bigmem.flush()
        temp = np.concatenate([((tif.pages[nplanes * i + p].asarray() * decay ** i)[np.newaxis, :, :] , print(i))[0] for i in range(totlen)], axis=0)
        print(p, 'saving as tif')
        # Read from mmap, save as tifs
        tifn = os.path.join(outpath, tifn)
        fname = tifn + "{}_{}decay.tif".format(p, "" if decay != 1 else "no")
        io.imsave(fname, temp, plugin='tifffile')
        # io.imsave(fname, bigmem, plugin='tifffile')
        # Delete mmap
        # if del_mmap:
        #     os.remove(fnamemm)
        #     del bigmem
        fnames.append(fname)
    return fnames


def caiman_main(fr, fnames, out, dend=False):
    # modified from https://github.com/senis000/CaBMI_analysis
    """
    Main function to compute the caiman algorithm. For more details see github and papers
    fpath(str): Folder where to store the plots
    fr(int): framerate
    fnames(list-str): list with the names of the files to be computed together
    z(array): vector with the values of z relative to y
    dend(bool): Boleean to change parameters to look for neurons or dendrites
    display_images(bool): to display and save different plots
    returns
    F_dff(array): array with the dff of the components
    com(array): matrix with the position values of the components as given by caiman
    cnm(struct): struct with different stimates and returns from caiman"""
    logfile = open(os.path.join(file_folder_path(out), 'log.txt'), 'w+')
    # parameters
    decay_time = 0.4  # length of a typical transient in seconds

    # Look for the best parameters for this 2p system and never change them again :)
    # motion correction parameters
    niter_rig = 1  # number of iterations for rigid motion correction
    max_shifts = (3, 3)  # maximum allow rigid shift
    splits_rig = 10  # for parallelization split the movies in  num_splits chuncks across time
    strides = (96, 96)  # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (48, 48)  # overlap between pathes (size of patch strides+overlaps)
    splits_els = 10  # for parallelization split the movies in  num_splits chuncks across time
    upsample_factor_grid = 4  # upsample factor to avoid smearing when merging patches
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts

    # parameters for source extraction and deconvolution
    p = 1  # order of the autoregressive system
    gnb = 2  # number of global background components
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 10  # amount of overlap between the patches in pixels
    K = 25  # number of components per patch

    if dend:
        gSig = [1, 1]  # expected half size of neurons
        init_method = 'sparse_nmf'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf = 1e-6  # sparsity penalty for dendritic data analysis through sparse NMF
    else:
        gSig = [3, 3]  # expected half size of neurons
        init_method = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf = None  # sparsity penalty for dendritic data analysis through sparse NMF

    # parameters for component evaluation
    min_SNR = 2.5  # signal to noise ratio for accepting a component
    rval_thr = 0.8  # space correlation threshold for accepting a component
    cnn_thr = 0.8  # threshold for CNN based classifier

    dview = None  # parallel processing keeps crashing.

    print('***************Starting motion correction*************')
    print('files:')
    print(fnames)

    # %% start a cluster for parallel processing
    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # %%% MOTION CORRECTION
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

    # %% Run piecewise-rigid motion correction using NoRMCorre
    mc.motion_correct_pwrigid(save_movie=True)
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)

    totdes = [np.nansum(mc.x_shifts_els), np.nansum(mc.y_shifts_els)]
    print('***************Motion correction has ended*************')
    # maximum shift to be used for trimming against NaNs

    # %% MEMORY MAPPING
    # memory map the file in order 'C'
    fnames = mc.fname_tot_els  # name of the pw-rigidly corrected file.
    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                               border_to_0=bord_px_els)  # exclude borders
    print(fname_new)

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    logfile.write('1: '+ str(images.shape) + '\n')

    # load frames in python format (T x X x Y)

    # %% restart cluster to clean up memory
    # cm.stop_server(dview=dview)
    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # %% RUN CNMF ON PATCHES
    print('***************Running CNMF...*************')

    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)

    cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf,
                    only_init_patch=False, gnb=gnb, border_pix=bord_px_els)
    cnm = cnm.fit(images)
    logfile.write("2: " + str(images.shape) + '\n')

    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier

    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
        estimate_components_quality_auto(images, cnm.estimates.A, cnm.estimates.C, cnm.estimates.b,
                                         cnm.estimates.f,
                                         cnm.estimates.YrA, fr, decay_time, gSig, dims,
                                         dview=dview, min_SNR=min_SNR,
                                         r_values_min=rval_thr, use_cnn=False,
                                         thresh_cnn_min=cnn_thr)

    logfile.write("3: " + str(images.shape) + '\n')


    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    A_in, C_in, b_in, f_in = cnm.estimates.A[:, idx_components], cnm.estimates.C[
        idx_components], cnm.estimates.b, cnm.estimates.f
    cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                     merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                     f_in=f_in, rf=None, stride=None, gnb=gnb,
                     method_deconvolution='oasis', check_nan=True)

    print('***************Fit*************')
    cnm2 = cnm2.fit(images)

    logfile.write("4: " + str(images.shape) + '\n')
    logfile.close()

    print('***************Extractind DFFs*************')
    # %% Extract DF/F values

    # cm.stop_server(dview=dview)
    try:
        F_dff = detrend_df_f(cnm2.estimates.A, cnm2.estimates.b, cnm2.estimates.C, cnm2.estimates.f,
                             YrA=cnm2.estimates.YrA, quantileMin=8, frames_window=250)
        # F_dff = detrend_df_f(cnm.A, cnm.b, cnm.C, cnm.f, YrA=cnm.YrA, quantileMin=8, frames_window=250)
    except:
        F_dff = cnm2.estimates.C * np.nan
        print('WHAAT went wrong again?')

    print('***************stopping cluster*************')
    # %% STOP CLUSTER and clean up log files
    # cm.stop_server(dview=dview)

    # ***************************************************************************************
    # Preparing output data
    # F_dff  -> DFF values,  is a matrix [number of neurons, length recording]

    # com  --> center of mass,  is a matrix [number of neurons, 2]
    print('***************preparing output data*************')
    del fname_new
    cnm2.save(out)
    with h5py.File(out, mode='a') as fp:
        fp.create_dataset('dff', data=F_dff)
        fp.create_dataset('snr', data=SNR_comp[idx_components])


def OnACID_A_init(fr, fnames, out, hfile, epochs=2):

    # %% set up some parameters

    decay_time = .4  # approximate length of transient event in seconds
    gSig = (4, 4)  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    thresh_CNN_noisy = 0.8 #0.65  # CNN threshold for candidate components
    gnb = 2  # number of background components
    init_method = 'cnmf'  # initialization method
    min_SNR = 2.5  # signal to noise ratio for accepting a component
    rval_thr = 0.8  # space correlation threshold for accepting a component
    ds_factor = 1  # spatial downsampling factor, newImg=img/ds_factor(increases speed but may lose some fine structure)

    # K = 25  # number of components per patch
    patch_size = 32  # size of patch
    stride = 3  # amount of overlap between patches

    max_num_added = 5
    max_comp_update_shape = np.inf
    update_num_comps = False


    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))  # recompute gSig if downsampling is involved
    mot_corr = True  # flag for online motion correction
    pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = np.ceil(10./ds_factor).astype('int')  # maximum allowed shift during motion correction
    sniper_mode = False  # use a CNN to detect new neurons (o/w space correlation)
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    init_batch = 500  # number of frames for initialization (presumably from the first file)
    K = 2  # initial number of components
    show_movie = False  # show the movie as the data gets processed
    print("Frame rate: {}".format(fr))
    params_dict = {'fr': fr,
                   'fnames': fnames,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'gnb': gnb,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'normalize': True,
                   'sniper_mode': sniper_mode,
                   'K': K,
                   'use_cnn': False,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'min_num_trial': 10,
                   'show_movie': show_movie,
                   'save_online_movie': False,
                   "max_num_added": max_num_added,
                   "max_comp_update_shape": max_comp_update_shape,
                   "update_num_comps": update_num_comps,
                   "dist_shape_update": update_num_comps,
                   'init_batch': init_batch,
                   'init_method': init_method,
                   'rf': patch_size // 2,
                   'stride': stride,
                   'thresh_CNN_noisy': thresh_CNN_noisy}
    opts = CNMFParams(params_dict=params_dict)
    with h5py.File(hfile, 'r') as hf:
        ests = Estimates(A=load_A(hf))
    cnm = online_cnmf.OnACID(params=opts, estimates=ests)
    cnm.estimates = ests
    cnm.fit_online()
    cnm.save(out)

def SNR_quality_test(path, animal, day):
    hfs = [h5py.File(os.path.join(path, animal, day, "bmi__{}.hdf5".format(i)), 'r') for i in range(4)]
    hf0 = h5py.File(os.path.join(path, animal, day, "SNR_IT5_190212.hdf5"), 'r')
    counter = 0
    for i in range(4):
        print(i)
        cmsnr = np.array(hfs[i]['SNR'])
        flags = np.isinf(cmsnr) | np.isnan(cmsnr)
        if np.sum(flags) > 0:
            print('nan or inf')
        cmsnr[flags] = 1
        cmlen = len(cmsnr)
        print('corr', np.corrcoef(cmsnr, hf0['SNR'][counter:counter+cmlen])[0, 1])
        counter += cmlen


def SNR_quality_test_all(folder):
    allcorrs = {}
    for animal in get_all_animals(folder):
    # for animal in ['IT10']:
        animal_path = os.path.join(folder, animal)
        for day in os.listdir(animal_path):
            if day[-5:] == '.hdf5':
                _, d = decode_from_filename(day)
            elif not day.isnumeric():
                continue
            else:
                d = day
            target = os.path.join(folder, f"{animal}/{d}/dffSNR_{animal}_{d}.hdf5")
            if not os.path.exists(target):
                corr = np.inf
            else:
                dff = h5py.File(target, 'r')
                hf = h5py.File(os.path.join(folder, f"{animal}/{d}/full_{animal}_{d}__data.hdf5"), 'r')
                if dff['SNR_ens'].shape[0] != hf['SNR'].shape[0]:
                    corr = np.nan
                else:
                    corr = np.corrcoef(dff['SNR_ens'], hf['SNR'])[0, 1]
            if animal in allcorrs:
                allcorrs[animal][d] = corr
            else:
                allcorrs[animal] = {d: corr}
    savemat(os.path.join(folder, 'dffSNR_test.mat'), allcorrs)
    allvals = np.array([allcorrs[a][d] for a in allcorrs for d in allcorrs[a]])
    infwheres = np.where(np.isinf(allvals))[0]
    nanwheres = np.where(np.isnan(allvals))[0]
    plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(len(allvals)), allvals, label='correct')
    plt.scatter(infwheres, np.zeros_like(infwheres), label='pipeline fail')
    plt.scatter(nanwheres, np.zeros_like(nanwheres), label='dimension mismatch')
    plt.title("SNR corr dff&caiman")
    plt.xlabel("arbitrary axis (sessions)")
    plt.ylabel("corrcoef (R pearson)")
    plt.legend()
    plt.savefig(os.path.join(folder, "dffSNR_test.png"))
    plt.savefig(os.path.join(folder, "dffSNR_test.eps"))
    plt.close()


def caimanTestProcess1():
    root = "/Users/albertqu/Documents/2.Courses/CogSci127/proj/data/"
    # "/media/user/Seagate Backup Plus Drive/raw/IT5/190212/"  # DATA ROOT
    tiff_path = os.path.join(root, "baseline_00001.tif")
    out = root  # os.path.join(root, 'splits')
    if not os.path.exists(out):
        os.makedirs(out)
    # print("start splitting")
    # # nodecay
    # fname0 = extract_planes(tiff_path, out, 0)
    # print('finish nodecay')
    # #decay
    # fname1 = extract_planes(tiff_path, out, 0, decay=0.9999)
    # print('finish decay')
    fname1 = os.path.join(out, 'plane0_decay.tif')
    print(fname1)
    # get frame rate
    fr = 9.72365281#loadmat(os.path.join(root, 'wmat.mat'))['fr'].item((0, 0))
    caiman_main(fr, [fname1], os.path.join(out, 'plane0_decay.hdf5'))


def uzsh_process():
    animal, day = 'IT5', '190212'
    root = "/media/user/Seagate Backup Plus Drive/raw/{}/{}/".format(animal, day)  # DATA ROOT
    tiff_path = os.path.join(root, "baseline_00001.tif")
    out = os.path.join(root, 'splits')
    if not os.path.exists(out):
        os.makedirs(out)
    # print("start splitting")
    # # nodecay
    # fname0 = extract_planes(tiff_path, out, 0)
    # print('finish nodecay')
    # decay
    # fname1 = extract_planes(tiff_path, out, 0, decay=0.9999)
    print('finish decay')
    fname0 = os.path.join(out, 'plane0_nodecay.tif')
    fname1 = os.path.join(out, 'plane0_decay.tif')
    # print(fname0)
    # get frame rate
    fr = loadmat(os.path.join(root, 'wmat.mat'))['fr'].item((0, 0))
    caiman_main(fr, [fname0], os.path.join(out, 'test_{}_{}_plane0_nodecay.hdf5'.format(animal, day)))
    # hfile0 = "{}_{}_plane0_nodecay.hdf5".format(animal, day)
    # OnACID_A_init(fr, [fname0], os.path.join(out, 'onacid_'+hfile0),
    #               os.path.join(out, hfile0))
    # hfile1 = "{}_{}_plane0_decay.hdf5".format(animal, day)
    # OnACID_A_init(fr, [fname1], os.path.join(out, 'onacid_' + hfile1),
    #               os.path.join(out, hfile1))

if __name__ == '__main__':
    uzsh_process()

