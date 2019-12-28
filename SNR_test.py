import numpy as np
import caiman as cm
import tifffile, os, h5py, time
from skimage import io
from scipy.io import loadmat
from collections.abc import Iterable
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto


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
        fnamemm = os.path.join(outpath, '{}{}_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_.mmap'
                               .format(fmm, p, d1, d2, d3, order, totlen))
        bigmem =  np.memmap(fnamemm, mode='w+', dtype=np.float32, shape=(totlen, dims[0], dims[1]), order=order)

        # for i in range(totlen):
        #     print(i)
        #     img = tif.pages[nplanes * i + p].asarray()
        #     bigmem[i, :, :] = img * decay ** i if decay != 1.0 else img
        # bigmem.flush()
        temp = np.concatenate([(tif.pages[nplanes * i + p].asarray()[np.newaxis, :, :], print(i))[0] for i in range(totlen)], axis=0)
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


    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    A_in, C_in, b_in, f_in = cnm.estimates.A[:, idx_components], cnm.estimates.C[
        idx_components], cnm.estimates.b, cnm.estimates.f
    cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                     merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                     f_in=f_in, rf=None, stride=None, gnb=gnb,
                     method_deconvolution='oasis', check_nan=True)

    print('***************Fit*************')
    cnm2 = cnm2.fit(images)

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



if __name__ == '__main__':
    root = "/media/user/Seagate Backup Plus Drive/raw/IT5/190212/"  # DATA ROOT
    tiff_path = os.path.join(root, "baseline_00001.tif")
    out = os.path.join(root, 'splits')
    if not os.path.exists(out):
    	os.makedirs(out)
    # print("start splitting")
    # # nodecay
    # fname0 = extract_planes(tiff_path, out, 0)
    # print('finish nodecay')
    # #decay
    # fname1 = extract_planes(tiff_path, out, 0, decay=0.9999)
    # print('finish decay')
    fname0 = os.path.join(out, 'plane0_nodecay.tif')
    print(fname0)
    # get frame rate
    fr = loadmat(os.path.join(root, 'wmat.mat'))['fr'].item((0, 0))

    caiman_main(fr, [fname0], os.path.join(out, 'out0_nodecay.hdf5'))

