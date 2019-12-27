from caiman.source_extraction.cnmf import deconvolution
from scipy.sparse import csc_matrix
import h5py
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
import os
import pandas as pd
import tifffile


def dff_test(root, animal, day, dff_func):
    f = h5py.File(root + "processed/{}/full_{}_{}__data.hdf5".format(animal, animal, day))
    csvf = root + "raw/{}/{}/bmi_IntegrationRois_00001.csv".format(animal, day)
    plots = "/Users/albertqu/Documents/7.Research/BMI/plots/caiman_test/corrDFFdoublepass_{}_{}".format(
        animal,
        day)
    if not os.path.exists(plots):
        os.makedirs(plots)

    number_planes_total = 6
    C = np.array(f['C'])
    dff = dff_func(f)
    blen = f.attrs['blen']
    nerden = f['nerden']
    ens_neur = np.array(f['ens_neur'])
    online_data = pd.read_csv(csvf)
    units = len(ens_neur)
    online = online_data.iloc[:, 2:2 + units].values.T
    online[np.isnan(online)] = 0
    frames = online_data['frameNumber'].values // number_planes_total + blen

    tests = [deconvolution.constrained_foopsi(dff[i], p=2) for i in range(dff.shape[0])]
    ts = np.vstack([t[0] for t in tests])
    nts = ts[nerden]
    ndff = dff[nerden]
    nC = C[nerden]
    allcorrs = [np.corrcoef(ts[i], dff[i])[0, 1] for i in range(ts.shape[0])]
    allcorrs = np.array(allcorrs)
    nacorrs = allcorrs[nerden]

    dff_ens = dff[ens_neur]
    dff_ens_p = dff_ens[:, frames]
    C_ens = C[ens_neur]
    C_ens_p = C_ens[:, frames]
    ts_ens = ts[ens_neur]
    ts_ens_p = ts_ens[:, frames]


    plt.hist([allcorrs, nacorrs], density=True)
    plt.legend(['allcorrs', 'neuron_only'])
    plt.title('Correlation distribution of dff and the double(on dff) pass inferred C')
    plt.xlabel('R');plt.ylabel('Relative Freq'); plt.show()
    fname = os.path.join(plots, "doublePassDFF_dff_corr")
    plt.savefig(fname+'.png')
    plt.savefig(fname+'.eps')
    plt.close('all')
    """
    animal, day = 'IT2', '181115'
    In [80]: %time tests = [deconvolution.constrained_foopsi(dff[i], p=2) for i in range(dff.shape[0])]                              
    CPU times: user 18min 43s, sys: 4min 44s, total: 23min 28s
    Wall time: 6min 30s
    
    In [164]: dffR                                                                                                                   
    Out[164]: (array([0.95562897, 0.94901895, 0.80737658, 0.89138115]), 0.017843456512294558)
    
    In [165]: doubleCR                                                                                                               
    Out[165]: (array([0.91732025, 0.90276186, 0.4199655 , 0.89416271]), 0.021650326709414264)
    
    In [166]: CR                                                                                                                     
    Out[166]: (array([0.95621973, 0.92302353, 0.46681481, 0.98135983]), 0.02671477859889339)
    """

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 15))
    i = 0
    axes[0].plot(np.vstack([zscore(dff_ens_p[i]), zscore(online[i]), zscore(ts_ens_p[i])]).T)
    axes[0].plot(zscore(C_ens_p[i]), c='k')
    axes[0].legend(['dff', 'online raw', 'double pass C', 'C'])
    axes[0].set_title('zscore')
    axes[1].plot(zscore(C_ens_p[i]), c='k')
    axes[1].plot(np.vstack([dff_ens_p[i], online[i], ts_ens_p[i]]).T)
    axes[1].legend(['C', 'dff', 'online raw', 'double pass C'])
    axes[1].set_title('Raw No zscore')

animal, day = 'IT2', '181115'
root = "/Volumes/DATA_01/NL/layerproject/"
caiman_dff = lambda hf: np.array(hf['dff'])
#dff_test(root, animal, day, dff_func=caiman_dff)


# TODO: VISUALLY INSPECT A and C.
ORDER = 'F'
T = 200
rawf_name = "baseline_00001.tif"
hf = h5py.File(root + "processed/{}/full_{}_{}__data.hdf5".format(animal, animal, day), 'r')
rawf = os.path.join(root, "raw/{}/{}/{}".format(animal, day, rawf_name))
rf = tifffile.TiffFile(rawf)

Y = np.concatenate([p.asarray()[:, :, np.newaxis] for p in rf.pages[:T]], axis=2)  #shape=(256, 256, T)
B = np.array(hf['base_im']).reshape((-1, 4))  #shape=(65536, 4)
Yr = Y.reshape((-1, T), order=ORDER)

data = hf['Nsparse']['data']
indices = hf['Nsparse']['indices']
indptr = hf['Nsparse']['indptr']
Asparse = csc_matrix((data, indices, indptr)) #(N, P)
#Asparse = csc_matrix(np.array(data), np.array(indices), np.array(indptr))

A_all = np.sum(Asparse.toarray(), axis=0)

C = hf['C']
C_samp = C[:, :T]

CP = Asparse.T @ C_samp

R = Yr-CP


# fig, axes = plt.subplots(nrows = 1, ncols=2)
# axes[0].imshow(pp.reshape((256, 256)))
# axes[1].imshow(Y[:, 0].reshape((256, 256)))
# plt.show()

# fig, axes = plt.subplots(nrows = 1, ncols=2)
# axes[0].imshow(pp.reshape((256, 256)))
# axes[1].imshow(Y[:, 0].reshape((256, 256)))

# fig, axes = plt.subplots(nrows = 1, ncols=2)
# axes[0].imshow(pp.reshape((256, 256)))
# axes[1].imshow(Y[:, 0].reshape((256, 256)))

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto

def caiman_main(fr, fnames, z=0, dend=False):
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

    if len(dims) <= 2:
        if len(z) == 1:
            com = np.concatenate((cm.base.rois.com(cnm2.estimates.A, dims[0], dims[1]),
                                  np.zeros((cnm2.estimates.A.shape[1], 1)) + z), 1)
        elif len(z) == dims[0]:
            auxcom = cm.base.rois.com(cnm2.estimates.A, dims[0], dims[1])
            zy = np.zeros((auxcom.shape[0], 1))
            for y in np.arange(auxcom.shape[0]):
                zy[y, 0] = z[int(auxcom[y, 0])]
            com = np.concatenate((auxcom, zy), 1)
        else:
            print('WARNING: Z value was not correctly defined, only X and Y values on file, z==zeros')
            print(['length of z was: ' + str(len(z))])
            com = np.concatenate((cm.base.rois.com(cnm2.estimates.A, dims[0], dims[1]),
                                  np.zeros((cnm2.estimates.A.shape[1], 1))), 1)
    else:
        com = cm.base.rois.com(cnm2.estimates.A, dims[0], dims[1], dims[2])

    return F_dff, com, cnm2, totdes, SNR_comp[idx_components]



