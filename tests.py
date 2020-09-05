# System
import h5py
import os
import tifffile
import pickle
# Data
from scipy.sparse import csc_matrix
from scipy.stats import zscore, ks_2samp, wilcoxon
from sklearn.utils.random import sample_without_replacement
import numpy as np
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
except ModuleNotFoundError:
    print("plotly not installed, certain functions might not be usable")
# Caiman
try:
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf as cnmf
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf.utilities import detrend_df_f
    from caiman.components_evaluation import estimate_components_quality_auto
    from caiman.source_extraction.cnmf import deconvolution
except ModuleNotFoundError:
    print("CaImAn not installed or environment not activated, certain functions might not be usable")
# Utils
from simulation import SpikeCalciumizer
from utils_loading import encode_to_filename, get_all_animals, get_animal_days
from preprocessing import get_roi_type
from analysis_functions import granger_select_order, statsmodel_granger_asymmetric, caiman_SNR


""" ---------------------------------------------------------
-------------------- Double Pass DFF Test--------------------
---------------------------------------------------------- """


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
    ens_neur = np.array(f['ens_neur'], dtype=np.int16)
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


def dff_test_main():
    animal, day = 'IT2', '181115'
    root = "/Volumes/DATA_01/NL/layerproject/"
    caiman_dff = lambda hf: np.array(hf['dff'])
    #dff_test(root, animal, day, dff_func=caiman_dff)


def small_Test_residual():
    animal, day = 'IT2', '181115'
    root = "/Volumes/DATA_01/NL/layerproject/"
    root = "/Volumes/DATA_01/NL/layerproject/"
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


def caiman_main_light_weight(fr, fnames, z=0, dend=False):
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


""" ---------------------------------------------------------
-------------------- Double Pass DFF Test--------------------
---------------------------------------------------------- """


def noise_free_corr(d1, d2, sn1, sn2=-1, tag="", cov_correct=0, noise_thres=0, save=None, show=True):
    # noise_thres: signal_to_noise_ratio
    if sn2 is None:
        sn2 = sn1
    if len(d1.shape) == 2:
        corrs = [noise_free_corr(d1[i], d2[i], sn1[i], sn2[i] if hasattr(sn2, '__iter__') else sn2,
                                 tag=str(i)+'_'+tag) for i in range(d1.shape[0])]
        corrs0 = [np.corrcoef(d1[i], d2[i])[0,1] for i in range(d1.shape[0])]
        m2 = np.mean(d1, axis=1) ** 2
        #m2= 0
        if noise_thres:
            valid = (np.maximum(np.var(d1, ddof=1, axis=1), sn1 ** 2) - sn1 ** 2 + m2) > (noise_thres*sn1**2)
            #print(np.sum(~valid))
            corrs = np.array(corrs)[valid]
            corrs0 = np.array(corrs0)[valid]
        else:
            corrs = np.array(corrs)
            corrs0 = np.array(corrs0)
        if show:
            fig, axes = plt.subplots(1, 1)
            corrs_nonan, corrs0_nonan = corrs[~np.isnan(corrs)], corrs0[~np.isnan(corrs0)]
            print(corrs0_nonan.shape, corrs_nonan.shape)
            sns.distplot(corrs_nonan, ax=axes, label='noise-corrected')
            sns.distplot(corrs0_nonan, ax=axes, label='raw')
            axes.set_title(f'thres: {noise_thres} Max Corr: {np.nanmax(corrs):.4f} Mean Corr: '
                           f'{np.nanmean(corrs):.4f}')
            axes.set_xlabel('Correlation')
            axes.legend()
            if save is not None:
                fname = os.path.join(save, f'reconvolve_corr_distplot_{tag}')
                fig.savefig(fname + '.png')
                fig.savefig(fname + '.eps')
        return corrs

    if np.any(np.isnan(d1)) or np.any(np.isnan(d2)):
        print(f"warning nan encountered in {tag}")
        return np.nan
    covd1d2 = np.cov(d1, d2, ddof=1)[0, 1]
    d1std = np.std(d1, ddof=1)
    d2std = np.std(d2, ddof=1)
    if sn2 == -1:
        d2Cstd = d2std
    else:
        d2Cstd = np.sqrt(d2std ** 2 - sn2 ** 2)
    covCorrection = covd1d2 - d2Cstd ** 2
    covC1, covC2 = 0, 0
    if cov_correct >= 1:
        covC1 = covCorrection

    if cov_correct >= 2:
        covC2 = 2 * covCorrection
    covd1Cd2C = covd1d2 - covC1
    d1Cstd = np.sqrt(d1std ** 2 - sn1 ** 2 - covC2)
    met = covd1Cd2C / (d1Cstd * d2Cstd)
    #print(tag, covd1d2 / (d1std * d2std), sn1, d1std, d2std, covCorrection, met)
    return met


def noise_free_corr_test(func=np.sqrt, samp=100, SN=10, **kwargs):
    allX, allY = [], []
    xs = np.linspace(0, 100, 10000)
    ys = func(xs)
    for i in range(samp):

        xsn = np.random.normal(xs, SN)
        ysn = np.random.normal(ys, SN)
        allX.append(xsn)
        allY.append(ysn)
    allX, allY = np.vstack(allX), np.vstack(allY)
    sn2=None
    if 'sn2' in kwargs:
        sn2 = kwargs['sn2']
    corrs = noise_free_corr(allX, allY, np.array([SN] * samp), sn2)
    return allX, allY, corrs, np.corrcoef(xs, ys)[0, 1]


def test_distribution(d1, d2, tag1='d1', tag2='d2', alltag='', fast_samp=None, save=None, show=True, ax=None):
    if len(d1.shape) == 2:
        ksps = [test_distribution(d1[i], d2[i], show=False) for i in range(d1.shape[0])]
        if show:
            gd1, gd2 = np.ravel(d1), np.ravel(d2)
            fig, axes = plt.subplots(1, 2)
            sns.distplot(gd1, ax=axes[0], label=tag1)
            sns.distplot(gd2, ax=axes[0], label=tag2)
            axes[0].legend()
            sns.distplot(ksps, ax=axes[1])
            axes[1].set_xlabel('KStest-pvalue')
            if save is not None:
                fname = os.path.join(save, f'kstest_distplot_{alltag}')
                fig.savefig(fname + '.png')
                fig.savefig(fname + '.eps')
        return ksps

    def fast_sampling(d):
        if len(d) <= fast_samp:
            return d
        rand_inds = sample_without_replacement(n_population=len(d), n_samples=fast_samp)
        return d[rand_inds]

    if fast_samp is not None:
        if fast_samp == 'min':
            fast_samp = min(len(d1), len(d2))
        d1 = fast_sampling(d1)
        d2 = fast_sampling(d2)

    ks_stat, p = ks_2samp(d1, d2)

    if show:
        if ax is not None:
            axes = ax
        else:
            fig, axes = plt.subplots(1, 1)
        sns.distplot(d1, ax=axes, label=tag1)
        sns.distplot(d2, ax=axes, label=tag2)
        axes.legend()
        axes.set_title(f"KS p={p:.4f}")
    return p


def deconvolve_reconvolve_test(dff, deconv_p=2, conv_p=2, optimize_g=0, s_min=0, lags=5,
                               fudge_factor=1., alpha_est=None, f_saturation=0, SN=None, show=True, tag="",
                               save=None):
    # optimize_g, s_min
    # s_min = 0, None
    c, bl, c1, g, sn, sp, lam = deconvolution.constrained_foopsi(dff, p=deconv_p, bas_nonneg=False,
                                                                 optimize_g=optimize_g, s_min=s_min,
                                                                 lags=lags, fudge_factor=fudge_factor)

    if alpha_est is None:
        alpha = 1
    else:
        alpha = alpha_est
    fluorescence_model = f"AR_{conv_p}"
    spc = SpikeCalciumizer(fmodel=fluorescence_model, fluorescence_saturation=f_saturation,
                           std_noise=sn if SN is None else SN, alpha=alpha, g=g, bl=bl)
    reconv = spc.binned_spikes_to_calcium(sp.reshape((1, len(sp))))
    reconv = reconv.ravel()
    spc.std_noise = 0
    reconv_clean = spc.binned_spikes_to_calcium(sp.reshape((1, len(sp))))
    reconv_clean = reconv_clean.ravel()

    # PLOTTING
    if show or save is not None:
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        xs = np.arange(1, len(reconv)+1)
        axes[0].plot(xs, np.vstack([dff, reconv]).T)
        axes[0].legend(['Original', 'Simulated'])
        axes[1].plot(xs, sp)
        axes[1].set_xlabel('frames')
        fig.suptitle(f"Reconvolve Neuron Test {tag}")
        if show:
            plt.show()

        if save is not None:
            fname = os.path.join(save, f'reconvolve_{tag}')
            fig.savefig(fname+'.png')
            fig.savefig(fname+'.eps')
    # include bl for better calculation
    return c, c1, sp, sn, reconv, reconv_clean


def deconvolve_reconvolve_single_session_test(processed, animal, day, randN=None, savePlot=None,
                                              showPlot=True, **kwargs):
    with h5py.File(encode_to_filename(processed, animal, day), 'r') as hfile:
        roi_type = get_roi_type(hfile, animal, day)
        e_sel = np.isin(roi_type, ['E1', 'E2', 'E'])
        ind_sel = np.isin(roi_type, ['IR', 'IG'])
        nerden = np.array(hfile['nerden'])
        dff = np.array(hfile['dff'])
        dff_e = dff[e_sel, :]
        dff_ind = dff[ind_sel, :]

    dff_all = np.vstack([dff_e, dff_ind])
    if randN is not None:
        rinds = np.arange(dff_all.shape[0])
        np.random.shuffle(rinds)
        dff_all = dff_all[rinds[:randN]]

    reconvs = np.full_like(dff_all, np.nan)
    all_sn = np.full(dff_all.shape[0], np.nan)
    recleans = np.full_like(dff_all, np.nan)
    valid_selector = np.full_like(dff_all.shape[0], 1, dtype=bool)
    for i in range(dff_all.shape[0]):
        print(i)
        try:
            c, c1, sp, sn, reconv, reconv_clean = deconvolve_reconvolve_test(dff_all[i], tag=str(i), show=False,
                                                                     save=None, **kwargs)
            reconvs[i] = reconv
            all_sn[i] = sn
            recleans[i] = reconv_clean
        except:
            valid_selector[i] = False


    TAG = f'{animal}_{day}_dff_reconvolve'
    for nthres in [0, 0.5, 1]:
        corrs_clean = noise_free_corr(dff_all[valid_selector], recleans[valid_selector], all_sn[valid_selector],
                                      noise_thres=nthres, tag=TAG, save=savePlot, show=showPlot)
        # corrs = noise_free_corr(dff_all[valid_selector], reconvs[valid_selector], all_sn[valid_selector],
        #                         None, noise_thres=nthres, tag=TAG, save=savePlot,
        #                         show=showPlot)
    ksps = test_distribution(dff_all[valid_selector], reconvs[valid_selector], tag1='dff', tag2='reconvolve',
                             alltag=TAG, save=savePlot, show=showPlot)
    return corrs_clean, ksps


def deconvolve_reconvolve(calcium, c2spike=None, spike2c=None, shuffle=False, conv_p=2, f_saturation=0,
                          **kwargs):
    """ Takes in 1D calcium data after filtering baseline, deconvolve, shuffle spikes (if shuffle equals
    True), and then reconvolve
    signal
    :param calcium:
    :param p:
    :param c2spike:
    :param kwargs:
    :return:
    """
    if c2spike is None:
        deconv_params = {"p": 2, "bas_nonneg":False, "optimize_g": 0,
                         "s_min": 0, "lags": 5, "fudge_factor": 1.}
        deconv_params.update(kwargs)
        try:
            c, bl, c1, g, sn, sp, lam = deconvolution.constrained_foopsi(calcium, **deconv_params)
        except:
            return np.full(len(calcium), np.nan)
        if shuffle:
            np.random.shuffle(sp)
        fluorescence_model = f"AR_{conv_p}"
        spc = SpikeCalciumizer(fmodel=fluorescence_model, fluorescence_saturation=f_saturation,
                               std_noise=sn, alpha=1, g=g, bl=bl)
        reconv = spc.binned_spikes_to_calcium(sp.reshape((1, len(sp))))
        reconv = reconv.ravel()
    else:
        assert spike2c is not None, "specific c2spike must have corresponding spike to calcium function"
        sp = c2spike(calcium)
        if shuffle:
            np.random.shuffle(sp)
        reconv = spike2c(sp)
    return reconv


def get_ens_to_indirect_GC(pfname, roi_type, thres=0.05):
    """Takes in pickle filename file and returns GC from ensemble neuron to indirect neuron"""
    with open(pfname, 'rb') as pfile:
        pf = pickle.load(pfile)
        R_p = pf['FC_pval_red']
        R = pf['FC_red']
        R_inds = pf['indices_red']
        R_rois = roi_type[R_inds]
        R_ens_sel = (R_rois == 'E1') | (R_rois == 'E2') | (R_rois == 'E')
        R_irs_sel = R_rois == 'IR'
        R_ens_inds = R_inds[R_ens_sel]
        R_irs_inds = R_inds[R_irs_sel]
        ens_to_IR_p = R_p[R_ens_sel, : ][:, R_irs_sel]
        ens_to_IR = R[R_ens_sel, : ][:, R_irs_sel]
        ens_to_IG_p = pf['FC_pval_ens-indirect']
        ens_to_IG = pf['FC_ens-indirect']
        if ens_to_IR.shape[0] < ens_to_IG.shape[0]:
            print(f"duplicates in ens in {[pfname]}")
            ens_to_IG_p = np.unique(ens_to_IG_p, axis=0)
            ens_to_IG = np.unique(ens_to_IG, axis=0)
        GC_inds = np.concatenate([R_ens_inds, R_irs_inds, pf['indices_indirect-ens']])
        ens_to_I = np.hstack([ens_to_IR, ens_to_IG])
        ens_to_I_p = np.hstack([ens_to_IR_p, ens_to_IG_p])
        if thres is not None:
            ens_to_I[ens_to_I_p > thres] = 0 # TODO: explore distribution without p thresholding
    # compare distribution to general GC
    # TODO: use GC_inds to obtain dff activities; don't forget ensembles
    return GC_inds, ens_to_I


def GC_distribution_check_single_session(folder, animal, day, gc_dist, fast_samp='min', show=False):
    # check one session's GC value against whole gc_distribution
    # TODO: dont forget to include SNRs for different neural sites
    utils = os.path.join(folder, 'utils')
    processed = os.path.join(folder, 'processed')
    roi_type = get_roi_type(processed, animal, day)
    pfname = encode_to_filename(utils, animal, day, hyperparams='granger')
    GC_inds, ens_to_I = get_ens_to_indirect_GC(pfname, roi_type)
    if ens_to_I.shape[0] == 0:
        return np.nan
    ksps = test_distribution(ens_to_I.ravel(), gc_dist, "{}_{}_GC".format(animal, day), "GC_all_sessions",
                      "GC_ens-indirect"+f"fastsamp_{fast_samp}" if fast_samp else "", fast_samp=fast_samp,
                             show=show)
    return ksps


def get_representative_GC_sessions(folder):
    processed = os.path.join(folder, 'processed')
    fc = os.path.join(folder, "utils/FC")
    alldist = get_general_GC_distribution(folder, resamp=None)
    results = []
    for animal in get_all_animals(processed):
        animal_path = os.path.join(processed, animal)
        for day in get_animal_days(animal_path):
            fast_samp_KSs = [GC_distribution_check_single_session(folder, animal, day, alldist, show=False)
                             for _ in range(10)]
            all_KS = GC_distribution_check_single_session(folder, animal, day, alldist,
                                                          fast_samp=None, show=False)
            results.append([animal, day, all_KS, np.nanmean(fast_samp_KSs), np.nanmax(fast_samp_KSs)])
    return pd.DataFrame(results, columns=['animal', 'day', 'alldist_KS_p', 'fastsamp_KS_p_mean',
                                   'fastsamp_KS_p_max'])

    #TODO: finish the method and think about valid neuron selections


def get_general_GC_distribution(folder, resamp=0.1, concat=True, save=True):
    """ Plots distplot for the entire ens-indirect GC distribution
    Input:
        folder: root folder
    """
    fc = os.path.join(folder, "utils/FC")
    rsampTag = f"_resamp{resamp}" if resamp else ""
    alldistf = os.path.join(fc, f"gc_ens_to_I_order2_alldist{rsampTag}.npy")
    if os.path.exists(alldistf):
        assert concat, "fast load only supports concatenated format"
        return np.load(alldistf)
    utils = os.path.join(folder, 'utils')

    processed = os.path.join(folder, 'processed')
    alles = []
    for animal in get_all_animals(processed):
        animal_path = os.path.join(processed, animal)
        for day in get_animal_days(animal_path):
            print(animal, day)
            roi_type = get_roi_type(processed, animal, day)
            pfname = encode_to_filename(utils, animal, day, hyperparams='granger')
            GC_inds, ens_to_I = get_ens_to_indirect_GC(pfname, roi_type)
            ens_to_I_rav = ens_to_I.ravel()
            if resamp is not None:
                resampsize = int(resamp * len(ens_to_I_rav))
                # TODO: maybe fix seeds?
                rand_inds = sample_without_replacement(n_population=len(ens_to_I_rav), n_samples=resampsize)
                ens_to_I_rav = ens_to_I_rav[rand_inds]
            alles.append(ens_to_I_rav)

    concats = np.concatenate(alles)
    if save:
        np.save(alldistf, concats)
    if concat:
        return concats
    else:
        return np.vstack(alles)


def ens_to_ind_GC_double_reconv_shuffle_test_single_session(folder, animal, day, test_reconv=True,
                                                            thres=0.05, snr_thres=0):
    """ Calculates granger causality from ensemble to indirect neurons
    :param ens_dff: E * T, E: number ensemble neurons
    :param ind_dff: I * T, I: number of indirect neurons
    :return:
    """
    # get ens_dff, get_ind_dff
    utils = os.path.join(folder, 'utils')
    processed = os.path.join(folder, 'processed')
    with h5py.File(encode_to_filename(processed, animal, day), 'r') as hfile:
        roi_type = get_roi_type(hfile, animal, day)
        blen = hfile.attrs['blen']
        e_sel = np.isin(roi_type, ['E1', 'E2', 'E'])
        inds = np.arange(len(roi_type))
        ir_sel = roi_type == 'IR'
        ig_sel = roi_type == 'IG'
        dff = np.array(hfile['dff'])[:, :blen]
        dff_e = dff[e_sel, :]
        dff_ir = dff[ir_sel, :]
        dff_ig = dff[ig_sel, :]
        dff_inds = np.vstack([dff_ir, dff_ig])
        GC_hf_inds = np.concatenate([inds[e_sel], inds[ir_sel], inds[ig_sel]])

    if dff_e.shape[0] > 0:
        # get GC ens_dff to ind_dff
        pfname = encode_to_filename(utils, animal, day, hyperparams='granger')
        GC_inds, ens_to_I = get_ens_to_indirect_GC(pfname, roi_type, thres=0.05)

        # TODO: take care of rows of nans
        # reconvolve shuffle dff_e and dff_ind
        dff_e_rshuffle = np.vstack([deconvolve_reconvolve(dff_e[i], shuffle=True)
                                    for i in range(dff_e.shape[0])])
        dff_ind_rshuffle = np.vstack([deconvolve_reconvolve(dff_inds[i], shuffle=True)
                                      for i in range(dff_inds.shape[0])])
        dff_all_rshuffle = np.vstack([dff_e_rshuffle, dff_ind_rshuffle])

        # calculate granger causality for the reconvolved data
        lag = granger_select_order(dff_all_rshuffle, maxlag=5, ic='bic')
        gcs_val1, p_vals1 = statsmodel_granger_asymmetric(dff_e_rshuffle,
                                                          dff_ind_rshuffle, lag, False)
        p_vals1 = p_vals1['ssr_chi2test']
        gcs_val1[p_vals1 > thres] = 0
        gcs_val1 = gcs_val1[:, :, -1]
        assert gcs_val1.shape == ens_to_I.shape
        assert np.allclose(GC_hf_inds, GC_inds)

        # reconv comparison
        if test_reconv:
            # reconvolve dff_e and dff_ind
            dff_e_reconv = np.vstack([deconvolve_reconvolve(dff_e[i]) for i in range(dff_e.shape[0])])
            dff_ind_reconv = np.vstack([deconvolve_reconvolve(dff_inds[i]) for i in range(dff_inds.shape[0])])
            dff_all_reconv = np.vstack([dff_e_reconv, dff_ind_reconv])
            lag = granger_select_order(dff_all_reconv, maxlag=5, ic='bic')
            gcs_val0, p_vals0 = statsmodel_granger_asymmetric(dff_e_reconv,
                                                              dff_ind_reconv, lag, False)
            p_vals0 = p_vals0['ssr_chi2test']
            gcs_val0[p_vals0 > thres] = 0
            gcs_val0 = gcs_val0[:, :, -1]
            assert gcs_val0.shape == ens_to_I.shape
            assert np.allclose(GC_hf_inds, GC_inds)
            snr_e = np.array([caiman_SNR(None, dff_e[i], 'fast') for i in range(len(dff_e))])
            snr_inds = np.array([caiman_SNR(None, dff_inds[i], 'fast') for i in range(len(dff_inds))])
            valid_sel_e = snr_e > snr_thres
            valid_sel_inds = snr_inds > snr_thres
            ens_to_I_valid = ens_to_I[valid_sel_e, :][:, valid_sel_inds]
            gcs_val0_valid = gcs_val0[valid_sel_e, :][:, valid_sel_inds]
            gcs_val1_valid = gcs_val1[valid_sel_e, :][:, valid_sel_inds]
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
            plt.subplots_adjust(hspace=0.4)
            visualize_gc_pairs(ens_to_I_valid, gcs_val0_valid, "GC", "reconvGC", axes[0],
                               diff_label='d(raw,reconv)', verbose=True)
            visualize_gc_pairs(gcs_val0_valid, gcs_val1_valid, "reconvGC", "rshufGC", axes[1],
                               diff_label='normalized', verbose=True)
            visualize_gc_pairs(ens_to_I_valid, gcs_val1_valid, "GC", "rshufGC", axes[2],
                               diff_label='normalized', verbose=True)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
            visualize_gc_pairs(ens_to_I, gcs_val1, "GC", "rshufGC", axes, diff_label='normalized',
                               verbose=True)
    else:
        raise RuntimeError(f"No ensemble neuron in {animal} {day}")


def visualize_gc_pairs(gcs0, gcs1, tag0, tag1, axes, diff_label='normalized', verbose=True):
    # compare the unconnected pairs
    nonan_sel = (~np.isnan(gcs0)) & (~np.isnan(gcs1))
    if not np.all(nonan_sel):
        print(f"NaNs in {tag0}|{tag1}, flatten and select noNaNs")
        gcs0, gcs1 = gcs0[nonan_sel], gcs1[nonan_sel]

    unconn_sel = gcs0 == 0
    sns.distplot(gcs1[unconn_sel], ax=axes[0], label="shuffled")
    axes[0].axvline(0, c='k', ls='--')
    axes[0].set_title(f"{tag1} (paired with {tag0}=0)")
    # compare connected pairs
    conn_sel = gcs0 > 0
    conn1 = gcs1[conn_sel]
    conn0 = gcs0[conn_sel]
    ksps = test_distribution(conn1.ravel(), conn0.ravel(), tag1=tag1, tag2=tag0,
                             alltag='', show=True, ax=axes[1])
    if verbose:
        print(tag0, tag1, ksps)
    axes[1].set_title(f"{tag0} vs {tag1} (connected pair),\nP_ks = {ksps:.4f}")
    sns.distplot(conn0 - conn1, ax=axes[2], label=diff_label)
    w, pw = wilcoxon(conn0 - conn1)
    axes[2].axvline(0, c='k', ls='--')
    axes[2].set_title(f"{diff_label} GC distribution (connected pair)\nW: {w:.3f} p: {pw:.3f}")



