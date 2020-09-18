from builtins import str
import logging
import numpy as np
import scipy
import os
import h5py
from scipy.sparse import spdiags
import scipy.ndimage.morphology as morph

def detrend_df_f(A, b, C, f, YrA=None, quantileMin=8, frames_window=250,
                 flag_auto=True, use_fast=False):
    """ Compute DF/F signal without using the original data.
    In general much faster than extract_DF_F

    Args:
        A: scipy.sparse.csc_matrix
            spatial components (from cnmf cnm.A)

        b: ndarray
            spatial background components

        C: ndarray
            temporal components (from cnmf cnm.C)

        f: ndarray
            temporal background components

        YrA: ndarray
            residual signals

        quantile_min: float
            quantile used to estimate the baseline (values in [0,100])

        frames_window: int
            number of frames for computing running quantile

        flag_auto: bool
            flag for determining quantile automatically

        use_fast: bool
            flag for using approximate fast percentile filtering

    Returns:
        F_df:
            the computed Calcium acitivty to the derivative of f
    """

    if C is None:
        logging.warning("There are no components for DF/F extraction!")
        return None

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)
    if 'array' not in str(type(b)):
        b = b.toarray()
    if 'array' not in str(type(C)):
        C = C.toarray()
    if 'array' not in str(type(f)):
        f = f.toarray()

    nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
    nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
    nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
    A = A * nA_inv_mat
    C = nA_mat * C
    if YrA is not None:
        YrA = nA_mat * YrA

    F = C + YrA if YrA is not None else C
    B = A.T.dot(b).dot(f)
    T = C.shape[-1]

    if flag_auto:
        ###### break down
        data_prct, val = df_percentile(F[:, :frames_window], axis=1)
        if frames_window is None or frames_window > T:
            Fd = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(F, data_prct)])
            Df = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(B, data_prct)])
            F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
        else:
            Fd = np.stack([scipy.ndimage.percentile_filter(
                f, prctileMin, (frames_window)) for f, prctileMin in
                zip(F, data_prct)])
            Df = np.stack([scipy.ndimage.percentile_filter(
                f, prctileMin, (frames_window)) for f, prctileMin in
                zip(B, data_prct)])
            F_df = (F - Fd) / (Df + Fd)
    else:
        if frames_window is None or frames_window > T:
            Fd = np.percentile(F, quantileMin, axis=1)
            Df = np.percentile(B, quantileMin, axis=1)
            F_df = (F - Fd) / (Df[:, None] + Fd[:, None])
        else:
            Fd = scipy.ndimage.percentile_filter(
                F, quantileMin, (frames_window, 1))
            Df = scipy.ndimage.percentile_filter(
                B, quantileMin, (frames_window, 1))
            F_df = (F - Fd) / (Df + Fd)
    return F_df


def novel_detrend_df_f(A, b, C, f, YrA=None, quantileMin=8, frames_window=250,
                 flag_auto=True, use_fast=False, scaleC=False):
    """ Compute DF/F signal without using the original data.
    In general much faster than extract_DF_F

    Args:
        A: scipy.sparse.csc_matrix
            spatial components (from cnmf cnm.A)

        b: ndarray
            spatial background components

        C: ndarray
            temporal components (from cnmf cnm.C)

        f: ndarray
            temporal background components

        YrA: ndarray
            residual signals

        quantile_min: float
            quantile used to estimate the baseline (values in [0,100])

        frames_window: int
            number of frames for computing running quantile

        flag_auto: bool
            flag for determining quantile automatically

        use_fast: bool
            flag for using approximate fast percentile filtering

    Returns:
        F_df:
            the computed Calcium acitivty to the derivative of f
    """
    if C is None:
        logging.warning("There are no components for DF/F extraction!")
        return None

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)
    if 'array' not in str(type(b)):
        b = b.toarray()
    if 'array' not in str(type(C)):
        C = C.toarray()
    if 'array' not in str(type(f)):
        f = f.toarray()

    nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
    nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
    nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
    A = A * nA_inv_mat
    oC = C
    C = nA_mat * C
    if YrA is not None:
        YrA = nA_mat * YrA

    F = C + YrA if YrA is not None else C
    B = A.T.dot(b).dot(f)
    T = C.shape[-1]
    # TODO: FIX C FOR ALL CONTROL CLAUSES
    if flag_auto:
        ###### break down
        data_prct, val = df_percentile(F[:, :frames_window], axis=1)
        if frames_window is None or frames_window > T:
            Fd = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(F, data_prct)])
            Df = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(B, data_prct)])
            F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
        else:
            Fd = np.stack([scipy.ndimage.percentile_filter(
                f, prctileMin, (frames_window)) for f, prctileMin in
                zip(F, data_prct)])
            Df = np.stack([scipy.ndimage.percentile_filter(
                f, prctileMin, (frames_window)) for f, prctileMin in
                zip(B, data_prct)])
            if scaleC:
                F_df = C / (Df + Fd)
            else:
                F_df = oC / (Df + Fd)
    else:
        if frames_window is None or frames_window > T:
            Fd = np.percentile(F, quantileMin, axis=1)
            Df = np.percentile(B, quantileMin, axis=1)
            F_df = (F - Fd) / (Df[:, None] + Fd[:, None])
        else:
            Fd = scipy.ndimage.percentile_filter(
                F, quantileMin, (frames_window, 1))
            Df = scipy.ndimage.percentile_filter(
                B, quantileMin, (frames_window, 1))
            F_df = (F - Fd) / (Df + Fd)
    return F_df


def df_percentile(inputData, axis = None):
    """
    Extracting the percentile of the data where the mode occurs and its value.
    Used to determine the filtering level for DF/F extraction
    """
    if axis is not None:

        def fnc(x): return df_percentile(x)
        result = np.apply_along_axis(fnc, axis, inputData)
        data_prct = result[:, 0]
        val = result[:, 1]
    else:
        # Create the function that we can use for the half-sample mode
        err = True
        while err:
            try:
                bandwidth, mesh, density, cdf = kde(inputData)
                err = False
            except:
                logging.warning("There are no components for DF/F extraction!")
                if type(inputData) is not list:
                    inputData = inputData.tolist()
                inputData += inputData

        data_prct = cdf[np.argmax(density)]*100
        val = mesh[np.argmax(density)]

    return data_prct, val

import scipy as sci
import scipy.optimize
import scipy.fftpack

def kde(data, N=None, MIN=None, MAX=None):

    # Parameters to set up the mesh on which to calculate
    N = 2**12 if N is None else int(2**sci.ceil(sci.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range/10 if MIN is None else MIN
        MAX = maximum + Range/10 if MAX is None else MAX

    # Range of the data
    R = MAX-MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = sci.histogram(data, bins=N, range=(MIN, MAX))
    DataHist = DataHist/M
    DCTData = scipy.fftpack.dct(DataHist, norm=None)

    I = [iN*iN for iN in range(1, N)]
    SqDCTData = (DCTData[1:]/2)**2

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(fixed_point, 0, guess,
                                       args=(M, I, SqDCTData))
    except ValueError:
        print('Oops!')
        return None

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData*sci.exp(-sci.arange(N)**2*sci.pi**2*t_star/2)
    # Inverse DCT to get density
    density = scipy.fftpack.idct(SmDCTData, norm=None)*N/R
    mesh = [(bins[i]+bins[i+1])/2 for i in range(N)]
    bandwidth = sci.sqrt(t_star)*R

    density = density/sci.trapz(density, mesh)
    cdf = np.cumsum(density)*(mesh[1]-mesh[0])

    return bandwidth, mesh, density, cdf

def fixed_point(t, M, I, a2):
    l=7
    I = sci.float64(I)
    M = sci.float64(M)
    a2 = sci.float64(a2)
    f = 2*sci.pi**(2*l)*sci.sum(I**l*a2*sci.exp(-I*sci.pi**2*t))
    for s in range(l, 1, -1):
        K0 = sci.prod(range(1, 2*s, 2))/sci.sqrt(2*sci.pi)
        const = (1 + (1/2)**(s + 1/2))/3
        time=(2*const*K0/M/f)**(2/(3+2*s))
        f=2*sci.pi**(2*s)*sci.sum(I**s*a2*sci.exp(-I*sci.pi**2*time))
    return t-(2*M*sci.sqrt(sci.pi)*f)**(-2/5)


def nitime_solution(animal, day, cutoff=True):
    # TODO: add support for custom freq bands
    import h5py
    import nitime
    import numpy as np
    import nitime.analysis as nta
    from utils_loading import encode_to_filename
    from nitime.viz import drawmatrix_channels
    import matplotlib.pyplot as plt
    folder = "/Volumes/DATA_01/NL/layerproject/processed/"
    hf = h5py.File(encode_to_filename(folder, animal, day), 'r')
    dff = np.array(hf['dff'])
    NEUR = 'ens'
    # Critical neuron pair vs general neuron gc
    if NEUR == 'ens':
        rois = dff[hf['ens_neur']]
    elif NEUR == 'neur':
        rois = dff[hf['nerden']]
    else:
        rois = dff[NEUR]

    rois_ts = nitime.TimeSeries(rois, sampling_interval=1 / hf.attrs['fr'])
    G = nta.GrangerAnalyzer(rois_ts)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
        caus_sim = G.simultaneous_causality[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
        caus_sim = G.simultaneous_causality
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    g3 = np.mean(caus_sim, -1)
    g4 = g1-g2
    fig03 = drawmatrix_channels(g1, ['E11', 'E12', 'E21', 'E22'], size=[10., 10.], color_anchor = 0)
    plt.colorbar()


def robust_viz(g, labels):
    from nitime.viz import drawmatrix_channels
    g[np.isnan(g)] = 0
    drawmatrix_channels(g, labels, size=[10., 10.], color_anchor=0)


def statsmodel_solution(animal, day, maxlag=5):
    import h5py
    import numpy as np
    from utils_loading import encode_to_filename
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import grangercausalitytests
    folder = "/Volumes/DATA_01/NL/layerproject/processed/"
    hf = h5py.File(encode_to_filename(folder, animal, day), 'r')
    dff = np.array(hf['dff'])
    NEUR = 'ens'
    # Critical neuron pair vs general neuron gc
    if NEUR == 'ens':
        rois = dff[hf['ens_neur']]
    elif NEUR == 'neur':
        rois = dff[hf['nerden']]
    else:
        rois = dff[NEUR]
    # rois: N * T
    gcs_val = np.zeros((rois.shape[0], rois.shape[0], maxlag))
    tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest','params_ftest']
    p_vals = {t:np.zeros((rois.shape[0], rois.shape[0], maxlag)) for t in tests}
    for i in range(rois.shape[0]):
        for j in range(rois.shape[0]):
            res = grangercausalitytests(rois[[j, i]].T, maxlag)
            for k in res:
                test, reg = res[k]
                ssrEig = reg[0].ssr
                ssrBeid = reg[1].ssr
                gcs_val[i, j, k-1] = np.log(ssrEig / ssrBeid)
                for t in tests:
                    p_vals[t][i, j, k-1] = test[t][1]
            #TODO: USE LOG stats of two ssrs
    return gcs_val, p_vals


def something():
    folder = '/Users/albertqu/Documents/7.Research/BMI/testFolder'
    from utils_loading import *
    utils = os.path.join(folder)
    utils = os.path.join(folder, 'utils')
    os.listdir(utils)
    online = os.path.join(utils, 'onlineSNR')
    online
    os.listdir(online)

    k = np.random.random((5, 5))
    k[[2, 3]]
    k
    1000 * 1000 * 12 * 4
    1000 * 1000 * 12 * 4 / 1024
    1000 * 1000 * 12 * 4 / 1024 / 1024
    k = np.random.random(100)
    kl = np.concatenate((k[-5:], k[:95]))
    from statsmodels.tsa.stattools import grangercausalitytests
    test, res = grangercausalitytests(np.vstack((k, kl)).T, 5)
    res = grangercausalitytests(np.vstack((k, kl)).T, 5)
    test, r = res[5]
    test
    r

    ssrEig = reg[0].srr
    ssrBeid = reg[1].ssr
    gcs_val[i, j] = np.log(ssrEig / ssrBeid)


reg = r

ssrEig = reg[0].srr
ssrBeid = reg[1].ssr
gcs_val[i, j] = np.log(ssrEig / ssrBeid)
ssrEig = reg[0].ssr
ssrBeid = reg[1].ssr
gcs_val[i, j] = np.log(ssrEig / ssrBeid)
ssrEig
ssrBeid
res
k = np.arange(10)
k[None]
k
k[:]
k
kl
a1 = np.random.random(100)
a2 = np.concatenate((a1[-5:], a1[:95]))
plt
import matplotlib.pyplot as plt

A = np.vstack((a1, a2))
plt.plot(A.T);
plt.legend(['a', 'b'])
plt.show()
import h5py
import nitime
import numpy as np
import nitime.analysis as nta
from utils_loading import encode_to_filename
from nitime.viz import drawmatrix_channels

rois = A
rois_ts = nitime.TimeSeries(rois, sampling_interval=1 / 5)
G = nta.GrangerAnalyzer(rois_ts, order=1)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
cutoff = False
G = nta.GrangerAnalyzer(rois_ts, order=1)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
fig03 = drawmatrix_channels(g1, ['a', 'b'], size=[10., 10.], color_anchor=0)
plt.show()
fig03 = drawmatrix_channels(g2, ['a', 'b'], size=[10., 10.], color_anchor=0)
plt.show()
fig03 = drawmatrix_channels(g1 - g2, ['a', 'b'], size=[10., 10.], color_anchor=0)
plt.show()
g1
g2


def robust_viz(g, labels):
    from nitime.viz import drawmatrix_channels
    g[np.isnan(g)] = 0
    drawmatrix_channels(g, labels, size=[10., 10.], color_anchor=0)


robust_viz(g1, ['a', 'b'])
plt.show()
plt.imshow(g1);
plt.colorbar()
plt.show()
robust_viz(g1, ['a', 'b']);
plt.colorbar();
plt.show()
g1
g0 = [g1, g2]
G = nta.GrangerAnalyzer(rois_ts)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
G = nta.GrangerAnalyzer(rois_ts)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
G = nta.GrangerAnalyzer(rois_ts)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
G = nta.GrangerAnalyzer(rois_ts, order=5)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=4)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g
g1
g2
a
a1
Ap = np.vstack([a1, a2 + np.random.random(100) * 0.01])
plt.plot(Ap);
plt.show()
plt.plot(Ap.T);
plt.show()
Ap = np.vstack([a1, a2 + np.random.random(100) * 0.1])
plt.plot(Ap.T);
plt.show()
roi_ts =
roi_ts = nitime.TimeSeries(rois, sampling_interval=0.2)
rois = Ap
roi_ts = nitime.TimeSeries(rois, sampling_interval=0.2)
G = nta.GrangerAnalyzer(rois_ts)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
G = nta.GrangerAnalyzer(rois_ts, order=5)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=1)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
g0
plt.plot(Ap.T);
plt.show()
rois = Ap
rois_ts = nitime.TimeSeries(rois, sampling_interval=1 / 5)
rois_ts[:2]
Ap[:2]
Ap[:2, :2]
G = nta.GrangerAnalyzer(rois_ts)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
G = nta.GrangerAnalyzer(rois_ts, order=1)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=5)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=4)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=3)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=3)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=2)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
g1
g2
G = nta.GrangerAnalyzer(rois_ts, order=1)
if cutoff:
    sel = np.where(G.frequencies < hf.attrs['fr'])[0]
    caus_xy = G.causality_xy[:, :, sel]
    caus_yx = G.causality_yx[:, :, sel]
else:
    caus_xy = G.causality_xy
    caus_yx = G.causality_yx
g1 = np.mean(caus_xy, -1)
g2 = np.mean(caus_yx, -1)
Gs = []
for i in range(10):
    G = nta.GrangerAnalyzer(rois_ts, order=i)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    Gs.append((g1[1, 1], g2[1, 1]))
Gs
for i in range(10):
    G = nta.GrangerAnalyzer(rois_ts, order=i)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    Gs.append((g1[0, 1], g2[0, 1]))
Gs
Gs = []
for i in range(10):
    G = nta.GrangerAnalyzer(rois_ts, order=i)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    Gs.append((g1[0, 1], g2[0, 1]))
Gs
Gs = []
for i in range(10):
    G = nta.GrangerAnalyzer(rois_ts)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
        caus_sim = G.simultaneous_causality[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
        caus_sim = G.simultaneous_causality
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    g3 = np.mean(caus_sim, -1)
    Gs.append((g1, g2, g3))
Gs = []
for i in range(10):
    G = nta.GrangerAnalyzer(rois_ts, order=i)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
        caus_sim = G.simultaneous_causality[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
        caus_sim = G.simultaneous_causality
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    g3 = np.mean(caus_sim, -1)
    Gs.append((g1, g2, g3))
Gs
Gs = []
for i in range(10):
    G = nta.GrangerAnalyzer(rois_ts, order=i)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
        caus_sim = G.simultaneous_causality[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
        caus_sim = G.simultaneous_causality
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    g3 = np.mean(caus_sim, -1)
    Gs.append((g1[0, 1], g2[0, 1], g3[0, 1]))
Gs
Gs = []
for i in range(11):
    G = nta.GrangerAnalyzer(rois_ts, order=i)
    if cutoff:
        sel = np.where(G.frequencies < hf.attrs['fr'])[0]
        caus_xy = G.causality_xy[:, :, sel]
        caus_yx = G.causality_yx[:, :, sel]
        caus_sim = G.simultaneous_causality[:, :, sel]
    else:
        caus_xy = G.causality_xy
        caus_yx = G.causality_yx
        caus_sim = G.simultaneous_causality
    g1 = np.mean(caus_xy, -1)
    g2 = np.mean(caus_yx, -1)
    g3 = np.mean(caus_sim, -1)
    Gs.append((g1[0, 1], g2[0, 1], g3[0, 1]))
Gs
maxlag = 10
rois
rois.shape
gcs_val = np.zeros((rois.shape[0], rois.shape[0], maxlag))
for i in range(rois.shape[0]):
    for j in range(rois.shape[0]):
        test, reg = grangercausalitytests(rois[[j, i]], maxlag)
        ssrEig = reg[0].ssr
        ssrBeid = reg[1].ssr
        gcs_val[i, j] = np.log(ssrEig / ssrBeid)
rois.shape
plt.plot(rois.T);
plt.show()
rois = rois.T
gcs_val = np.zeros((rois.shape[0], rois.shape[0], maxlag))
for i in range(rois.shape[0]):
    for j in range(rois.shape[0]):
        test, reg = grangercausalitytests(rois[[j, i]], maxlag)
        ssrEig = reg[0].ssr
        ssrBeid = reg[1].ssr
        gcs_val[i, j] = np.log(ssrEig / ssrBeid)
rois.shape
rois[[0, 1]].shape
rois = rois.T
rois[[0, 1]]
rois[[0, 1]].shape
rois.shape
rois[[0, 1]].T.shape
rois[[0, 1]][0, 0]
rois[[0, 1]][0]
rois[[0, 1]][0, :]
rois[[0, 1]].T[0]
rois[[1, 0]].T[0]
rois.shape
plt.plot(rois);
plt.show()
plt.plot(rois.T);
plt.show()
gcs_val = np.zeros((rois.shape[0], rois.shape[0], maxlag))
for i in range(rois.shape[0]):
    for j in range(rois.shape[0]):
        test, reg = grangercausalitytests(rois[[j, i]].T, maxlag)
        ssrEig = reg[0].ssr
        ssrBeid = reg[1].ssr
        gcs_val[i, j] = np.log(ssrEig / ssrBeid)
gcs_val = np.zeros((rois.shape[0], rois.shape[0], maxlag))
for i in range(rois.shape[0]):
    for j in range(rois.shape[0]):
        k = 0
        for test, reg in grangercausalitytests(rois[[j, i]].T, maxlag):
            ssrEig = reg[0].ssr
            ssrBeid = reg[1].ssr
            gcs_val[i, j, k] = np.log(ssrEig / ssrBeid)
            k += 1
k = grangercausalitytests(rois[[j, i]].T, maxlag)
k.shape
k
for k in range(10):
    print(k)
    k += 1
    gcs_val = np.zeros((rois.shape[0], rois.shape[0], maxlag))
    for i in range(rois.shape[0]):
        for j in range(rois.shape[0]):
            res = grangercausalitytests(rois[[j, i]].T, maxlag)
            for k in res:
                test, reg = res[i]
                ssrEig = reg[0].ssr
                ssrBeid = reg[1].ssr
                gcs_val[i, j, k] = np.log(ssrEig / ssrBeid)
    gcs_val = np.zeros((rois.shape[0], rois.shape[0], maxlag))
    for i in range(rois.shape[0]):
        for j in range(rois.shape[0]):
            res = grangercausalitytests(rois[[j, i]].T, maxlag)
            for k in res:
                test, reg = res[k]
                ssrEig = reg[0].ssr
                ssrBeid = reg[1].ssr
                gcs_val[i, j, k - 1] = np.log(ssrEig / ssrBeid)
gcs_val
gcs_val.shape
p_vals = None
Gs
gcs_val[0, 0, 0]
gcs_val[0, 1, 0]
gcs_val[1, 0, 0]
gcs_val[1, 1, 0]
gcs_val[0, 0, :]
gcs_val[1, 1, :]
Gs = np.array(Gs)
Gs.shape
tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest','params_ftest']
nGs_xy = Gs[:, 0]
nGs_yx = Gs[:, 1]
sGs_xy = np.concatenate(([0], gcs_val[0, 1, :]))
sGs_xy_tests = np.vstack([np.concatenate(([0], p_vals[t][0, 1, :])) for t in tests])
sGs_yx = np.concatenate(([0], gcs_val[1, 0, :]))
sGs_yx_tests = np.vstack([np.concatenate(([0], p_vals[t][1, 0, :])) for t in tests])

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(20, 10))
axes[0, 0].plot(np.vstack((nGs_xy, sGs_xy)).T)
axes[0, 0].legend(['nitime', 'stats'])
axes[0, 1].plot(np.vstack((nGs_yx, sGs_yx)).T)
axes[0, 0].set_title('X->Y')
axes[0, 1].set_title('Y->X')
axes[1, 0].plot(sGs_xy_tests.T)
axes[1, 1].plot(sGs_yx_tests.T)
axes[1, 0].legend(tests)


# STATIONARITY
def stationarity():
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    k = np.arange(64).reshape((2, 32))
    k
    k.reshape((2, 2, 16))
    k.reshape((2, 2, 16), order='C')
    k.reshape((2, 2, 16), order='F')
    dff.shape
    dffS = dff[:, :3000 * (dff.shape[1] // 3000)]
    dffS.shape
    dff[0, :20]
    dffWrap = dffS.reshape((-1, 3000, 27), order='C')
    dffWrap = dffS.reshape((-1, 3000, 17), order='C')
    dffWrap[0, 0, :20]
    means = np.mean(dffWrap, axis=1)
    std = np.std(dffWrap, axis=1)
    stds = np.std(dffWrap, axis=1)
    stds.shape
    cols = []
    for i in range(stds.shape[0]):
        for j in range(stds.shape[1]):
            cols.append((i, j, stds[i, j]))
    stdPDF = pd.DataFrame(np.array(cols), columns=['neuron', 'window', 'std'])
    import seaborn as sns
    sns.regplot(x="neuron", y="std", stdPDF)
    sns.regplot(x="neuron", y="std", data=stdPDF)
    sns.regplot(x="window", y="std", data=stdPDF)
    % hist
    64: 66
    cols = []
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            cols.append((i, j, means[i, j]))
    meanPDF = pd.DataFrame(np.array(cols), columns=['neuron', 'window', 'mean'])
    sns.regplot(x="window", y="mean", data=meanPDF)
    sns.regplot(x="window", y="mean", data=meanPDF)
    sns.regplot(x="window", y="std", data=stdPDF)

def network_burst():
    import h5py, os
    animal, day = 'PT7', '181129'
    folder = os.path.join("/Volumes/DATA_01/NL/layerproject", 'processed')
    processed = folder
    hf = h5py.File(encode_to_filename(processed, animal, day), 'r')
    from utils_loading import encode_to_filename
    hf = h5py.File(encode_to_filename(processed, animal, day), 'r')
    dff = np.array(hf['dff'])
    import matplotlib.pyplot as plt
    import numpy as np
    dff = np.array(hf['dff'])
    hits = np.array(hf['hits'])
    misses = np.array(hf['miss'])
    SS = 0
    MS = 1
    plt.plot(np.mean(ens, axis=0));
    plt.scatter(hits, np.full_like(hits, SS), s=MS, c='r');
    plt.scatter(misses, np.full_like(misses, SS), s=MS, c='g');
    plt.show()
    ens = dff[np.array(hf['ens_neur']).astype(np.int)]
    plt.plot(np.mean(ens, axis=0));
    plt.scatter(hits, np.full_like(hits, SS), s=MS, c='r');
    plt.scatter(misses, np.full_like(misses, SS), s=MS, c='g');
    plt.show()
    MS = -0.3
    plt.plot(np.mean(ens, axis=0));
    plt.scatter(hits, np.full_like(hits, SS), s=MS, c='r');
    plt.scatter(misses, np.full_like(misses, SS), s=MS, c='g');
    plt.show()
    MS = 1
    SS = -0.3
    plt.plot(np.mean(ens, axis=0));
    plt.scatter(hits, np.full_like(hits, SS), s=MS, c='r');
    plt.scatter(misses, np.full_like(misses, SS), s=MS, c='g');
    plt.show()
    plt.plot(np.mean(ens, axis=0));
    plt.scatter(hits, np.full_like(hits, SS), s=MS, c='r');
    plt.scatter(misses, np.full_like(misses, SS), s=MS, c='g');
    plt.show()
    sig = np.mean(ens, axis=0)

    def autocorr(x):
        result = numpy.correlate(x, x, mode='full')
        return result[result.size / 2:]

    plt.plot(autocorr(sig))
    import numpy
    def autocorr(x):
        result = numpy.correlate(x, x, mode='full')
        return result[result.size / 2:]

    plt.plot(autocorr(sig))

    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:]

    plt.plot(autocorr(sig))
    plt.show()
    plt.plot(autocorr(sig))
    plt.show()
    firsthalf = sig[:, :9000]
    sig.shape
    firsthalf = sig[:9000]
    lasthalf = sig[-9000:]
    plt.plot(autocorr(firsthalf));
    plt.plot(autocorr(lasthalf));
    plt.legend();
    plt.show()
    plt.plot(autocorr(firsthalf));
    plt.plot(autocorr(lasthalf));
    plt.legend(['first', 'second']);
    plt.show()
    plt.plot(autocorr(firsthalf));
    plt.plot(autocorr(lasthalf));
    plt.legend(['first', 'second']);
    plt.show()
    plt.plot(np.diff(autocorr(firsthalf)));
    plt.plot(np.diff(autocorr(lasthalf)));
    plt.legend(['first', 'second']);
    plt.show()
    plt.plot(np.diff(autocorr(firsthalf)) + 2);
    plt.plot(np.diff(autocorr(lasthalf)));
    plt.legend(['first', 'second']);
    plt.show()
    plt.plot(np.diff(autocorr(firsthalf)) + 2);
    plt.plot(np.diff(autocorr(lasthalf)));
    plt.legend(['first', 'second']);
    plt.show()
    plt.plot(autocorr(firsthalf));
    plt.plot(autocorr(lasthalf));
    plt.legend(['first', 'second']);
    plt.show()

def irgendwas():
    pdf = pd.DataFrame(expdata.T, columns=[0, 1, 2, 3])
    pdf = pd.DataFrame(exp_data.T, columns=[0, 1, 2, 3])
    out = coint_johansen(pdf)
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    out = coint_johansen(transform_data. - 1, 5)
    out = coint_johansen(transform_data, -1, 5)
    out = coint_johansen(pdf, -1, 5)
    out
    out.lr1
    out.cvt[:, 1]
    out.cvt
    alpha = 0.01
    cvt = out.cvt[:, int(np.round((0.1 - alpha) / alpha))]
    cvt = out.cvt[:, int(np.round((0.1 - alpha) / 0.05))]
    int(np.round((0.1 - alpha) / 0.05))
    alpha = 0.05
    int(np.round((0.1 - alpha) / 0.05))
    alpha = 0.1
    int(np.round((0.1 - alpha) / 0.05))
    alpha = 0.05
    cvt = out.cvt[:, int(np.round((0.1 - alpha) / 0.05))]
    cvt
    traces
    out.lr1
    import statsmodels.tsa.api as smt
    from statsmodels.tsa.api import VAR


mod = smt.VAR(pdf)
res = mod.fit(maxlags=5, ic='aic')
res
res.summary()
print(res.summary())
mod.lag
lag_order = results.k_ar
lag_order = res.k_ar
lag_order
res.forecast(pdf.values[-lag_order:], 5)
res.plot_forecast(10)
pred = res.forcast(pdf.values[-2 * lag_order: -lag_order], 5)
pred = res.forecast(pdf.values[-2 * lag_order: -lag_order], 5)
pred
res.forecast(pdf.values[-lag_order:], 5)
res.plot_forecast(10)
pred = res.forcast(pdf.values[-2 * lag_order: -lag_order], 5)
pred = res.forecast(pdf.values[-2 * lag_order: -lag_order], 5)
actual = pdf.values[-lag_order:]
actual
pred
fig, axes = plt.subplots(nrows=4, ncols=1)
for i in range(4):
    axes[i].plot(actual[:, i])
    axes[i].plot(pred[:, i])
    axes[i].legend(['actual', 'prediction'])
plt.show()
fig, axes = plt.subplots(nrows=4, ncols=1)
for i in range(4):
    axes[i].plot(actual[:, i])
    axes[i].plot(pred[:, i])
    axes[i].legend(['actual', 'prediction'])
pred = res.forecast(pdf.values[-2 * lag_order: -lag_order], 10)
pred.shape
pred = res.forecast(pdf.values[:4500], 4500)
pred.shape
actual = pdf.values[-4500:]
fig, axes = plt.subplots(nrows=4, ncols=1)
for i in range(4):
    axes[i].plot(actual[:, i])
    axes[i].plot(pred[:, i])
    axes[i].legend(['actual', 'prediction'])
mod
mod.select_order(15)
mod.k_ar
orderRes = mod.select_order(15)
orderRes.summary()
print(orderRes.summary())
res = mod.fit(maxlags=15, ic='aic')
res.summary()
res.summary()
res.summary
res['AIC']
orderRes = mod.select_order(5)
orderRes
orderRes.summary()
print(orderRes.summary())
res = mod.fit(maxlags=5, ic='aic')
res
res.summary
res.summary()
res.sigma_u
res.resid
res.sigma_u
res.resid.mean()
sigma = res.sigma_u
numpy.linalg.det(sigma)
np.linalg.det(sigma)
pdf[0]
mod1_eig = smt.VAR(pdf[0])
mod1_eig = smt.VAR(pdf[[0, 1, 2]])
res1_eig = mod1_eig.fit(maxlags=5, ic='aic')
res1_eig.k_ar
res1_eig.resid.mean()
mod
mod.endog_names
reses = [mod.fit(i) for i in range(1, 6)]
pdf.shape
pdf.iloc[:-30]
pdf.iloc[:-30].shape
train = pdf.iloc[:-30]
test = pdf.iloc[-30:]
mod = smt.VAR(train)
reses = [mod.fit(i) for i in range(1, 6)]
pred0 = reses[0].forecast(pdf.iloc[-31:-30], 30)
pdf.iloc[-31:-30]
pred0 = reses[0].forecast(pdf.iloc[-31:-30].values, 30)
pred1 = reses[0].forecast(pdf.iloc[:-30].values, 30)
fig, axes = plt.subplots(nrows=4, ncols=1)
for i in range(4):
    axes[i].plot(pred0[:, i])
    axes[i].plot(pred1[:, i])
    axes[i].plot(test.values[:, i])
    axes[i].legend(['pred0', 'pred1', 'actual'])
fig, axes = plt.subplots(nrows=4, ncols=1)
for i in range(4):
    axes[i].plot(pred0[:, i])
    axes[i].plot(pred1[:, i] + 1)
    axes[i].plot(test.values[:, i])
    axes[i].legend(['pred0', 'pred1', 'actual'])
fig, axes = plt.subplots(nrows=4, ncols=1)
for i in range(4):
    axes[i].plot(pred0[:, i])
    axes[i].plot(pred1[:, i])
    axes[i].plot(test.values[:, i])
    axes[i].legend(['pred0', 'pred1', 'actual'])
diffs = [np.linalg.norm(res.forecast(train.values, 30) - test.values) for res in reses]
diffs
reses = [mod.fit(i) for i in range(1, 16)]
diffs = [np.linalg.norm(res.forecast(train.values, 30) - test.values) for res in reses]
diffs.shape
diffs
plt.plot(np.arange(1, 16), diffs);
plt.plot(np.arange(1, 16), diffs);
plt.ylabel('residual (30 held out frames)');
plt.xlabel('lag')
plt.plot(np.arange(1, 16), diffs);
plt.ylabel('residual (30 held out frames)');
plt.xlabel('lag');
plt.xticks(np.arange(1, 16), np.arange(1, 16))
aics = mod.select_order(15).selected_orders['aic']
aics
aics = mod.select_order(15).ics['aic']
aics.shape
aics
len(aics)
plt.plot(np.arange(16), aics);
plt.plot(np.arange(16), mod.select_order(15).ics['bic']);
plt.ylabel('IC');
plt.xlabel('lag');
plt.xticks(np.arange(0, 16), np.arange(0, 16))
plt.plot(np.arange(16), aics);
plt.plot(np.arange(16), mod.select_order(15).ics['bic']);
plt.ylabel('IC');
plt.xlabel('lag');
plt.legend(['aic', 'bic']);
plt.xticks(np.arange(0, 16), np.arange(0, 16))



from tests import *
processed = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/processed/"
animal, day = "IT4", "181004"
hf = h5py.File(encode_to_filename(processed, animal, day), 'r')
nerden = np.array(hf['nerden'])
dff = np.array(hf['dff'])[nerden]
blen = hf.attrs['blen']
roi_type = get_roi_type(processed, animal, day)
ner_roi = roi_type[nerden]
ner_ens_sel = (ner_roi == 'E1') | (ner_roi == 'E2') | (ner_roi == 'E')
dff_e = dff[ner_ens_sel]
plt.xcorr(dff_e[3], dff[43])

from statsmodels.tsa.stattools import grangercausalitytests
res1 = grangercausalitytests(np.vstack([dff_e[3], dff[43]]).T, 4, verbose=False)

dff_e_shuffle = deconvolve_reconvolve(dff_e[3], shuffle=True)
dff_ind_shuffle = deconvolve_reconvolve(dff[43], shuffle=False)

res2 = grangercausalitytests(np.vstack([dff_e_shuffle, dff_ind_shuffle]).T, 4, verbose=False)


all_corr = np.full(dff.shape[0])

all_corr = np.zeros((dff.shape[0], dff.shape[0]));all_lags = np.zeros((dff.shape[0], dff.shape[0]))
all_corr = np.zeros((dff.shape[0], dff.shape[0]));all_lags = np.zeros((dff.shape[0], dff.shape[0]));for
all_corr = np.zeros((dff.shape[0], dff.shape[0]));all_lags = np.zeros((dff.shape[0], dff.shape[0]));
plt.plot(np.arange(-dff.shape[1], dff.shape[1]+1), np.correlate(dff[0], dff[0], mode='full'))
dff = dff[:, :blen]
plt.plot(np.arange(-dff.shape[1], dff.shape[1]+1), np.correlate(dff[0], dff[0], mode='full'))
blen
blen*2
plt.plot(np.arange(-dff.shape[1]+1, dff.shape[1]), np.correlate(dff[0], dff[0], mode='full'))
np.sqrt(np.dot(dff[0], dff[0]))
np.linalg.norm(dff[0])
plt.plot(np.arange(-dff.shape[1]+1, dff.shape[1]), np.correlate(dff[0], dff[0], mode='full')/ (np.linalg.norm(dff[0])**2))
for i in range(dff.shape[0]):
    for j in range(dff.shape[0]):
        start = -dff.shape[1] + 1
for i in range(dff.shape[0]):
    for j in range(dff.shape[0]):
        start = -dff.shape[1] + 1
        corr = np.correlate(dff[i], dff[j], mode='full')
        normed = corr / (np.linalg.norm(dff[i]) * np.linalg.norm(dff[j]))
        maxlag = np.argmax(normed)
        all_corr[i, j] = normed[maxlag]
        all_lags[i, j] = normed[maxlag+start]
all_corr[0]
plt.plot(all_corr)
plt.plot(all_corr[0])
all_lags[0][all_corr[0] > 0.2]
all_lags[0]
    utils = os.path.join(folder, 'utils')
    processed = os.path.join(folder, 'processed')
    roi_type = get_roi_type(processed, animal, day)
folder = "/Volumes/DATA_01/NL/layerproject"
    utils = os.path.join(folder, 'utils')
    processed = os.path.join(folder, 'processed')
    roi_type = get_roi_type(processed, animal, day)
hfile
hf
animal, day = 'IT4', '181004'
    utils = os.path.join(folder, 'utils')
    processed = os.path.join(folder, 'processed')
    roi_type = get_roi_type(processed, animal, day)
roi_type.shape
nerden.shape

nerden = np.array(hf['nerden'])
nerden.shape

all_corr = np.zeros((dff_e.shape[0], dff.shape[0]));all_lags = np.zeros((dff_e.shape[0], dff.shape[0]));
for i in range(dff_e.shape[0]):
    for j in range(dff.shape[0]):
        start = -dff.shape[1] + 1
        corr = np.correlate(dff_e[i], dff[j], mode='full')
        normed = corr / (np.linalg.norm(dff_e[i]) * np.linalg.norm(dff[j]))
        maxlag = np.argmax(normed)
        all_corr[i, j] = normed[maxlag]
        all_lags[i, j] = maxlag+start
from scipy.signal import correlate
for i in range(dff_e.shape[0]):
    for j in range(dff.shape[0]):
        start = -dff.shape[1] + 1
        corr = correlate(dff_e[i], dff[j], mode='full')
        normed = corr / (np.linalg.norm(dff_e[i]) * np.linalg.norm(dff[j]))
        maxlag = np.argmax(normed)
        all_corr[i, j] = normed[maxlag]
        all_lags[i, j] = maxlag+start
plt.imshow(all_corr)
sns.heatmap(all_corr)
all_lags[0][all_corr[0] > 0.3]
all_lags
all_lags = all_lags.astype(np.int)
all_lags[0][all_corr[0] > 0.3]
plt.plot(all_lags[0][all_corr[0]> 0.3], all_corr[0][all_corr[0] > 0.3])
plt.scatter(all_lags[0][all_corr[0]> 0.3], all_corr[0][all_corr[0] > 0.3])
plt.scatter(all_lags[3][all_corr[3]> 0.3], all_corr[3][all_corr[3] > 0.3])
plt.xcorr(dff_e[0], dff[0], maxlags=100)
plt.xcorr(dff_e[0], dff[0], maxlags=100)
np.arange(dff.shape[0])[all_corr[3]>0.3]
plt.xcorr(dff_e[3], dff[43])
plt.plot(dff_e[3]);plt.plot(dff[43]);plt.legend(['ens3', 'neur43'])
all_lags[3, 43]
plt.plot(dff_e[3]);plt.plot(dff[43]);plt.legend(['ens3', 'neur43'])

res = grangercausalitytests(np.vstack([dff[43], dff_e[3]]).T, 2, verbose=False)
res[0]
res[1]
                    for k in res:
                        test, reg = res[k]
                        ssrEig = reg[0].ssr
                        ssrBeid = reg[1].ssr
                        print(np.log(ssrEig / ssrBeid))
test, reg = res[1]
reg.llf
reg[0].llf
reg[1].llf
reg[1].nobs
reg[0].ssr
reg[1].ssr
reg[0].param
reg[0].params
reg[1].params
x = np.vstack([dff_e[3], dff[43]]).T
x.shape
dta = lagmat2ds(x, 1, trim='both', dropex=1)
from statsmodels.tsa.tsatools import lagmat, lagmat2ds, add_trend
dta = lagmat2ds(x, 1, trim='both', dropex=1)
dta.shape
dta[:3, :]
x[:3]
pred1 = reg[0].predict(reg[0].params, dta[:, 1])
reg[0].prams
reg[0].params
reg[0].exog.shape
reg[0].exog
            dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
from statsmodels.tools.tools import add_constant, Bunch
dtaown[:3]
            dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
            dtaown = add_constant(dta[:, 1], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
dtaown[:3]
dtajoint[:3]
res1 = grangercausalitytests(np.vstack([dff_e[3], dff[43]]).T, 2, verbose=False)
res1[1].params
test1, reg1 = res1[1]
reg1[1].params
                        test1, reg1 = res1[k]
                        ssrEig1 = reg1[0].ssr
                        ssrBeid1 = reg1[1].ssr
ssrEig1
ssrBeid1
reg1[0].params
k
                        test1, reg1 = res1[1]
                        ssrEig1 = reg1[0].ssr
                        ssrBeid1 = reg1[1].ssr
ssrEig1
ssrBeid1
reg1[0]
reg1[0].params
reg1[1].params
x = np.vstack([dff_e[3], dff[43]]).T
dta = lagmat2ds(x, 1, trim='both', dropex=1)
            dtaown = add_constant(dta[:, 1:(1 + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
dtaother = add_constant(dta[:, 2], prepend=False)
dtaother.shape
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()
from statsmodels.regression.linear_model import OLS, yule_walker
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()
res2dother = OLS(dta[:, 0], dtaother).fit()
res2dother.ssr
res2dother.params
res2down.ssr
plt.plot(dta[:, 0]);plt.plot(res2dother.predict(res2dother.params, dtaother))
res2dother.params
dta[:3]
plt.plot(dta[:, 0]);plt.plot(res2dother.params[0] * dta[:, 2])
plt.plot(dta[:, 0]);plt.plot(5 * dta[:, 2])
np.sum(np.square(dta[:, 0] - 5 * dta[:, 2]))
np.sum(np.square(dta[:, 0] - 5 * dta[:, 2] - res2dother.params[1]))
np.sum(np.square(dta[:, 0] - 1.568331 * dta[:, 2] - res2dother.params[1]))
plt.plot(dta[:, 0]);plt.plot(5 * dta[:, 2])
plt.plot(dta[:, 0]);plt.plot(dta[:, 2]**2)
plt.plot(dta[:, 0]);plt.plot(dta[:, 2]**2 * 10)
plt.plot(dta[:, 0]);plt.plot(dta[:, 2]**2 * 13)
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 13 - res2dother.params[1]))
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 13))
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 13- dta[:, 2]))
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 13- 1.56* dta[:, 2]))
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 13- 0.3* dta[:, 2]))
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 13- 0.1* dta[:, 2]))
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 10))
np.sum(np.square(dta[:, 0] - dta[:, 2] ** 2 * 15))
np.xcorr(dta[:, 0], dta[:, 2])
plt.xcorr(dta[:, 0], dta[:, 2], maxlags=10)
plt.plot(dff_e[3]);plt.plot(dff[43]);plt.legend(['ens3', 'neur43'])
plt.plot(dff_e[3]);plt.plot(dff[43]);plt.legend(['ens3', 'neur43'])
plt.plot(dff_e[3]);plt.plot(dff[43]);plt.legend(['ens3', 'neur43'])
reg1[0].params
np.sum(np.square(dta[:, 0] - dta[:, 1] * 0.87768))
np.sum(np.square(dta[:, 0] - dta[:, 1] * 0.87768 - 0.013167))
reg1[0].ssr
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0])
dta.shape
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0])
plt.plot(dta[:, 0]);plt.plot(dta[:, 1])
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0])
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0])
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0]+reg1[0].params[1])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]+reg1[0].params[1])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]+reg1[0].params[1]);plt.plot(dta[:, 2]);plt.legend(['residual', 'la'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]+reg1[0].params[1]);plt.plot(dta[:, 2]);plt.legend(['residual', 'la'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]+reg1[0].params[1]);plt.plot(dta[:, 2] * 2);plt.legend(['residual', 'la'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]+reg1[0].params[1]);plt.plot(dta[:, 2] * 2);plt.legend(['residual', 'la'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]+reg1[0].params[1])
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0]+reg1[0].params[1])
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0]+reg1[0].params[1]);plt.legend(['original', 't-1'])
dta[:3, 0]
dta[:3]
plt.plot(dta[:, 0]);plt.plot(dta[:, 1] * reg1[0].params[0]+reg1[0].params[1]);plt.legend(['original', 't-1'])
plt.plot((dta[:, 0]- dta[:, 1] * reg1[0].params[0]+reg1[0].params[1])/np.max(dta[:, 0]))
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]);plt.plot(dta[:, 0]); plt.legend(['diff', 'original'])
plt.plot(dta[:, 0]); plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]); plt.legend(['original', 'diff'])
dta_res = dta[:, 0] - dta[:, 1] * reg1[0].params[0] - reg1[0].params[1]
dtares = add_constant(dta_res, prepend=False)
dtares = add_constant(dta[:, 2], prepend=False)
res2dother = OLS(dta_res, dtares).fit()
res2dother.ssr
res2dres = OLS(dta_res, dtares).fit()
res2dres.ssr
res2down.ssr
res2djoint.ssr
res2dres.params
res2djoint.params
hf
neuron_acts = hf['neuron_acts']
neuron_acts = hf['neuron_act']
neuron_acts = np.array(hf['neuron_act'])
neuron_acts.shape
e_sel
dff_e.shape
deconv_e = neuron_acts[ner_ens_sel]
ner_ens_sel.shape
deconv_ner = neuron_acts[nerden]
deconv_e = neuron_acts[ner_ens_sel]
deconv_e = deconv_ner[ner_ens_sel]
deconv_e.shape
plt.xcorr(dff_e[3], dff[43])
dff.shape
plt.xcorr(deconv_e[3], deconv_ner[43])
plt.xcorr(deconv_e[3], deconv_ner[0])
plt.xcorr(deconv_e[3], deconv_ner[43])
resd = grangercausalitytests(np.vstack([deconv_e[3], deconv_ner[43]]).T, 2, verbose=False)
testd, regd = resd[1]
regd.shape
regd[0].ssr
regd[1].ssr
plt.plot(deconv_e[3]);plt.plot(deconv_ner[43]);plt.legend(['ens3', 'neur43'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]);plt.plot(deconv_ner[43]); plt.legend(['diff', 'original'])
deconv_ner = deconv_ner[:, :blen]
deconv_e = deconv_ner[ner_ens_sel]
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]);plt.plot(deconv_ner[43]); plt.legend(['diff', 'original'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]);plt.plot(deconv_ner[43] * 100); plt.legend(['diff', 'original'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]);plt.plot(deconv_ner[43] * 50); plt.legend(['diff', 'original'])
dta[:3]
dta_res.shape
dta_res[:3]
dtares[:3]
dff[43][:3]
dtadeconv = add_constant(deconv_ner[43][:-1], prepend=False)
res2ddeconv = OLS(dta_res, dtadeconv).fit()
res2ddeconv.ssr
regd[0].ssr
res2down.ssr
res2djoint.ssr
testd, regd = resd[2]
regd[0].ssr
regd[1].ssr
np.log(regd[0].ssr / regd[1].ssr)
np.log(res2down[0].ssr / res2djoint[1].ssr)
np.log(res2down.ssr / res2djoint.ssr)
res2down.ssr
res2djoint.ssr
res[1][1].srr
res[1][1][0].ssr
res[1][1][1].ssr
res1[1][1][1].ssr
res1[1][1][0].ssr
plt.xcorr(dff_e[3], dff[43])
plt.plot(deconv_e[3]);plt.plot(deconv_ner[43]);plt.legend(['ens3', 'neur43'])
plt.plot(dff_e[3]);plt.plot(dff[43]);plt.legend(['ens3', 'neur43'])
plt.xcorr(dff_e[3], dff[43])
np.correlate(dff_e[3][1:], dff[43][:-1])
np.correlate(dff_e[3][1:], dff[43][:-1]) / (np.linalg.norm(dff_e[3]) np.linalg.norm(dff[43][:-1]))
np.correlate(dff_e[3][1:], dff[43][:-1]) / (np.linalg.norm(dff_e[3]) *np.linalg.norm(dff[43][:-1]))
np.correlate(dff_e[3][1:], dff[43][:-1]) / (np.linalg.norm(dff_e[3]) *np.linalg.norm(dff[43]))
np.log(res1[0].ssr / res1[1].ssr)
np.log(reg1[0].ssr / reg1[1].ssr)
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]);plt.plot(dta[:, 2]); plt.legend(['diff', 'original'])
plt.plot(dta[:, 0]- dta[:, 1] * reg1[0].params[0]-reg1[0].params[1]);plt.plot(dta[:, 2]); plt.legend(['diff', 'original'])
plt.xcorr(dff_e[3], dff[43])
from statsmodels.tsa.tsatools import lagmat
lagmat(np.arange(10), maxlag=2, trim='forward')
lagmat(np.arange(10), maxlag=2, trim='forward', original='in')
%hist
