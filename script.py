from builtins import str
import logging
import numpy as np
import scipy
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


def something():
    from run_at_night_cabmi import tonightAllRun_uzsh
    %time tonightAllRun_uzsh()
    from pipeline import *
    bmi1_old = h5py.File("/media/user/Seagate Backup Plus Drive1/raw/IT5/190129/bmi__1.hdf5", 'r')
    bmi1_old = h5py.File("/media/user/Seagate Backup Plus Drive1/raw/IT5/190129/bmi__1_old.hdf5", 'r')
    bmi1 = h5py.File("/media/user/Seagate Backup Plus Drive1/raw/IT5/190129/bmi__1.hdf5", 'r')
    np.sum(np.any(np.isnan(bmi1['dff']), axis=1))
    np.sum(np.any(np.isnan(bmi1_old['dff']), axis=1))
    np.allclose(bmi1['dff'], bmi_old['C'])
    np.allclose(bmi1['C'], bmi1_old['C'])
    bmi1.shape
    bmi1['C'].shape
    bmi1_old['C'].shape
    plt
    plt.plot(bmi1['C'][0], bmi1_old['C'][0])
    plt.plot(bmi1['C'][579], bmi1_old['C'][579])
    plt.plot(bmi1['C'][2], bmi1_old['C'][2])
    dis = []; for i in range(580):
        if not np.allclose(bmi1['C'][i], bmi1_old['C'][i]):
            dis.append(i)
    dis = []
    for i in range(580):
        if not np.allclose(bmi1['C'][i], bmi1_old['C'][i]):
            dis.append(i)
    dis
    dis[0]
    len(dis)
    np.allclose(bmi1['C'][0], bmi1_old['C'][0])
    plt.plot(np.vstack([bmi1['C'][0], bmi1_old['C'][0]]).T);plt.show()
    d = bmi1['C']-bmi1_old['C']
    d = np.array(bmi1['C'])-np.array(bmi1_old['C'])
    d = np.array(bmi1['C'])-np.array(bmi1_old['C'][:580])
    plt.plot(d[0])
    np.max(d[0])
    np.mean(bmi1['C'][0])
    diff = []
    diff = [];
    dis = []
    for i in range(580):
        mC = np.mean(bmi1['C'][i])
        if np.max(d[i])< mC / 10:
            dis.append(i)
    dis[0]
    dis = []
    for i in range(2):
        mC = np.mean(bmi1['C'][i])
        if np.max(d[i])< mC / 10:
            print(np.max(d[i]), mC)
            dis.append(i)
    dis = []
    for i in range(2):
        mC = np.mean(bmi1['C'][i])
        if np.max(d[i])> mC / 10:
            print(np.max(d[i]), mC)
            dis.append(i)
    dis[0]
    dis
    dis = []
    for i in range(2):
        mC = np.mean(bmi1['C'][i])
        if np.max(d[i])> mC / 100:
            print(np.max(d[i]), mC)
            dis.append(i)
    dis[0]
    dis = []
    for i in range(2):
        mC = np.mean(bmi1['C'][i])
        if np.max(d[i])> 10:
            print(np.max(d[i]), mC)
            dis.append(i)
    dis
    np.max(d[579])
    dis = []
    for i in range(2):
        mC = np.mean(bmi1['C'][i])
        if np.max(d[i])> mC / 10:
            dis.append(i)
    dis = []
    for i in range(580):
        mC = np.mean(bmi1['C'][i])
        if np.max(d[i])> mC / 10:
            dis.append(i)
    dis[0]
    np.max(d[75])
    plt.plot(np.vstack((bmi1['C'][75], bmi1_old['C'][75])).T)
    dis = []
    for i in range(580):
        mC = np.std(bmi1['C'][i])
        if np.max(d[i])> mC:
            dis.append(i)
    dis[0]
    plt.plot(np.vstack((bmi1['C'][235], bmi1_old['C'][235])).T)
    np.corrcoef(bmi1['C'][235], bmi1_old['C'][235])
    dis = []
    for i in range(580):
        dis.append(np.corrcoef(bmi1['C'][i], bmi1_old['C'][i]))
    dis = np.array(dis)
    np.min(dis)
    plt.plot(dis);plt.ylim((-0.1, 1.1))
    dis = []
    for i in range(580):
        dis.append(np.corrcoef(bmi1['C'][i], bmi1_old['C'][i])[1, 0])
    plt.plot(dis);plt.ylim((-0.1, 1.1))
    plt.scatter(np.arange(580), dis);plt.ylim((-0.1, 1.1))
    C = np.array(bmi1['C'])
    oC = np.array(bmi1_old['C'])
    plt.plot(C[236], C[237])
    C.shape
    oC.shape
    plt.plot(C[236], oC[237])
    plt.plot(C[580], oC[581])
    plt.plot(C[580-1], oC[580])
    nC = np.vstack((oC[:236], oC[237:]))
    dis = []
    for i in range(580):
        dis.append(np.corrcoef(C[i], nC[i])[1, 0])
    plt.scatter(np.arange(580), dis);plt.ylim((-0.1, 1.1))
    plt.plot(C[238], oC[239])
    plt.plot(np.vstack((bmi1['C'][238], bmi1_old['C'][239])).T)