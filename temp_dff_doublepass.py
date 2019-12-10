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
hf = h5py.File(root + "processed/{}/full_{}_{}__data.hdf5".format(animal, animal, day))
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



