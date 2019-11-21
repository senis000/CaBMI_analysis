from caiman.source_extraction.cnmf import deconvolution
import h5py
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
import os
import pandas as pd

animal, day = 'IT2', '181115'
root = "/Volumes/DATA_01/NL/layerproject/"
f = h5py.File(root+"processed/{}/full_{}_{}__data.hdf5".format(animal, animal, day))
csvf = root+"raw/{}/{}/bmi_IntegrationRois_00001.csv".format(animal, day)
plots = "/Users/albertqu/Documents/7.Research/BMI/plots/caiman_test/corrDFFdoublepass_{}_{}".format(animal,
                                                                                                    day)
if not os.path.exists(plots):
    os.makedirs(plots)

number_planes_total = 6
C = np.array(f['C'])
dff = np.array(f['dff'])
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