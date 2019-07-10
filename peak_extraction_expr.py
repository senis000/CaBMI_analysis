import h5py
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from utils_cabmi import median_absolute_deviation


f = h5py.File("/Users/albertqu/Documents/7.Research/BMI/analysis_data/processed/PT6/full_PT6_181128__data.hdf5", 'r')
med = np.median(f['C'][0]);med_std = median_absolute_deviation(f['C'][0])
mean = np.mean(f['C'][0]);std=np.std(f['C'][0])
peakind=signal.find_peaks_cwt(f['C'][0], np.arange(1, 20))
peakind1 = peakind
peakind2 = peakind[f['C'][0][peakind]>med+med_std]
peakind3 = peakind[f['C'][0][peakind]>med+2*med_std]
peakind4 = peakind1[f['C'][0][peakind1]>mean+std]
peakind5 = peakind1[f['C'][0][peakind1]>mean+2*std]
peakind = peakind2
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
axes[0].plot(f['C'][0]);axes[0].plot(peakind, f['C'][0][peakind], 'x');axes[1].plot(f['neuron_act'][1]);plt.show()