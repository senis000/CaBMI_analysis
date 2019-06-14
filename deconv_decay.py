import h5py
from shuffling_functions import signal_partition
import numpy as np
import matplotlib.pyplot as plt

def expr(seg):
    norm = [s / np.max(s) for s in seg]
    plt.subplot(211)
    for s in seg:
        plt.plot(s)
    plt.subplot(212)
    for n in norm:
        plt.plot(n)
    plt.show()

f = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/processed/full_IT3_180928__data.hdf5"
h = h5py.File(f, 'r')
C = np.array(h['C'])[h['nerden']]
s1 = C[0]
s2 = C[1]

pr, ipis = signal_partition(s1, 30)
pr2, _ = signal_partition(s2, 30)
seg1 = [s1[p[0]:p[1]] for p in pr]
seg2 = [s2[p[0]:p[1]] for p in pr2]

prp, _ = signal_partition(s1, 50)
seg1p = [s1[p[0]:p[1]] for p in prp]
max_seg = max(seg1, key=lambda x: len(x))
len(max_seg)
plt.plot(max_seg)
plt.show()
max_seg = max(seg1p, key=lambda x: len(x))
plt.plot(max_seg)
plt.show()
plt.hist([len(s) for s in seg1])
plt.show()
plt.show()
mod_seg = [s for s in seg1p if len(s) <= 50]
expr(mod_seg)

expr(seg1p)
expr(mod_seg)
expr(mod_seg[:10])
expr(mod_seg[10:20])