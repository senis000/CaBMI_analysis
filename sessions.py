In [1]: val = "/Users/albertqu/Documents/7.Research/BMI/analysis_data/processed/full_IT3_180928__data.hdf5"

In [2]: import h5py

In [3]: ks = h5py.File(val, 'r')

In [4]: import matplotlib.pyplot as plt; plt.plot(ks['hits'])
Out[4]: [<matplotlib.lines.Line2D at 0x122241a90>]

In [5]: plt.show()

In [6]: grads = [ks['hits'][i]- ks['hits'][i-1] for i in range(1, len(ks['hits']
   ...: ))]

In [7]: grads
Out[7]:
[282.0,
 761.0,
 206.0,
 398.0,
 273.0,
 1667.0,
 248.0,
 977.0,
 264.0,
 1823.0,
 221.0,
 517.0,
 236.0,
 544.0,
 263.0,
 1857.0,
 1628.0,
 2402.0,
 166.0,
 268.0,
 193.0,
 186.0,
 1710.0,
 665.0,
 535.0,
 317.0]

In [8]: plt.plot(grads)
Out[8]: [<matplotlib.lines.Line2D at 0x1225706a0>]

In [9]: plt.show()

In [10]: freqs = np.copy(ks['freq'])
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-10-8375f25ef7ef> in <module>
----> 1 freqs = np.copy(ks['freq'])

NameError: name 'np' is not defined

In [11]: import numpy as np

In [12]: freqs = np.copy(ks['freq'])

In [13]: plt.plot(freqs)
Out[13]: [<matplotlib.lines.Line2D at 0x12a8a8198>]

In [14]: plt.show()

In [15]: plt.plot(freqs);plt.show()

In [16]: freqs[5000:7000]
Out[16]: array([nan, nan, nan, ..., nan, nan, nan], dtype=float32)

In [17]: nnan = freqs[freqs != np.nan]

In [18]: len(nnan)
Out[18]: 20500

In [19]: len(freqs)
Out[19]: 20500

In [20]: nnan = freqs[np.logical_not(np.isnan(freqs))]

In [21]: plt.plot(nnan); plt.show()

In [22]: moving_windows = 20 * ks['fr']
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-22-9a9c849e3831> in <module>
----> 1 moving_windows = 20 * ks['fr']

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

/anaconda3/lib/python3.6/site-packages/h5py/_hl/group.py in __getitem__(self, name)
    175                 raise ValueError("Invalid HDF5 object reference")
    176         else:
--> 177             oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
    178
    179         otype = h5i.get_type(oid)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5o.pyx in h5py.h5o.open()

KeyError: "Unable to open object (object 'fr' doesn't exist)"

In [23]: moving = 200

In [24]: nodes = [nnan[i, i+moving] for i in range(len(nnan)-moving+1)]
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-24-47955453f9a2> in <module>
----> 1 nodes = [nnan[i, i+moving] for i in range(len(nnan)-moving+1)]

<ipython-input-24-47955453f9a2> in <listcomp>(.0)
----> 1 nodes = [nnan[i, i+moving] for i in range(len(nnan)-moving+1)]

IndexError: too many indices for array

In [25]: nodes = [nnan[i:i+moving] for i in range(len(nnan)-moving+1)]

In [26]: max_freq, min_freq = np.max(nnan), np.min(nnan)

In [27]: max_freq
Out[27]: 18000.0

In [28]: min_freq
Out[28]: 0.0

In [29]: k_bin = 10

In [30]: np.arange(min_freq, max_freq, max_freq - min_freq // 10)
Out[30]: array([0.])

In [31]: np.arange(min_freq, max_freq, (max_freq - min_freq) // 10)
Out[31]:
array([    0.,  1800.,  3600.,  5400.,  7200.,  9000., 10800., 12600.,
       14400., 16200.])

In [32]: bins = np.arange(min_freq, max_freq, (max_freq - min_freq) // 10)

In [33]: ww = [np.histogram(node, bins, density=True) for node in nodes]

In [34]: plt.plot(ww[0][1], ww[0][0]);plt.show()
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-34-777ec62f68ec> in <module>
----> 1 plt.plot(ww[0][1], ww[0][0]);plt.show()

/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py in plot(scalex, scaley, data, *args, **kwargs)
   2809     return gca().plot(
   2810         *args, scalex=scalex, scaley=scaley, **({"data": data} if data
-> 2811         is not None else {}), **kwargs)
   2812
   2813

/anaconda3/lib/python3.6/site-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1808                         "the Matplotlib list!)" % (label_namer, func.__name__),
   1809                         RuntimeWarning, stacklevel=2)
-> 1810             return func(ax, *args, **kwargs)
   1811
   1812         inner.__doc__ = _add_data_doc(inner.__doc__,

/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py in plot(self, scalex, scaley, *args, **kwargs)
   1609         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D._alias_map)
   1610
-> 1611         for line in self._get_lines(*args, **kwargs):
   1612             self.add_line(line)
   1613             lines.append(line)

/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_base.py in _grab_next_args(self, *args, **kwargs)
    391                 this += args[0],
    392                 args = args[1:]
--> 393             yield from self._plot_args(this, kwargs)
    394
    395

/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_base.py in _plot_args(self, tup, kwargs)
    368             x, y = index_of(tup[-1])
    369
--> 370         x, y = self._xy_from_xy(x, y)
    371
    372         if self.command == 'plot':

/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_base.py in _xy_from_xy(self, x, y)
    229         if x.shape[0] != y.shape[0]:
    230             raise ValueError("x and y must have same first dimension, but "
--> 231                              "have shapes {} and {}".format(x.shape, y.shape))
    232         if x.ndim > 2 or y.ndim > 2:
    233             raise ValueError("x and y can be no greater than 2-D, but have "

ValueError: x and y must have same first dimension, but have shapes (10,) and (9,)

In [35]: ww[0][1]
Out[35]:
array([    0.,  1800.,  3600.,  5400.,  7200.,  9000., 10800., 12600.,
       14400., 16200.])

In [36]: ww[0][0]
Out[36]:
array([7.22222222e-05, 2.77777778e-06, 0.00000000e+00, 1.11111111e-05,
       5.00000000e-05, 4.19444444e-04, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00])

In [37]: plt.plot(ww[0][0]);plt.show()

In [38]: plt.plot(ww[1][0]);plt.show()

In [39]: plt.plot(ww[30][0]);plt.show()

In [40]: plt.plot(ww[100][0]);plt.show()

In [41]: plt.plot(ww[len(ww)-1][0]);plt.show()

In [42]: plt.subplot(321)
Out[42]: <matplotlib.axes._subplots.AxesSubplot at 0x12a3bf7b8>

In [43]: tot = len(ww)

In [44]: plt.plot(0 * tot // 6);plt.show()

In [45]: plt.plot(ww[0][0]);

In [46]: plt.subplot(322);plt.plot(ww[tot // 6][0])
Out[46]: [<matplotlib.lines.Line2D at 0x12a4b0780>]

In [47]: plt.subplot(323);plt.plot(ww[2 * tot // 6][0])
Out[47]: [<matplotlib.lines.Line2D at 0x12a4f4c88>]

In [48]: plt.subplot(324);plt.plot(ww[3 * tot // 6][0])
Out[48]: [<matplotlib.lines.Line2D at 0x12a5839e8>]

In [49]: plt.subplot(325);plt.plot(ww[4 * tot // 6][0])
Out[49]: [<matplotlib.lines.Line2D at 0x12a5067f0>]

In [50]: plt.subplot(326);plt.plot(ww[5 * tot // 6][0])
Out[50]: [<matplotlib.lines.Line2D at 0x12a5b5eb8>]

In [51]: plt.show()

In [52]: plt.subplot(321);plt.plot(ww[0][0]);

In [53]: plt.subplot(322);plt.plot(ww[tot // 6][0])
Out[53]: [<matplotlib.lines.Line2D at 0x12a6a54a8>]

In [54]: plt.subplot(323);plt.plot(ww[2 * tot // 6][0])
Out[54]: [<matplotlib.lines.Line2D at 0x122921198>]

In [55]: plt.subplot(324);plt.plot(ww[3 * tot // 6][0])
Out[55]: [<matplotlib.lines.Line2D at 0x12ae8e208>]

In [56]: plt.subplot(325);plt.plot(ww[4 * tot // 6][0])
Out[56]: [<matplotlib.lines.Line2D at 0x12aeb67f0>]

In [57]: plt.subplot(326);plt.plot(ww[5 * tot // 6][0])
Out[57]: [<matplotlib.lines.Line2D at 0x12ba6d2b0>]

In [58]: plt.show()

In [59]: plt.plot(ww[tot-1][0]);plt.show()

In [60]: bins
Out[60]:
array([    0.,  1800.,  3600.,  5400.,  7200.,  9000., 10800., 12600.,
       14400., 16200.])

In [61]: bins = np.concatenate((bins, [18000]))

In [62]: bins
Out[62]:
array([    0.,  1800.,  3600.,  5400.,  7200.,  9000., 10800., 12600.,
       14400., 16200., 18000.])

In [63]: ww = [np.histogram(node, bins, density=True) for node in nodes]

In [64]: plt.plot(ww[0][0]);plt.show()

In [65]: plt.plot(ww[0][0]);plt.xticks(np.arange(len(bins)), bins);plt.show()

In [66]: plt.plot(ww[0][0] * 1800);plt.xticks(np.arange(len(bins)), bins);plt.show()

In [67]: plt.subplot(321);plt.plot(ww[0][0]);plt.xticks(np.arange(len(bins)), bins);

In [68]: plt.subplot(322);plt.plot(ww[1][0]);plt.xticks(np.arange(len(bins)), bins);

In [69]: plt.subplot(323);plt.plot(ww[2][0]);plt.xticks(np.arange(len(bins)), bins);

In [70]: plt.subplot(324);plt.plot(ww[3][0]);plt.xticks(np.arange(len(bins)), bins);

In [71]: plt.subplot(325);plt.plot(ww[4][0]);plt.xticks(np.arange(len(bins)), bins);

In [72]: plt.subplot(326);plt.plot(ww[len(ww)-1][0]);plt.xticks(np.arange(len(bins)), bins);

In [73]: plt.show()

In [74]: plt.subplot(321);plt.plot(ww[0][0]* 1800);plt.xticks(np.arange(len(bins)), bins, rotation=45);

In [75]: plt.subplot(322);plt.plot(ww[1][0]* 1800);plt.xticks(np.arange(len(bins)), bins, rotation=45);

In [76]: In [74]: plt.subplot(323);plt.plot(ww[2][0]* 1800);plt.xticks(np.arange(len(bins)), bins, rotation=45)
    ...: ;
    ...:
    ...: In [75]: plt.subplot(324);plt.plot(ww[3][0]* 1800);plt.xticks(np.arange(len(bins)), bins, rotation=45)
    ...: ;

In [77]: plt.subplot(325);plt.plot(ww[4][0]* 1800);plt.xticks(np.arange(len(bins)), bins, rotation=45);
    ...:
    ...:
    ...: plt.subplot(326);plt.plot(ww[len(ww)-1][0]* 1800);plt.xticks(np.arange(len(bins)), bins, rotation=45);
    ...:

In [78]: plt.show()

In [79]: plt.plot(kk['cursor'])
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-79-fc06cae6b0f1> in <module>
----> 1 plt.plot(kk['cursor'])

NameError: name 'kk' is not defined

In [80]: plt.plot(ks['cursor'])
Out[80]: [<matplotlib.lines.Line2D at 0x12b75c240>]

In [81]: plt.show()

In [82]: plt.plot(nnan)
Out[82]: [<matplotlib.lines.Line2D at 0x12b7fb8d0>]

In [83]: plt.show()

In [84]: plt.plot(freqs);plt.scatter(ks['hits'], np.full_like(ks['hits'], 7200), 'r');plt.scatter(ks['hits'], n
    ...: p.full_like(ks['miss'], 7200), 'g');
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-84-dfae4f9c203a> in <module>
----> 1 plt.plot(freqs);plt.scatter(ks['hits'], np.full_like(ks['hits'], 7200), 'r');plt.scatter(ks['hits'], np.full_like(ks['miss'], 7200), 'g');

/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py in scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, data, **kwargs)
   2860         vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths,
   2861         verts=verts, edgecolors=edgecolors, **({"data": data} if data
-> 2862         is not None else {}), **kwargs)
   2863     sci(__ret)
   2864     return __ret

/anaconda3/lib/python3.6/site-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1808                         "the Matplotlib list!)" % (label_namer, func.__name__),
   1809                         RuntimeWarning, stacklevel=2)
-> 1810             return func(ax, *args, **kwargs)
   1811
   1812         inner.__doc__ = _add_data_doc(inner.__doc__,

/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py in scatter(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, **kwargs)
   4295                 offsets=offsets,
   4296                 transOffset=kwargs.pop('transform', self.transData),
-> 4297                 alpha=alpha
   4298                 )
   4299         collection.set_transform(mtransforms.IdentityTransform())

/anaconda3/lib/python3.6/site-packages/matplotlib/collections.py in __init__(self, paths, sizes, **kwargs)
    899         Collection.__init__(self, **kwargs)
    900         self.set_paths(paths)
--> 901         self.set_sizes(sizes)
    902         self.stale = True
    903

/anaconda3/lib/python3.6/site-packages/matplotlib/collections.py in set_sizes(self, sizes, dpi)
    872             self._sizes = np.asarray(sizes)
    873             self._transforms = np.zeros((len(self._sizes), 3, 3))
--> 874             scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor
    875             self._transforms[:, 0, 0] = scale
    876             self._transforms[:, 1, 1] = scale

TypeError: ufunc 'sqrt' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

In [85]: plt.plot(freqs);plt.scatter(ks['hits'], np.full_like(ks['hits'], 7200), c='r');plt.scatter(ks['hits'],
    ...:  np.full_like(ks['miss'], 7200), c='g');
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-85-9101633e12f7> in <module>
----> 1 plt.plot(freqs);plt.scatter(ks['hits'], np.full_like(ks['hits'], 7200), c='r');plt.scatter(ks['hits'], np.full_like(ks['miss'], 7200), c='g');

/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py in scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, data, **kwargs)
   2860         vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths,
   2861         verts=verts, edgecolors=edgecolors, **({"data": data} if data
-> 2862         is not None else {}), **kwargs)
   2863     sci(__ret)
   2864     return __ret

/anaconda3/lib/python3.6/site-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1808                         "the Matplotlib list!)" % (label_namer, func.__name__),
   1809                         RuntimeWarning, stacklevel=2)
-> 1810             return func(ax, *args, **kwargs)
   1811
   1812         inner.__doc__ = _add_data_doc(inner.__doc__,

/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py in scatter(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, **kwargs)
   4180         y = np.ma.ravel(y)
   4181         if x.size != y.size:
-> 4182             raise ValueError("x and y must be the same size")
   4183
   4184         if s is None:

ValueError: x and y must be the same size

In [86]: plt.plot(freqs);plt.scatter(ks['hits'], np.full_like(ks['hits'], 7200), c='r');plt.scatter(ks['miss'],
    ...:  np.full_like(ks['miss'], 7200), c='g');

In [87]: plt.show()

In [88]: ks['hits']
Out[88]: <HDF5 dataset "hits": shape (27,), type "<f8">

In [89]: print(ks['hits'])
<HDF5 dataset "hits": shape (27,), type "<f8">

In [90]: np.array(ks['hits'])
Out[90]:
array([ 9988., 10270., 11031., 11237., 11635., 11908., 13575., 13823.,
       14800., 15064., 16887., 17108., 17625., 17861., 18405., 18668.,
       20525., 22153., 24555., 24721., 24989., 25182., 25368., 27078.,
       27743., 28278., 28595.])

In [91]: plt.plot(freqs);plt.scatter(ks['hits'], np.full_like(ks['hits'], 7200), c='r');plt.scatter(ks['miss'],
    ...:  np.full_like(ks['miss'], 7200), c='g');plt.show()

In [92]: len(freqs)
Out[92]: 20500

In [93]: ks.__dict__.keys()
Out[93]: dict_keys(['_swmr_mode', '_id'])

In [94]: ks.__dict__
Out[94]: {'_swmr_mode': False, '_id': <h5py.h5f.FileID at 0x113869200>}

In [95]: ks['_id'].__dict__
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-95-f4abb202cafc> in <module>
----> 1 ks['_id'].__dict__

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

/anaconda3/lib/python3.6/site-packages/h5py/_hl/group.py in __getitem__(self, name)
    175                 raise ValueError("Invalid HDF5 object reference")
    176         else:
--> 177             oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
    178
    179         otype = h5i.get_type(oid)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5o.pyx in h5py.h5o.open()

KeyError: "Unable to open object (object '_id' doesn't exist)"

In [96]: for k in ks:
    ...:     print(k)
    ...:
C
Nsparse
SNR
array_miss
array_t1
base_im
com_cm
cursor
dff
ens_neur
freq
hits
miss
nerden
neuron_act
online_data
red_im
redlabel
trial_end
trial_start

In [97]: len(ks['online_data'])
Out[97]: 12279

In [98]: len(ks['online_data']) / 10
Out[98]: 1227.9

In [99]: len(ks['online_data']) / 600
Out[99]: 20.465

In [100]: ks['C'].shape
Out[100]: (1769, 29500)

In [101]: plt.plot(freqs);plt.scatter(ks['hits'] - ks['blen'], np.full_like(ks['hits'], 7200), c='r');plt.scatt
     ...: er(ks['miss']-ks['blen'], np.full_like(ks['miss'], 7200), c='g');plt.show()
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-101-53b7f7441148> in <module>
----> 1 plt.plot(freqs);plt.scatter(ks['hits'] - ks['blen'], np.full_like(ks['hits'], 7200), c='r');plt.scatter(ks['miss']-ks['blen'], np.full_like(ks['miss'], 7200), c='g');plt.show()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

/anaconda3/lib/python3.6/site-packages/h5py/_hl/group.py in __getitem__(self, name)
    175                 raise ValueError("Invalid HDF5 object reference")
    176         else:
--> 177             oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
    178
    179         otype = h5i.get_type(oid)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5o.pyx in h5py.h5o.open()

KeyError: "Unable to open object (object 'blen' doesn't exist)"

In [102]: ks['blen']
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-102-327c7ec8bab2> in <module>
----> 1 ks['blen']

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

/anaconda3/lib/python3.6/site-packages/h5py/_hl/group.py in __getitem__(self, name)
    175                 raise ValueError("Invalid HDF5 object reference")
    176         else:
--> 177             oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
    178
    179         otype = h5i.get_type(oid)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5o.pyx in h5py.h5o.open()

KeyError: "Unable to open object (object 'blen' doesn't exist)"

In [103]: for k in ks:
     ...:     print(k)
     ...:
C
Nsparse
SNR
array_miss
array_t1
base_im
com_cm
cursor
dff
ens_neur
freq
hits
miss
nerden
neuron_act
online_data
red_im
redlabel
trial_end
trial_start

In [104]: ks['base_im']
Out[104]: <HDF5 dataset "base_im": shape (256, 256, 4), type "<f8">

In [105]: ks.blen
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-105-e8283b674f25> in <module>
----> 1 ks.blen

AttributeError: 'File' object has no attribute 'blen'

In [106]: plt.plot(freqs);plt.scatter(ks['hits'] - 9000, np.full_like(ks['hits'], 7200), c='r');plt.scatter(ks[
     ...: 'miss']-9000, np.full_like(ks['miss'], 7200), c='g');plt.show()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-106-6de1e3452f09> in <module>
----> 1 plt.plot(freqs);plt.scatter(ks['hits'] - 9000, np.full_like(ks['hits'], 7200), c='r');plt.scatter(ks['miss']-9000, np.full_like(ks['miss'], 7200), c='g');plt.show()

TypeError: unsupported operand type(s) for -: 'Dataset' and 'int'

In [107]: plt.plot(freqs);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.s
     ...: catter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.show()

In [108]: plt.plot(freqs);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.s
     ...: catter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.legend(['freq', 'hits',
     ...: 'miss']);plt.show()

In [109]: plt.plot(freqs);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.s
     ...: catter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.plots(ks['cursor']);plt.
     ...: legend(['freq', 'hits', 'miss', 'cursor']);plt.show()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-109-d607313bc160> in <module>
----> 1 plt.plot(freqs);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.plots(ks['cursor']);plt.legend(['freq', 'hits', 'miss', 'cursor']);plt.show()

AttributeError: module 'matplotlib.pyplot' has no attribute 'plots'

In [110]: plt.plot(freqs);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.s
     ...: catter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.plot(ks['cursor']);plt.l
     ...: egend(['freq', 'hits', 'miss', 'cursor']);plt.show()

In [111]: plt.plot(freqs);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.s
     ...: catter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.plot(freqs); plt.plot(ks
     ...: ['cursor']);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.scatt
     ...: er(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.legend(['freq', 'cursor', 'h
     ...: its', 'miss']);plt.show()plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show()
  File "<ipython-input-111-b56dcaf7dc8c>", line 1
    plt.plot(freqs);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.plot(freqs); plt.plot(ks['cursor']);plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show()plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show()
                                                                                                                                                                                                                                                                                                                                                                                                                                                   ^
SyntaxError: invalid syntax


In [112]: plt.plot(freqs); plt.plot(ks['cursor']); plt.scatter(np.array(ks['hits']) - 9000, np.full_like(ks['hi
     ...: ts'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['miss'], 7200), c='g');plt
     ...: .legend(['freq', 'cursor', 'hits', 'miss']);plt.show()

In [113]: plt.plot(freqs); plt.plot(-np.array(ks['cursor'])) * 2000; plt.scatter(np.array(ks['hits']) - 9000, n
     ...: p.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['miss'],
     ...:  7200), c='g');plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show()

In [114]: plt.plot(freqs); plt.plot(-np.array(ks['cursor']) * 2000); plt.scatter(np.array(ks['hits']) - 9000, n
     ...: p.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['miss'],
     ...:  7200), c='g');plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show()

In [115]: plt.plot(freqs); plt.plot(7200 + np.array(ks['cursor']) * 10); plt.scatter(np.array(ks['hits']) - 900
     ...: 0, np.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['mis
     ...: s'], 7200), c='g');plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show()

In [116]: plt.plot(freqs); plt.plot(8000 + np.array(ks['cursor']) * 100); plt.scatter(np.array(ks['hits']) - 90
     ...: 00, np.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['mi
     ...: ss'], 7200), c='g');plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show()

In [117]: plt.plot(freqs); plt.plot(8000 + np.array(ks['cursor']) * 500); plt.scatter(np.array(ks['hits']) - 90
     ...: 00, np.full_like(ks['hits'], 7200), c='r');plt.scatter(np.array(ks['miss'])-9000, np.full_like(ks['mi
     ...: ss'], 7200), c='g');plt.legend(['freq', 'cursor', 'hits', 'miss']);plt.show() 