import tifffile, os
from collections.abc import Iterable
from skimage import io
import numpy as np


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

        for i in range(totlen):
            img = tif.pages[nplanes * i + p].asarray()
            bigmem[i, :, :] = img * decay ** i if decay != 1.0 else img
        bigmem.flush()

        # Read from mmap, save as tifs
        tifn = os.path.join(outpath, tifn)
        fname = tifn + "{}_{}decay.tif".format(p, "" if decay != 1 else "no")
        io.imsave(fname, bigmem, plugin='tifffile')
        # Delete mmap
        if del_mmap:
            os.remove(fnamemm)
            del bigmem
        fnames.append(fname)
    return fnames

if __name__ == '__main__':
    root = "/Volumes/ALBERTSHD/BMI/CarmenaLab/processed/IT3"  # DATA ROOT
    tiff_path = os.path.join(root, "baseline_00001.tif")
    out = os.path.join(root, 'splits')
    # nodecay
    fname2 = extract_planes(tiff_path, out, 4)
    #decay
    fname1 = extract_planes(tiff_path, out, 4, decay=0.9999)

