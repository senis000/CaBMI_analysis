import numpy as np
import caiman as cm
import tifffile, os, h5py, time
from skimage import io
from scipy.io import loadmat
from collections.abc import Iterable
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto
from SNR_test import *

def slave1():
    root = "/Users/albertqu/Documents/2.Courses/CogSci127/proj/data/"
    # "/media/user/Seagate Backup Plus Drive/raw/IT5/190212/"  # DATA ROOT
    tiff_path = os.path.join(root, "baseline_00001.tif")
    out = root  # os.path.join(root, 'splits')
    if not os.path.exists(out):
        os.makedirs(out)
    # print("start splitting")
    # # nodecay
    # fname0 = extract_planes(tiff_path, out, 0)
    # print('finish nodecay')
    # #decay
    # fname1 = extract_planes(tiff_path, out, 0, decay=0.9999)
    # print('finish decay')
    fname1 = os.path.join(out, 'plane0_decay.tif')
    print(fname1)
    # get frame rate
    fr = 9.72365281#loadmat(os.path.join(root, 'wmat.mat'))['fr'].item((0, 0))
    caiman_main(fr, [fname1], os.path.join(out, 'plane0_decay.hdf5'))


if __name__ == '__main__':
    animal, day = 'IT5', '190212'
    root = "/media/user/Seagate Backup Plus Drive/raw/{}/{}/".format(animal, day)  # DATA ROOT
    tiff_path = os.path.join(root, "baseline_00001.tif")
    out = os.path.join(root, 'splits')
    if not os.path.exists(out):
        os.makedirs(out)
    # print("start splitting")
    # # nodecay
    # fname0 = extract_planes(tiff_path, out, 0)
    # print('finish nodecay')
    #decay
    #fname1 = extract_planes(tiff_path, out, 0, decay=0.9999)
    print('finish decay')
    fname0 = os.path.join(out, 'plane0_nodecay.tif')
    fname1 = os.path.join(out, 'plane0_decay.tif')
    # print(fname0)
    # get frame rate
    fr = loadmat(os.path.join(root, 'wmat.mat'))['fr'].item((0, 0))
    caiman_main(fr, [fname0], os.path.join(out, 'test_{}_{}_plane0_nodecay.hdf5'.format(animal, day)))
    # hfile0 = "{}_{}_plane0_nodecay.hdf5".format(animal, day)
    # OnACID_A_init(fr, [fname0], os.path.join(out, 'onacid_'+hfile0),
    #               os.path.join(out, hfile0))
    # hfile1 = "{}_{}_plane0_decay.hdf5".format(animal, day)
    # OnACID_A_init(fr, [fname1], os.path.join(out, 'onacid_' + hfile1),
    #               os.path.join(out, hfile1))
