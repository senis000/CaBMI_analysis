
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import scipy
import h5py
import pandas as pd
from itertools import combinations
from scipy.stats.mstats import zscore
from scipy import ndimage
import copy
from matplotlib import interactive

from utils_loading import encode_to_filename

interactive(True)

def basic_entry (folder, animal, day):
    '''
    to generate an entry to the pd dataframe with the basic features
    Needs to have a folder with all_processed together
    '''
    
    file_template = "full_{}_{}__data.hdf5"
    file_name = os.path.join(folder, animal, file_template.format(animal, day))
    f = h5py.File(file_name, 'r')

    com = np.asarray(f['com'])
    nerden = np.asarray(f['nerden'])
    ens_neur = np.asarray(f['ens_neur'])
    e2_neur = np.asarray(f['e2_neur'])
    if np.isnan(np.sum(ens_neur)):
        print('We dont have all ensemble neurons')
    ens_neur = np.int16(ens_neur[~np.isnan(ens_neur)])
    if np.isnan(np.sum(e2_neur)):
        print('We dont have all e2 neurons')
    e2_neur = np.int16(e2_neur[~np.isnan(e2_neur)])
    
    com_ens = com[ens_neur, :]
    com_e2 = com[e2_neur, :]  
        
    # depth
    depth_mean = np.nanmean(com_ens[:,2])
    depth_max = np.nanmax(com_ens[:,2])
    
    if 
    depth_mean_e2 = np.nanmean(com_e2[:,2])
    depth_max_e2 = np.nanmax(com_e2[:,2])
    
    
    return df_entry
    
    
