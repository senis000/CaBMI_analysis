
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
    online_data = np.asarray(f['online_data'])[:,2:]
    dff = np.asarray(f['dff'])
    cursor = np.asarray(f['cursor'])
    startBMI = np.asarray(f['trial_start'])[0]
    f.close()
    if np.isnan(np.sum(ens_neur)):
        print('Only ' + str(4 - np.sum(np.isnan(ens_neur))) + ' ensemble neurons')
    ens_neur = np.int16(ens_neur[~np.isnan(ens_neur)])
    if np.isnan(np.sum(e2_neur)):
        print('Only ' + str(2 - np.sum(np.isnan(e2_neur))) + ' e2 neurons')
    e2_neur = np.int16(e2_neur[~np.isnan(e2_neur)])
    
    com_ens = com[ens_neur, :]
    com_e2 = com[e2_neur, :]  
    
    dff_ens = dff[ens_neur, :]
    dff_e2 = dff[e2_neur, :]
        
    # depth
    depth_mean = np.nanmean(com_ens[:,2])
    depth_max = np.nanmax(com_ens[:,2])
    
#     if len(e2_neur) > 0:
#         depth_mean_e2 = np.nanmean(com_e2[:,2])
#         depth_max_e2 = np.nanmax(com_e2[:,2])
#     else:
#         depth_mean_e2 = np.nan
#         depth_max_e2 = np.nan
        
    # distance
    if ens_neur.shape[0]>1:
        auxdist = []
        for nn in np.arange(ens_neur.shape[0]):
            for nns in np.arange(nn+1, ens_neur.shape[0]):
                auxdist.append(scipy.spatial.distance.euclidean(com_ens[nn,:], com_ens[nns,:]))
        
        dist_mean = np.nanmean(auxdist)
        dist_max = np.nanmax(auxdist)

    
        # diff of depth
        auxddepth = []
        for nn in np.arange(ens_neur.shape[0]):
            for nns in np.arange(nn+1, ens_neur.shape[0]):
                auxddepth.append(scipy.spatial.distance.euclidean(com_ens[nn,2], com_ens[nns,2]))
        
        diffdepth_mean = np.nanmean(auxddepth)
        diffdepth_max = np.nanmax(auxddepth)    
    else:
        dist_mean = np.nan
        dist_max = np.nan
        diffdepth_mean = np.nan
        diffdepth_max = np.nan
    
    # dynamic range
    auxonstd = []
    for nn in np.arange(online_data.shape[1]):
        auxonstd.append(np.nanstd(online_data[:,nn]))
        
    onstd_mean = np.nanmean(auxonstd)
    onstd_max = np.nanmax(auxonstd)
    
    auxpostwhostd = []
    for nn in np.arange(dff_ens.shape[0]):
        auxpostwhostd.append(np.nanstd(dff_ens[nn,:]))
        
    post_whole_std_mean = np.nanmean(auxpostwhostd)
    post_whole_std_max = np.nanmax(auxpostwhostd)
    
    auxpostbasestd = []
    for nn in np.arange(dff_ens.shape[0]):
        auxpostbasestd.append(np.nanstd(dff_ens[nn,:startBMI]))
        
    post_base_std_mean = np.nanmean(auxpostbasestd)
    post_base_std_max = np.nanmax(auxpostbasestd)
    
    auxcursorstd = []
    cursor_std = np.nanstd(cursor)
    
    
    
    row_entry = np.asarray([depth_mean, depth_max, dist_mean, dist_max, diffdepth_mean, diffdepth_max, \
                             onstd_mean, onstd_max, post_whole_std_mean, post_whole_std_max, post_base_std_mean, \
                             post_base_std_max, cursor_std])
    
    return row_entry


def plot_basic_results(mat_results, mat_animal):
    '''
    Function to plot all the basic results in terms of linear regression 
    '''
    
    # depth mean
    


def create_dataframe(folder):
    animals = os.listdir(folder)
    for aa,animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        mat_animal = np.zeros((len(filenames), 13)) + np.nan
        for dd,filename in enumerate(filenames):
            day = filename[-17:-11]
            print ('Analyzing animal: ' + animal + ' day: ' + day)
            mat_animal[dd, :] = xga.basic_entry (folder, animal, day)
        plot_basic_results(mat_results, mat_animal)
        
        

        
    
    
    
