from __future__ import division
from __future__ import print_function
from macpath import basename
#from matplotlib.tests.test_backend_pgf import test_bbox_inches
#from __main__ import traceback
# Pipeline to obtain rois from 2p data based on Caiman

__author__ = 'Nuria'


from numpy.distutils.system_info import dfftw_info
try:
    zip, str, map, range
except:
    from builtins import zip
    from builtins import str
    from builtins import map
    from builtins import range
from past.utils import old_div
from skimage import io
import tifffile

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import scipy
import h5py
import pandas as pd
from itertools import combinations

import cv2
try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.components_evaluation import evaluate_components_CNN
from caiman.motion_correction import motion_correct_iteration
from caiman.utils.stats import df_percentile
import bokeh.plotting as bpl

from skimage.feature import peak_local_max
from scipy.stats.mstats import zscore
from scipy import ndimage
import copy
from matplotlib import interactive
import sys, traceback
import imp
import shutil
interactive(True)

# utils
from utils_loading import get_all_animals, get_animal_days, encode_to_filename


def all_run(folder, animal, day, number_planes=4, number_planes_total=6, fresh=False):
    """
    Function to run all the different functions of the pipeline that gives back the analyzed data
    Folder (str): folder where the input/output is/will be stored
    animal/day (str) to be analyzed
    number_planes (int): number of planes that carry information
    number_planes_total (int): number of planes given back by the recording system, it may differ from number_planes
    to provide time for the objective to return to origen"""

    folder_path = folder + 'raw/' + animal + '/' + day + '/'
    folder_final = folder + 'processed/' + animal + '/' + day + '/'
    err_file = open(folder_path + "errlog.txt", 'a+')  # ERROR HANDLING
    if not os.path.exists(folder_final):
        os.makedirs(folder_final)

    finfo = folder_path +  'wmat.mat'  #file name of the mat
    matinfo = scipy.io.loadmat(finfo)

    ffull = [folder_path + lookup_with_default('fname', matinfo, folder)[0]]            # filename to be processed
    fbase = [folder_path + lookup_with_default('fbase', matinfo, folder)[0]]
    
    fbase1 = [folder + 'raw/' + animal + '/' + day + '/' + 'baseline_00001.tif']
    fbase2 = [folder + 'raw/' + animal + '/' + day + '/' + 'bmi_00001.tif']

    try:
        num_files, len_bmi = separate_planes(folder, animal, day, ffull, 'bmi', number_planes,
                                             number_planes_total)
        if os.path.exists(fbase2[0]):
            num_files_b, len_base = separate_planes_multiple_baseline(folder, animal, day, fbase1, fbase2,
                                                                      number_planes=number_planes)
        else:
            num_files_b, len_base = separate_planes(folder, animal, day, fbase, 'baseline', number_planes, number_planes_total)
    except Exception as e:
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit('Error in separate planes')

    if fresh:
        nam = folder_path + 'readme.txt'
        readme = open(nam, 'w+')
        readme.write("num_files_b = " + str(num_files_b) + '; \n')
        readme.write("num_files = " + str(num_files)+ '; \n')
        readme.write("len_base = " + str(len_base)+ '; \n')
        readme.write("len_bmi = " + str(len_bmi)+ '; \n')
        readme.close()

    try:
        analyze_raw_planes(folder, animal, day, num_files, num_files_b, number_planes, False)
    except Exception as e:
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit('Error in analyze raw')

    try:
        shutil.rmtree(folder + 'raw/' + animal + '/' + day + '/separated/')
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

# This section is done nowadays in another computer to minimize time for analysis
#     try:
#         put_together(folder, animal, day, len_base, len_bmi, number_planes, number_planes_total)
#     except Exception as e:
#         tb = sys.exc_info()[2]
#         err_file.write("\n{}\n".format(folder_path))
#         err_file.write("{}\n".format(str(e.args)))
#         traceback.print_tb(tb, file=err_file)
#         err_file.close()
#         sys.exit('Error in put together')

    err_file.close()
    

def separate_planes(folder, animal, day, ffull, var='bmi', number_planes=4, number_planes_total=6, order='F', lim_bf=9000):
    """
    Function to separate the different planes in the bigtiff file given by the recording system.
    TO BECOME OBSOLETE
    Folder (str): folder where the input/output is/will be stored
    animal/day (str) to be analyzed
    ffull (str): address of the file where the bigtiff is stored
    var(str): variable to specify if performing BMI or baseline
    number_planes(int): number of planes that carry information
    number_planes_total(int): number of planes given back by the recording system, it may differ from number_planes
    to provide time for the objective to return to origen
    order(str): order to stablish the memmap C/F
    lim_bf (int): limit of frames per split to avoid saving big tiff files"""
    
    
    # function to separate a layered TIFF (tif generated with different layers info)
    folder_path = folder + 'raw/' + animal + '/' + day + '/separated/'  
    fanal = folder + 'raw/' + animal + '/' + day + '/analysis/' 
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not os.path.exists(fanal):
        os.makedirs(fanal)
    
    err_file = open("errlog.txt", 'a+')  # ERROR HANDLING
      
    print('loading image...')
    ims = tifffile.TiffFile(ffull[0])
    LEN_IM = int(len(ims.pages)/number_planes_total)
    dims = [LEN_IM] + list(ims.pages[0].shape)
    print('Image loaded')
    
    num_files = int(np.ceil(LEN_IM/lim_bf))

    if hasattr(number_planes, '__iter__'):
        plane_iters = number_planes
    else:
        plane_iters = np.arange(number_planes)


    for plane in plane_iters:
        len_im = LEN_IM
        print ('length of tiff is: ' + str(len_im) + ' volumes')  
        for nf in np.arange(num_files):
            # create the mmap file
            fnamemm = folder_path + 'temp_plane_' + str(plane) + '_nf_' + str(nf) +  '.mmap'
            if len_im>lim_bf:
                auxlen = lim_bf
                len_im -= lim_bf
            else:
                auxlen = len_im
            big_file = np.memmap(fnamemm, mode='w+', dtype=np.int16, shape=(np.prod(dims[1:]), auxlen), order=order)
                
            # fill the mmap file
            for ind in np.arange(auxlen):
                new_img = ims.pages[int((ind + lim_bf*nf)*number_planes_total + plane)].asarray()
                big_file[:, ind] = np.reshape(new_img, np.prod(dims[1:]), order=order)
            
            #to plot the image before closing big_file (as a checkup that everything went smoothly)
            if not os.path.exists(fanal + str(plane) + '/'):
                os.makedirs(fanal + str(plane) + '/')
            imgtosave = np.transpose(np.reshape(np.nanmean(big_file,1), [dims[1],dims[2]])) # careful here if the big_file is too big to average
            plt.imshow(imgtosave)
            plt.savefig(fanal + str(plane) + '/' + 'nf' + str(nf) + '_rawmean.png', bbox_inches="tight")
            plt.close()
            
            big_file.flush()
            del big_file
    
    # clean memory    
    ims.close()

    
    # save the mmaps as tiff-files for caiman
    for plane in plane_iters:
        len_im = LEN_IM
        print ('saving a  tiff of: ' + str(len_im) + ' volumes') 
        for nf in np.arange(num_files):
            fnamemm = folder_path + 'temp_plane_' + str(plane) + '_nf_' + str(nf) +  '.mmap'
            fnametiff = folder_path + var + '_plane_' + str(plane) + '_nf_' + str(nf) + '.tiff'
            if len_im>lim_bf:
                auxlen = lim_bf
                len_im -= lim_bf
            else:
                auxlen = len_im

            big_file = np.memmap(fnamemm, mode='r', dtype=np.int16, shape=(np.prod(dims[1:]), auxlen), order=order)
            img_tosave = np.transpose(np.reshape(big_file, [dims[1], dims[2], int(auxlen)]))
            io.imsave(fnametiff, img_tosave, plugin='tifffile') # saves each plane different tiff
            del big_file
            del img_tosave
            try:
                os.remove(fnamemm)
            except OSError as e:  ## if failed, report it back to the user ##
                print ("Error: %s - %s." % (e.filename, e.strerror))
    
    len_im = LEN_IM
    del ims
    
    err_file.close()
    
    return num_files, len_im


def separate_planes_multiple_baseline(folder, animal, day, fbase1, fbase2, var='baseline', number_planes=4, number_planes_total=6, order='F', lim_bf=9000):
    """
    Function to separate the different planes in the bigtiff file given by the recording system WHEN there is more than one baseline file.
    TO BECOME OBSOLETE
    Folder (str): folder where the input/output is/will be stored
    animal/day (str) to be analyzed
    ffull (str): address of the file where the bigtiff is stored
    ffull2 (str): address of the  file where the consecutive bigtiff is stored
    var(str): variable to specify if performing BMI or baseline
    number_planes(int): number of planes that carry information
    number_planes_total(int): number of planes given back by the recording system, it may differ from number_planes
    to provide time for the objective to return to origen
    order(str): order to stablish the memmap C/F
    lim_bf (int): limit of frames per split to avoid saving big tiff files"""

    
    # function to separate a layered TIFF (tif generated with different layers info)
    folder_path = folder + 'raw/' + animal + '/' + day + '/separated/'       
    fanal = folder + 'raw/' + animal + '/' + day + '/analysis/' 
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not os.path.exists(fanal):
        os.makedirs(fanal)
        
    err_file = open("errlog.txt", 'a+')  # ERROR HANDLING
    
    # load the big file we will need to separate
    print('loading image...')
    imb1 = tifffile.TiffFile(fbase1[0])
    imb2 = tifffile.TiffFile(fbase2[0])
    print('Images loaded')
    len_imb1 = int(len(imb1.pages)/number_planes_total)
    len_imb2 = int(len(imb2.pages)/number_planes_total)
    dims1 = [len_imb1] + list(imb1.pages[0].shape)  
    dims2 = [len_imb2] + list(imb2.pages[0].shape)  
    
    dims = dims1 + np.asarray([dims2[0], 0, 0])
    len_im = copy.deepcopy(int(dims[0]))
    num_files = int(np.ceil(len_im/lim_bf))

    if hasattr(number_planes, '__iter__'):
        plane_iters = number_planes
    else:
        plane_iters = np.arange(number_planes)
    
    for plane in plane_iters:
        first_images_left = int(dims1[0])
        len_im = int(dims[0])
        print ('length of tiff is: ' + str(len_im) + ' volumes')  
        for nf in np.arange(num_files):
            # create the mmap file
            fnamemm = folder_path + 'temp_plane_' + str(plane) + '_nf_' + str(nf) +  '.mmap'
            if len_im>lim_bf:
                auxlen = lim_bf
                len_im -= lim_bf
            else:
                auxlen = len_im
            big_file = np.memmap(fnamemm, mode='w+', dtype=np.int16, shape=(np.prod(dims[1:]), auxlen), order=order)
                
            # fill the mmap file
            for ind in np.arange(auxlen):
                if first_images_left > 0:
                    new_img = imb1.pages[int(plane+(ind + lim_bf*nf)*number_planes_total)].asarray()
                    first_images_left -= 1
                else:
                    new_img = imb2.pages[int(plane+(ind - int(dims1[0]) + lim_bf*nf)*number_planes_total)].asarray()
                    
                big_file[:, ind] = np.reshape(new_img, np.prod(dims[1:]), order=order)
                
            #to plot the image before closing big_file (as a checkup that everything went smoothly)
            if not os.path.exists(fanal + str(plane) + '/'):
                os.makedirs(fanal + str(plane) + '/')
            imgtosave = np.transpose(np.reshape(np.nanmean(big_file,1), [dims[1],dims[2]]))
            plt.imshow(imgtosave)
            plt.savefig(fanal + str(plane) + '/' + 'nf' + str(nf) + '_rawmean.png', bbox_inches="tight")
            plt.close()
            
            big_file.flush()
            del big_file
    
    # save the mmaps as tiff-files for caiman
    for plane in plane_iters:
        len_im = int(dims[0])
        print ('saving a  tiff of: ' + str(len_im) + ' volumes') 
        for nf in np.arange(num_files):
            fnamemm = folder_path + 'temp_plane_' + str(plane) + '_nf_' + str(nf) +  '.mmap'
            fnametiff = folder_path + var + '_plane_' + str(plane) + '_nf_' + str(nf) + '.tiff'
            if len_im>lim_bf:
                auxlen = lim_bf
                len_im -= lim_bf
            else:
                auxlen = len_im

            big_file = np.memmap(fnamemm, mode='r', dtype=np.int16, shape=(np.prod(dims[1:]), auxlen), order=order)
            img_tosave = np.transpose(np.reshape(big_file, [dims[1], dims[2], int(auxlen)]))
            io.imsave(fnametiff, img_tosave, plugin='tifffile') # saves each plane different tiff
            del big_file
            del img_tosave
            try:
                os.remove(fnamemm)
            except OSError as e:  ## if failed, report it back to the user ##
                print ("Error: %s - %s." % (e.filename, e.strerror))
    
    # clean memory    
    imb1.close()
    imb2.close()
    
    del imb1
    del imb2 
    
    return num_files, int(dims[0])


def lookup_with_default(k, matinfo, folder):
    if k not in matinfo:
        default = scipy.io.loadmat(folder + '/raw/PT19/190801/wmat.mat')
        print('DEFAAAAAAUUUUUUUULT')
        return default[k]
    else:
        return matinfo[k]


def analyze_raw_planes(folder, animal, day, num_files, num_files_b, number_planes=4, dend=False, display_images=True):
    """
    Function to analyze every plane and get the result in a hdf5 file. It uses caiman_main
    Folder(str): folder where the input/output is/will be stored
    animal/day(str) to be analyzed
    num_files(int): number of files for the bmi file
    num_files_b(int): number of files for the baseline file
    number_planes(int): number of planes that carry information
    number_planes_total(int): number of planes given back by the recording system, it may differ from number_planes
    to provide time for the objective to return to origen
    dend(bool): Boleean to change parameters to look for neurons or dendrites
    display_images(bool): to display and save different plots"""

    folder_path = folder + 'raw/' + animal + '/' + day + '/separated/'
    finfo = folder + 'raw/' + animal + '/' + day + '/wmat.mat'  #file name of the mat 
    matinfo = scipy.io.loadmat(finfo)
    initialZ = int(lookup_with_default('initialZ', matinfo, folder)[0][0])
    fr = lookup_with_default('fr', matinfo, folder)[0][0]
    
    if dend:
        sec_var = 'Dend'
    else:
        sec_var = ''
    
    print('*************Starting with analysis*************')
    neuron_mats = []

    if hasattr(number_planes, '__iter__'):
        plane_iters = number_planes
    else:
        plane_iters = np.arange(number_planes)

    for plane in plane_iters:
        dff_all = []
        neuron_act_all = []
        fnames = []
        for nf in np.arange(int(num_files_b)):
            fnames.append(folder_path + 'baseline' + '_plane_' + str(plane) + '_nf_' + str(nf) + '.tiff')
        print('performing plane: ' + str(plane))
        for nf in np.arange(int(num_files)):
            fnames.append(folder_path + 'bmi' + '_plane_' + str(plane) + '_nf_' + str(nf) + '.tiff')
            
        fpath = folder + 'raw/' + animal + '/' + day + '/analysis/' + str(plane) + '/'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
            
        try:
            f = h5py.File(folder + 'raw/' + animal + '/' + day + '/' + 'bmi_' + sec_var + '_' + str(plane) + '.hdf5', 'w-')
        except IOError:
            print(" OOPS!: The file already existed ease try with another file, new results will NOT be saved")
            continue 
            
        zval = calculate_zvalues(folder, plane)
        print(fnames)
        dff, com, cnm2, totdes, SNR = caiman_main(fpath, fr, fnames, zval, dend, display_images)
        print ('Caiman done: saving ... plane: ' + str(plane) + ' file: ' + str(nf)) 
        
        Asparse = scipy.sparse.csr_matrix(cnm2.estimates.A)
        f.create_dataset('dff', data = dff)                   #activity
        f.create_dataset('com', data = com)                         #distance
        g = f.create_group('Nsparse')                               #neuron shape
        g.create_dataset('data', data = Asparse.data)
        g.create_dataset('indptr', data = Asparse.indptr)
        g.create_dataset('indices', data = Asparse.indices)
        g.attrs['shape'] = Asparse.shape
        f.create_dataset('neuron_act', data = cnm2.estimates.S)          #spikes
        f.create_dataset('C', data = cnm2.estimates.C)                   #temporal activity
        f.create_dataset('base_im', data = cnm2.estimates.b)                 #baseline image
        f.create_dataset('tot_des', data = totdes)                      #total displacement during motion correction
        f.create_dataset('SNR', data = SNR)                             #SNR of neurons
        f.close()  
        
    print('... done') 
     
    
def put_together(folder, animal, day, number_planes=4, number_planes_total=6, sec_var='', toplot=False, trial_time=30, tocut=False, len_experiment=30000, bmi2=False):       
    """
    Function to put together the different hdf5 files obtain for each plane and convey all the information in one and only hdf5
    it requires somo files in the original folder
    Folder(str): folder where the input/output is/will be stored
    animal/day(str) to be analyzed
    len_base(int): length of the baseline file (in frames)
    len_bmi(int): length of the bmi file (in frames)
    number_planes(int): number of planes that carry information
    number_planes_total(int): number of planes given back by the recording system, it may differ from number_planes
    to provide time for the objective to return to origen
    sec_var(str): secondary variable to save file. For extra information
    toplot(bool): to allow plotting/saving of some results"""
    
    # Folder to load/save
    folder_path = folder + 'raw/' + animal + '/' + day + '/'
    folder_dest = folder + 'processed/' + animal + '/'
    fanal = folder_path + 'analysis/'
    if not os.path.exists(folder_dest):
        os.makedirs(folder_dest)
    if not os.path.exists(fanal):
        os.makedirs(fanal)
    
    # Load information
    print ('loading info')
    vars = imp.load_source('readme', folder_path + 'readme.txt') 
    finfo = folder_path +  'wmat.mat'  #file name of the mat 
    matinfo = scipy.io.loadmat(finfo)
    ffull = [folder_path + lookup_with_default('fname', matinfo, folder)[0]]
    metadata = tifffile.TiffFile(ffull[0]).scanimage_metadata
    fr = lookup_with_default('fr', matinfo, folder)[0][0]   
    folder_red = folder + 'raw/' + animal + '/' + day + '/'
    fmat = folder_red + 'red.mat' 
    redinfo = scipy.io.loadmat(fmat)
    red = redinfo['red'][0]
    com_list = []
    neuron_plane = np.zeros(number_planes)
    tot_des_plane = np.zeros((number_planes, 2))
    for plane in np.arange(number_planes):
        try:
            f = h5py.File(folder_path + 'bmi_' + sec_var + '_' + str(plane) + '.hdf5', 'r')
        except OSError:
            break
        auxb = np.nansum(np.asarray(f['base_im']),1)
        bdim = int(np.sqrt(auxb.shape[0]))  
        base_im = np.transpose(np.reshape(auxb, [bdim,bdim])) 
        fred = folder_path + 'red' + str(plane) + '.tif'
        red_im = tifffile.imread(fred)
        auxdff = np.asarray(f['dff'])
        auxC = np.asarray(f['C'])
        neuron_plane[plane] = auxC.shape[0]
        tot_des_plane[plane,:] = np.asarray(f['tot_des'])
        if np.nansum(auxdff) == 0:
            auxdff = auxC * np.nan
 
        if plane == 0:
            all_dff = auxdff
            all_C = np.asarray(f['C'])
            all_com = np.asarray(f['com'])
            com_list.append(np.asarray(f['com']))
            g = f['Nsparse']
            all_neuron_shape = scipy.sparse.csr_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), g.attrs['shape'])
            all_base_im = np.ones((base_im.shape[0], base_im.shape[1], number_planes)) *np.nan
            all_neuron_act = np.asarray(f['neuron_act'])
            all_red_im = np.ones((red_im.shape[0], red_im.shape[1], number_planes)) *np.nan 
            all_SNR = np.asarray(f['SNR'])
        else:
            all_dff = np.concatenate((all_dff, auxdff), 0)
            all_C = np.concatenate((all_C, np.asarray(f['C'])), 0)
            all_com = np.concatenate((all_com, np.asarray(f['com'])), 0)
            com_list.append(np.asarray(f['com']))
            g = f['Nsparse']
            gaux = scipy.sparse.csr_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), g.attrs['shape'])
            all_neuron_shape = scipy.sparse.hstack([all_neuron_shape, gaux])
            all_neuron_act = np.concatenate((all_neuron_act, np.asarray(f['neuron_act'])), 0)
            all_SNR = np.concatenate((all_SNR, np.asarray(f['SNR'])),0)
        all_red_im[:, :, plane] = red_im
        all_base_im[:, :, plane] = base_im
        f.close()
    
    print ('success!!')
            
    auxZ = np.zeros((all_com.shape))
    auxZ[:,2] = np.repeat(lookup_with_default('initialZ', matinfo, folder)[0][0],all_com.shape[0])
    all_com += auxZ
    
    # Reorganize sparse matrix of spatial components
    dims = all_neuron_shape.shape  
    dims = [int(np.sqrt(dims[0])), int(np.sqrt(dims[0])), all_neuron_shape.shape[1]]
    Asparse = scipy.sparse.csr_matrix(all_neuron_shape)
    Afull = np.reshape(all_neuron_shape.toarray(),dims)
    
    # separates "real" neurons from dendrites
    print ('finding neurons')
    pred, _ = evaluate_components_CNN(all_neuron_shape, dims[:2], [4,4])
    nerden = np.zeros(Afull.shape[2]).astype('bool')
    nerden[np.where(pred[:,1]>0.75)] = True
    
    # obtain the real position of components A
    new_com = obtain_real_com(fanal, Afull, all_com, nerden, toplot)
    
    # sanity check of the neuron's quality
    if toplot:
        plot_Cs(fanal, all_C, nerden)
    
    print('success!!')
    
    # identify ens_neur (it already plots sanity check in raw/analysis
    
    
    # for those experiments which had 2 BMIs files and didn't get attached correctly
    if bmi2:
        online_data0 = pd.read_csv(folder_path + 'bmi_IntegrationRois_00000.csv')
        online_data1 = pd.read_csv(folder_path + lookup_with_default('fcsv', matinfo, folder)[0])
        last_ts = np.asarray(online_data0['timestamp'])[-1]
        last_frame = np.asarray(online_data0['frameNumber'])[-1]
        online_data1['timestamp'] += last_ts
        online_data1['frameNumber'] += last_frame
        online_data = pd.concat([online_data0, online_data1])
        vars.len_base = 9000
        vars.len_bmi += np.round(last_frame/number_planes_total).astype(int)
    else:
        online_data = pd.read_csv(folder_path + lookup_with_default('fcsv', matinfo, folder)[0])
        
    try:
        mask = matinfo['allmask']
    except KeyError:
        mask = np.nan
            
    
    print('finding ensemble neurons')
    
    ens_neur = detect_ensemble_neurons(fanal, all_dff, online_data, len(online_data.keys())-2,
                                             new_com, metadata, neuron_plane, number_planes_total, vars.len_base)
    
    auxens_neur = ens_neur[~np.isnan(ens_neur)]
    nerden[auxens_neur.astype('int')] = True     
    
    
    # obtain trials hits and miss
    trial_end, trial_start, array_t1, array_miss, hits, miss = check_trials(matinfo, vars, fr, trial_time)
    if np.sum(np.isnan(trial_end)) != 0:
        print ("STOPPING nan's found in the trial_end")
        return
    
    print('finding red neurons')
    
    # obtain the neurons label as red (controlling for dendrites)
    redlabel = red_channel(red, neuron_plane, nerden, Afull, new_com, all_red_im, all_base_im, fanal, number_planes, toplot=toplot)
    redlabel[auxens_neur.astype('int')] = True   
    
    
    # obtain the frequency
    try:
        frequency = obtainfreq(matinfo['frequency'][0], vars.len_bmi)
    except KeyError:
        frequency = np.nan
    
    cursor = matinfo['cursor'][0]
    
    # finding the correct E2 neurons
    e2_neur = get_best_e2_combo(ens_neur, online_data, cursor, trial_start, trial_end, vars.len_base)
    
    online_data = np.asarray(online_data)
    if tocut:
        all_C, all_dff, all_neuron_act, trial_end, trial_start, hits, miss, array_t1, array_miss, cursor, frequency, online_data = \
        cut_experiment(all_C, all_dff, all_neuron_act, trial_end, trial_start, hits, miss, cursor, frequency, vars.len_base, len_experiment, online_data)
    
    # sanity checks
    if toplot:
        plt.figure()
        plt.plot(np.nanmean(all_C,0)/10000)
        plt.title('Cs')
        plt.savefig(fanal + animal + '_' + day + '_Cs.png', bbox_inches="tight")
        plt.figure()
        plt.plot(matinfo['cursor'][0])
        plt.title('cursor')
        plt.savefig(fanal + animal + '_' + day + '_cursor.png', bbox_inches="tight")
    plt.figure()
    plt.plot(np.nanmean(all_dff,0))
    plt.title('dFFs')
    plt.savefig(fanal + animal + '_' + day + '_dffs.png', bbox_inches="tight")
   
    plt.close('all')


    # does fr exist?
    fr = lookup_with_default('fr', matinfo, folder)[0][0]

    #fill the file with all the correct data!
    try:
        fall = h5py.File(folder_dest + 'full_' + animal + '_' + day + '_' + sec_var + '_data.hdf5', 'w-')

        print('saviiiiiing')     
        fall.create_dataset('dff', data = all_dff) # (array) (Ft - Fo)/Fo . Increment of fluorescence
        fall.create_dataset('C', data = all_C)  # (array) Relative fluorescence of each component
        fall.create_dataset('SNR', data = all_SNR)  # (array) Signal to noise ratio of each component
        fall.create_dataset('com_cm', data = all_com) # (array) Position of the components as given by caiman 
        fall.create_dataset('com', data = new_com) # (array) Position of the components as calculated in pipeline (better approx) 
        fall.attrs['blen'] = vars.len_base # (int) lenght of the baseline
        gall = fall.create_group('Nsparse') # (sparse matrix) spatial filter of each component
        gall.create_dataset('data', data = Asparse.data) # (part of the sparse matrix)
        gall.create_dataset('indptr', data = Asparse.indptr) # (part of the sparse matrix)
        gall.create_dataset('indices', data = Asparse.indices) # (part of the sparse matrix)
        gall.attrs['shape'] = Asparse.shape # (part of the sparse matrix)
        fall.create_dataset('neuron_act', data = all_neuron_act) # (array) Spike activity (S in caiman)
        fall.create_dataset('base_im', data = all_base_im) # (array) matrix with all the average image of the baseline for each plane
        fall.create_dataset('red_im', data = all_red_im) # (array) matrix with all the imagesfrom the red chanel for each plane
        fall.create_dataset('online_data', data = online_data) # (array) Online recordings of the BMI
        fall.create_dataset('ens_neur', data = ens_neur) # (array) Index of the ensemble neurons among the rest of components
        fall.create_dataset('e2_neur', data = e2_neur) # (array) Index of the E2 neurons among the rest of components
        fall.create_dataset('trial_end', data = trial_end) # (array) When a trial ended. Can be a hit or a miss
        fall.create_dataset('trial_start', data = trial_start) # (array) When a trial started
        fall.attrs['fr'] =  fr # (int) Framerate
        fall.create_dataset('redlabel', data = redlabel) # (array-bool) True labels neurons as red
        fall.create_dataset('nerden', data = nerden) # (array-bool) True labels components as neurons
        fall.create_dataset('hits', data = hits) # (array) When the animal hit the target 
        fall.create_dataset('miss', data = miss) # (array) When the animal miss the target
        fall.create_dataset('array_t1', data = array_t1) # (array) index of the trials that ended in hit
        fall.create_dataset('array_miss', data = array_miss) # (array) Index of the trials that ended in miss
        fall.create_dataset('cursor', data = cursor) # (array) Online cursor of the BMI
        fall.create_dataset('freq', data = frequency) # (array) Frenquency resulting of the online cursor.
        
        fall.close()
        print('all done!!')
        
    except IOError:
        print(" OOPS!: The file already existed please try with another file, no results will be saved!!!")


def check_trials(matinfo, vars, fr, trial_time=30):
    trial_end = (np.unique(matinfo['trialEnd'][0]) + vars.len_base).astype('int')
    trial_start = (np.unique(matinfo['trialStart'][0]) + vars.len_base).astype('int')
    if len(matinfo['hits']) > 0 : 
        hits = (matinfo['hits'][0] + vars.len_base).astype('float')
    else:
        hits = []
    if len(matinfo['miss']) > 0 : 
        miss = (matinfo['miss'][0] + vars.len_base).astype('float')
    else:
        miss = []
    # to remove false end of trials
    if trial_start[0] > trial_end[0]:
        trial_end = trial_end[1:]
    if trial_start.shape[0] > trial_end.shape[0]:
        trial_start = trial_start[:-1]
    flag_correct = False
    if (trial_end.shape[0] > trial_start.shape[0]): flag_correct = True
    if not flag_correct:
        if (len(np.where((trial_end - trial_start)<0)[0])>0): flag_correct = True
    if flag_correct:
        ind = 0
        while ind < trial_start.shape[0]: # CAREFUL it can get stack foreveeeeeer
            tokeep = np.ones(trial_end.shape[0]).astype('bool')
            if trial_end.shape[0] < trial_start.shape[0]:
                trial_start = trial_start[:trial_end.shape[0]]
            elif (trial_end[ind] - trial_start[ind]) < 0 :            
                hitloc = np.where(trial_end[ind]==hits)[0]
                misloc = np.where(trial_end[ind]==miss)[0]
                if len(hitloc) > 0:
                    hits[hitloc[0]] = np.nan
                if len(misloc) > 0:
                    miss[misloc[0]] = np.nan
                tokeep[ind]=False
                trial_end = trial_end[tokeep]
            else:
                ind += 1
        
        # to remove ending trials that may have occur without trial start at the end of experiment
        if trial_start.shape[0] != trial_end.shape[0]:
            todel = trial_end>trial_start[-1] 
            todel[np.where(todel)[0][0]] = False
            for tend in trial_end[todel]:
                hitloc = np.where(tend==hits)[0]
                misloc = np.where(tend==miss)[0]
                if len(hitloc) > 0:
                    hits[hitloc[0]] = np.nan
                if len(misloc) > 0:
                    miss[misloc[0]] = np.nan
            trial_end = trial_end[:trial_start.shape[0]]
        
        if np.sum((trial_end - trial_start) > trial_time*fr + 10) != 0:  # make sure that no trial is more than it should be +10 because it can vara bit
            print ("Something wrong happened here, you better check this one out")
            #return np.nan, np.nan, np.nan, np.nan
    
        # to remove trials that ended in the same frame as they started
        tokeep = np.ones(trial_start.shape[0]).astype('bool')
        for ind in  np.arange(trial_start.shape[0]):
            if (trial_end[ind] - trial_start[ind]) == 0 :
                tokeep[ind]=False
                hitloc = np.where(trial_end[ind]==hits)[0]
                misloc = np.where(trial_end[ind]==miss)[0]
                if len(hitloc) > 0:
                    hits[hitloc[0]] = np.nan
                if len(misloc) > 0:
                    miss[misloc[0]] = np.nan
        
        hits = hits[~np.isnan(hits)]
        miss = miss[~np.isnan(miss)]   
        trial_end = trial_end[tokeep]
        trial_start = trial_start[tokeep]
    elif trial_end.shape[0] < trial_start.shape[0]:
        trial_start = trial_start[:trial_end.shape[0]]
            
    # preparing the arrays (number of trial for hits/miss)
    array_t1 = np.zeros(hits.shape[0], dtype=int)
    array_miss = np.zeros(miss.shape[0], dtype=int)
    for hh, hit in enumerate(hits): array_t1[hh] = np.where(trial_end==hit)[0][0]
    for mm, mi in enumerate(miss): array_miss[mm] = np.where(trial_end==mi)[0][0]
    
    return trial_end, trial_start, array_t1, array_miss, hits, miss


def view_wmat(wmat):
    imp_fields = []
    for k in wmat.keys():
        if k.find('__') != -1:
            continue
        it = wmat[k]
        if not isinstance(it, np.ndarray):
            print(k, type(k))
            continue
        if np.prod(it.shape) == 1:
            if isinstance(it.item(), np.ndarray):
                print(k, it.item().shape if np.prod(it.item().shape) > 1 else it.item())
            else:
                print(k, type(it.item()), it.item())
        elif it.shape[1] > 1 and len(it.shape) == 2:
            print(k, it.shape)
            imp_fields.append((k, it.shape))
        else:
            print(k, it.shape)
    return imp_fields


def mat_compare(wmat1, wmat2):
    imp_eq_fields = []
    imp_ineq_fields = []

    def tuple_equal(t1, t2):
        bval = True
        for i in range(len(t1)):
            if isinstance(t1[i], np.ndarray):
                bval = np.array_equal(t1[i], t2[i])
            else:
                bval = t1[i] == t2[i]
            if not bval:
                return bval
        return bval
    for k in wmat1.keys():
        if k.find('__') != -1:
            continue
        it1 = wmat1[k]
        it2 = wmat2[k]
        if not isinstance(it1, np.ndarray):
            print(k, type(k))
            continue
        if np.prod(it1.shape) == 1:
            if isinstance(it1.item(), np.ndarray):
                boolval = np.array_equal(it1.item(), it2.item())
                print(k, (('i1', it1.item().shape, 'i2', it2.item().shape) if np.prod(it1.item().shape) > 1
                          else ('i1', it1.item(), 'i2', it2.item())), boolval)
            else:
                if isinstance(it1.item(), tuple):
                    boolval = tuple_equal(it1.item(), it2.item())
                else:
                    boolval = (it1.item() == it2.item())
                print(k, type(it1.item()), ('i1', it1.item(), 'i2', it2.item()), boolval)
        elif it1.shape[1] > 1 and len(it1.shape) == 2:
            boolval = np.array_equal(it1, it2)
            print(k, ('i1', it1.shape, 'i2', it2.shape), boolval)
        else:
            boolval = np.array_equal(it1, it2)
            print(k, ('i1', it1.shape, 'i2', it2.shape), boolval)
        if boolval:
            imp_eq_fields.append(k)
        else:
            imp_ineq_fields.append(k)
    return imp_eq_fields, imp_ineq_fields


def wmat_merge(wmat1f, wmat2f, norm_dur1=True, norm_dur2=True):
    folder = wmat1f[:wmat1f.find('wmat1.mat')]
    wmat1, wmat2 = scipy.io.loadmat(wmat1f), scipy.io.loadmat(wmat2f)
    mod1 = wmat1['cursor'].shape[1] % 100
    mod2 = wmat2['cursor'].shape[1] % 100
    wmat1['duration'] = wmat1['cursor'].shape[1]
    wmat2['duration'] = wmat2['cursor'].shape[1]
    if norm_dur1 and mod1:
        wmat1['duration'] = wmat1['cursor'].shape[1] - mod1
    if norm_dur2 and mod2:
        wmat2['duration'] = wmat2['cursor'].shape[1] - mod2
    wmat = {}
    for k in wmat2.keys():
        if k in ('hits', 'miss', 'trialEnd'):
            wmat[k] = np.concatenate((wmat1[k], wmat2[k]+wmat1['duration']), axis=1)
        elif k == 'trialStart':
            if wmat1[k].shape[1] > wmat1['trialEnd'].shape[1]:
                s1 = wmat1[k][:, :-1]
            wmat[k] = np.concatenate((s1, wmat2[k]+wmat1['duration']), axis=1)
        elif k in ('cursor', 'frequency'):
            wmat[k] = np.concatenate((wmat1[k][:, :wmat1['duration']],
                                      wmat2[k][:, :wmat2['duration']]), axis=1)
        else:
            wmat[k] = wmat2[k]
    scipy.io.savemat(os.path.join(folder, 'wmat.mat'), wmat)
    return wmat


def red_channel(red, neuron_plane, nerden, Afull, new_com, all_red_im, all_base_im, fanal, number_planes=4, maxdist=4, toplot=True):  
    """
    Function to identify red neurons with components returned by caiman
    red(array-int): mask of red neurons position for each frame
    nerden(array-bool): array of bool labelling as true components identified as neurons.
    neuron_plane:  list number_of neurons for each plane
    new_com(array): position of the neurons
    all_red_im(array): matrix MxNxplanes: Image of the red channel 
    all_base_im(array): matrix MxNxplanes: Image of the green channel 
    fanal(str): folder where to store the analysis sanity check
    number_planes(int): number of planes that carry information
    maxdist(int): spatial tolerance to assign red label to a caiman component
    returns
    redlabel(array-bool): boolean vector labelling as True the components that are red neurons 
    """
    #function to identify red neurons
    all_red = []
    ind_neur = 0
    if len(red) < number_planes:
        number_planes = len(red)
    for plane in np.arange(number_planes):
        maskred = copy.deepcopy(np.transpose(red[plane]))
        mm = np.sum(maskred,1)
        maskred = maskred[~np.isnan(mm),:].astype('float32')
        
        # for some reason the motion correction sometimes works one way but not the other
        _, _, shift, _ = motion_correct_iteration(all_base_im[:,:,plane].astype('float32'), all_red_im[:,:,plane].astype('float32'),1)
        
        if np.nansum(abs(np.asarray(shift))) < 20:  # hopefully this motion correction worked
            maskred[:,0] -= shift[1].astype('float32')
            maskred[:,1] -= shift[0].astype('float32')
            # creates a new image with the shifts found
            M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
            min_, max_ = np.min(all_red_im[:,:,plane]), np.max(all_red_im[:,:,plane])
            new_img = np.clip(cv2.warpAffine(all_red_im[:,:,plane], M, (all_red_im.shape[0], all_red_im.shape[1]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), min_, max_)

        else:
            print ('Trying other way since shift was: ' + str(shift))
            # do the motion correctio the other way arround
            new_img, _, shift, _ = motion_correct_iteration(all_red_im[:,:,plane].astype('float32'), all_base_im[:,:,plane].astype('float32'),1)
            if np.nansum(abs(np.asarray(shift))) < 20:
                maskred[:,0] += shift[1].astype('float32')
                maskred[:,1] += shift[0].astype('float32')
            else:
                print ('didnt work with shift: ' + str(shift))
                new_img = all_red_im[:,:,plane]
                #somehow it didn't work either way
                print ('There was an issue with the motion correction during red_channel comparison. Please check plane: ' + str(plane))
            
        
        # find distances
        
        neur_plane = neuron_plane[plane].astype('int')
        aux_nc = np.zeros(neur_plane)
        aux_nc = new_com[ind_neur:neur_plane+ind_neur, :2]
        aux_nerden = nerden[ind_neur:neur_plane+ind_neur]
        dists = np.zeros((neur_plane,maskred.shape[0]))
        dists = scipy.spatial.distance.cdist(aux_nc, maskred)
        
        # identfify neurons based on distance
        redlabel = np.zeros(neur_plane).astype('bool')
        aux_redlabel = np.zeros(neur_plane)
        iden_pairs = []  # to debug
        for neur in np.arange(neur_plane):
            if aux_nerden[neur]:
                aux_redlabel[neur] = np.sum(dists[neur,:]<maxdist)
                redn = np.where(dists[neur,:]<maxdist)[0]
                if len(redn):
                    iden_pairs.append([neur, redn[0]])  # to debug
        redlabel[aux_redlabel>0]=True
        all_red.append(redlabel)
        auxtoplot = aux_nc[redlabel,:]

        if toplot:
            imgtoplot = np.zeros((new_img.shape[0], new_img.shape[1]))
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(1,2,1)
            for ind in np.arange(maskred.shape[0]):
                auxlocx = maskred[ind,1].astype('int')
                auxlocy = maskred[ind,0].astype('int')
                imgtoplot[auxlocx-1:auxlocx+1,auxlocy-1:auxlocy+1] = np.nanmax(new_img)
            ax1.imshow(new_img + imgtoplot, vmax=np.nanmax(new_img))
            
            imgtoplot = np.zeros((new_img.shape[0], new_img.shape[1]))
            ax2 = fig1.add_subplot(1,2,2)
            for ind in np.arange(auxtoplot.shape[0]):
                auxlocx = auxtoplot[ind,1].astype('int')
                auxlocy = auxtoplot[ind,0].astype('int')
                imgtoplot[auxlocx-1:auxlocx+1,auxlocy-1:auxlocy+1] = np.nanmax(new_img)
            ax2.imshow(new_img + imgtoplot, vmax=np.nanmax(new_img))
            plt.savefig(fanal + str(plane) + '/redneurmask.png', bbox_inches="tight")
            
            fig2 = plt.figure()
            R = new_img
            auxA = np.unique(np.arange(Afull.shape[2])[ind_neur:neur_plane+ind_neur]*redlabel)
            G = np.transpose(np.nansum(Afull[:,:,auxA],2))
            B = np.zeros((R.shape))
            R = R/np.nanmax(R)
            G = G/np.nanmax(G)
            
            RGB =  np.dstack((R,G,B))
            plt.imshow(RGB)
            plt.savefig(fanal + str(plane) + '/redneurmask_RG.png', bbox_inches="tight")
            plt.close("all") 
        
        ind_neur += neur_plane
        
    all_red = np.concatenate(all_red)
    return all_red


def obtain_real_com(fanal, Afull, all_com, nerden, toplot=True, img_size = 20, thres=0.1):
    """
    Function to obtain the "real" position of the neuron regarding the spatial filter
    fanal(str): folder where the plots will be stored
    Afull(array): matrix with all the spatial components
    all_com(array): matrix with the position in xyz given by caiman
    nerden(array-bool):  array of bool labelling as true components identified as neurons
    toplot(bool): flag to plot and save
    thres(int): tolerance to identify the soma of the spatial filter
    minsize(int): minimum size of a neuron. Should be change for types of neurons / zoom / spatial resolution
    Returns
    new_com(array): matrix with new position of the neurons
    """
    #function to obtain the real values of com
    faplot = fanal + 'Aplot/'
    if not os.path.exists(faplot):
        os.makedirs(faplot)
    new_com = np.zeros((Afull.shape[2], 3))
    for neur in np.arange(Afull.shape[2]):
        center_mass = scipy.ndimage.measurements.center_of_mass(Afull[:,:,neur]>thres)
        if np.nansum(center_mass)==0 :
            center_mass = scipy.ndimage.measurements.center_of_mass(Afull[:,:,neur])
        new_com[neur,:] = [center_mass[0], center_mass[1], all_com[neur,2]]
        if (center_mass[0] + img_size) > Afull.shape[0]:
            x2 = Afull.shape[0]
        else:
            x2 = int(center_mass[0]+img_size)
        if (center_mass[0] - img_size) < 0:
            x1 = 0
        else:
            x1 = int(center_mass[0]-img_size)
        if (center_mass[1] + img_size) > Afull.shape[1]:
            y2 = Afull.shape[1]
        else:
            y2 = int(center_mass[1]+img_size)
        if (center_mass[1] - img_size) < 0:
            y1 = 0
        else:
            y1 = int(center_mass[1]-img_size)
            
        if toplot:
            img = Afull[x1:x2,y1:y2,neur]
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(121)
            ax1.imshow(np.transpose(img))
            ax1.set_xlabel('nd: ' + str(nerden[neur]))
            ax2 = fig1.add_subplot(122)
            ax2.imshow(np.transpose(img>thres))
            ax2.set_xlabel('neuron: ' + str(neur))
            plt.savefig(faplot + str(neur) + '.png', bbox_inches="tight")
            plt.close('all')
        
    return new_com
        
                
def detect_ensemble_neurons(fanal, all_dff, online_data, units, com, metadata, neuron_plane, number_planes_total, len_base, auxtol=10, cormin=0.5):
    """
    Function to identify the ensemble neurons across all components
    fanal(str): folder where the plots will be stored
    dff(array): Dff values of the components. given by caiman
    online_data(array): activity of the ensemble neurons registered on the online bmi experiment
    units (int): number of neurons in the ensembles
    com(array): position of the neurons
    mask(array): position of the ensemble neurons as given by the experiment
    number_planes_total(int): number of planes given back by the recording system, it may differ from number_planes
    len_base(int): lenght of the baseline
    auxtol (int): max difference distance for ensemble neurons
    cormin (int): minimum correlation between neuronal activity from caiman DFF and online recording
    returns
    final_neur(array): index of the ensemble neurons"""
    
    # initialize vars
    neurcor = np.ones((units, all_dff.shape[0])) * np.nan
    finalcorr = np.zeros(units)
    finalneur = np.zeros(units)
    finaldist = np.zeros(units)
    pmask = np.zeros((metadata['FrameData']['SI.hRoiManager.pixelsPerLine'], metadata['FrameData']['SI.hRoiManager.linesPerFrame']))
    iter = 40
    
    ind_neuron_plane = np.cumsum(neuron_plane).astype('int')
    
    #extract reference from metadata
    a = metadata['RoiGroups']['imagingRoiGroup']['rois']['scanfields']['pixelToRefTransform'][0][0]
    b = metadata['RoiGroups']['imagingRoiGroup']['rois']['scanfields']['pixelToRefTransform'][0][2]
    all_zs = metadata['FrameData']['SI.hStackManager.zs']  
    
    
    for un in np.arange(units):
        print(['finding neuron: ' + str(un)])
        
        tol = copy.deepcopy(auxtol)
        tempcormin = copy.deepcopy(cormin)
        
        for npro in np.arange(all_dff.shape[0]):
            ens = (online_data.keys())[2+un]
            frames = (np.asarray(online_data['frameNumber']) / number_planes_total).astype('int') + len_base 
            auxonline = (np.asarray(online_data[ens]) - np.nanmean(online_data[ens]))/np.nanmean(online_data[ens]) 
            auxdd = all_dff[npro,frames]
            neurcor[un, npro] = pd.DataFrame(np.transpose([auxdd[~np.isnan(auxonline)], auxonline[~np.isnan(auxonline)]])).corr()[0][1]
    
        auxneur = copy.deepcopy(neurcor)
        neurcor[neurcor<tempcormin] = np.nan    
        
        
        # extract position from metadata
        relativepos = metadata['RoiGroups']['integrationRoiGroup']['rois'][un]['scanfields']['centerXY']
        centermass =  np.reshape(((np.asarray(relativepos) - b)/a).astype('int'), [1,2])
        zs = metadata['RoiGroups']['integrationRoiGroup']['rois'][un]['zs']
        plane = all_zs.index(zs)
        if plane == 0:
            neurcor[un, ind_neuron_plane[0]:] = np.nan
        else:
            neurcor[un, :ind_neuron_plane[plane-1]] = np.nan
            neurcor[un, ind_neuron_plane[plane]:] = np.nan
        not_good_enough = True
        
        while not_good_enough:
            if np.nansum(neurcor[un,:]) != 0:
                if np.nansum(np.abs(neurcor[un, :])) > 0 :
                    maxcor = np.nanmax(neurcor[un, :])
                    indx = np.where(neurcor[un, :]==maxcor)[0][0]
                    auxcom = np.reshape(com[indx,:2], [1,2])
                    dist = scipy.spatial.distance.cdist(centermass, auxcom)[0][0]
                    not_good_enough =  dist > tol
    
                    finalcorr[un] = neurcor[un, indx]
                    finalneur[un] = indx
                    finaldist[un] = dist
                    neurcor[un, indx] = np.nan
                else:
                    print('Error couldnt find neuron' + str(un) + ' with this tolerance. Increasing tolerance')
                    if iter > 0:
                        neurcor = auxneur
                        tol *= 1.1
                        iter -= 1
                    else:
                        print ('wtf??')
#                         break
            elif tempcormin > 0:
                    print('Error couldnt find neuron' + str(un) + ' reducing minimum correlation')
                    neurcor = auxneur
                    tempcormin-= 0.1
                    neurcor[neurcor<tempcormin] = np.nan    
                    tol-= 2  #If reduced correlation reduce distance
                    not_good_enough = True
            else:
                print('No luck, finding neurons by distance')
                auxcom = com[:,:2]
                dist = scipy.spatial.distance.cdist(centermass, auxcom)[0]
                if plane == 0:
                    dist[ind_neuron_plane[0]:] = np.nan
                else:
                    dist[:ind_neuron_plane[plane-1]] = np.nan
                    dist[ind_neuron_plane[plane]:] = np.nan
                indx = np.where(dist==np.nanmin(dist))[0][0]
                finalcorr[un] = np.nan
                if np.nanmin(dist) < auxtol:
                    finaldist[un] = np.nanmin(dist)
                    finalneur[un] = indx
                else:
                    print ('where are my neurons??')
                    finalneur[un] = np.nan
                    finaldist[un] = np.nan
                not_good_enough = False
        print('tol value at: ', str(tol), 'correlation thres at: ', str(tempcormin))
        print('Correlated with value: ', str(finalcorr[un]), ' with a distance: ', str(finaldist[un]))

        if ~np.isnan(finalneur[un]):
            auxp = com[finalneur[un].astype(int),:2].astype(int)
            pmask[auxp[1], auxp[0]] = 2   #to detect
            pmask[centermass[0,1], centermass[0,0]] = 1   #to detect
    
    plt.figure()
    plt.imshow(pmask)
    plt.savefig(fanal + 'ens_masks.png', bbox_inches="tight")
    
    
    fig1 = plt.figure(figsize=(16,6))
    for un in np.arange(units): 
        ax = fig1.add_subplot(units, 1, un + 1)
        ens = (online_data.keys())[2+un]
        frames = (np.asarray(online_data['frameNumber']) / number_planes_total).astype('int') + len_base 
        auxonline = (np.asarray(online_data[ens]) - np.nanmean(online_data[ens]))/np.nanmean(online_data[ens])
        auxonline[np.isnan(auxonline)] = 0
        ax.plot(zscore(auxonline[-5000:]))
        if ~np.isnan(finalneur[un]):
            auxdd = all_dff[finalneur[un].astype('int'), frames] 
            ax.plot(zscore(auxdd[-5000:]))
        
    plt.savefig(fanal + 'ens_online_offline.png', bbox_inches="tight")

    return finalneur
 

def calculate_zvalues(folder, plane):
    """
    Function to obtain the position Z of each neuron depending of their position in Y
    Folder(str): folder where the input/output is/will be stored 
    plane(int): number of plane being calculated
    returns
    z (array): position in Z of components
    """

    finfo = folder + 'actuator.mat'  #file name of the mat 
    actuator = scipy.io.loadmat(finfo)
    options={ 0: 'yd1i',
              1: 'yd2i',
              2: 'yd3i',
              3: 'yd4i'}
    z = actuator[options[plane]][0]
    
    return z


def obtainfreq(origfreq, len_bmi=36000, iterat=2):
    """ Function to remove NANs from the frequency vector. First values will be 0
    origfreq(array): vector of original frequency recorded, full of nans
    len_bmi(int): lenght of the recording bmi
    iterat(int): maximum number of iterations to find consecutive nans
    returns
    freq(array): vector of frequencies without nans. Nans before the experiment are changed as 0,
    nans during the experiment are change to the previous frequency value"""
    freq = copy.deepcopy(origfreq)
    freq[:np.where(~np.isnan(origfreq))[0][0]] = 0
    if len_bmi<freq.shape[0]: freq = freq[:len_bmi]
    for it in np.arange(iterat):
        nanarray = np.where(np.isnan(freq))[0]
        for inan in nanarray[-1::-1]:
            freq[inan] = freq[inan-1]
    return freq


def plot_Cs(fanal, C, nerden):
    """
    Function to plot the temporal activity of caiman components. To serve as a sanity check
    fanal(str): folder where the plots will be saved
    C(array): matrix with the temporal activity of the components
    nerden(array-bool): array of bool labelling as true components identified as neurons. """
    
    #function to obtain the real values of com
    folder_path = fanal + '/Cplot/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for ind in np.arange(C.shape[0]):
        fig1 = plt.figure(figsize=(12,6))
        ax1 = fig1.add_subplot(121)
        ax1.plot(C[ind,:])
        ax2 = fig1.add_subplot(122)
        ax2.plot(C[ind,1000:2000])
        ax2.set_xlabel(str(nerden[ind]))
        fig1.savefig(folder_path + str(ind) + '.png', bbox_inches="tight")
        plt.close('all')


def cut_experiment(all_C, all_dff, all_neuron_act, trial_end, trial_start, hits, miss, cursor, frequency, len_base, len_experiment, online_data):
    """
    Function to remove part of the experiment that was compromised by quality of image.
    Input: All variable to change
    Returns: variable changed """
    
    print ('Removing part of experiment due to lack of image quality')
    frame = np.array(online_data[:, 1]).astype(np.int32) // 6
    len_online = np.where(frame<(len_experiment-len_base))[0][-1]
    online_data = online_data[:len_online,:]
    all_C = all_C [:,:len_experiment]
    all_dff = all_dff [:,:len_experiment]
    all_neuron_act = all_neuron_act [:,:len_experiment]
    trial_end = trial_end[:np.where(trial_end>len_experiment)[0][0]]
    trial_start = trial_start[:np.where(trial_start>len_experiment)[0][0]]
    if trial_start.shape[0] > trial_end.shape[0]:
        trial_start = trial_start[:-1]
    auxhit = np.where(hits>len_experiment)[0]
    auxmiss = np.where(miss>len_experiment)[0]
    if len(auxhit) != 0:
        hits = hits[:auxhit[0]]
    if len(auxmiss) != 0:
        miss = miss[:auxmiss[0]]
    cursor = cursor[:(len_experiment - len_base)]
    if np.nansum(frequency)>0:
        frequency = frequency[:(len_experiment - len_base)]
    array_t1 = np.zeros(hits.shape[0], dtype=int)
    array_miss = np.zeros(miss.shape[0], dtype=int)
    for hh, hit in enumerate(hits): array_t1[hh] = np.where(trial_end==hit)[0][0]
    for mm, mi in enumerate(miss): array_miss[mm] = np.where(trial_end==mi)[0][0]    
    
    return all_C, all_dff, all_neuron_act, trial_end, trial_start, hits, miss, array_t1, array_miss, cursor, frequency, online_data
    
   
def caiman_main(fpath, fr, fnames, z=0, dend=False, display_images=False):
    """
    Main function to compute the caiman algorithm. For more details see github and papers
    fpath(str): Folder where to store the plots
    fr(int): framerate
    fnames(list-str): list with the names of the files to be computed together
    z(array): vector with the values of z relative to y
    dend(bool): Boleean to change parameters to look for neurons or dendrites
    display_images(bool): to display and save different plots
    returns
    F_dff(array): array with the dff of the components
    com(array): matrix with the position values of the components as given by caiman
    cnm(struct): struct with different stimates and returns from caiman"""
    
    # parameters
    decay_time = 0.4                    # length of a typical transient in seconds    

    # Look for the best parameters for this 2p system and never change them again :)
    # motion correction parameters
    niter_rig = 1               # number of iterations for rigid motion correction
    max_shifts = (3, 3)         # maximum allow rigid shift
    splits_rig = 10            # for parallelization split the movies in  num_splits chuncks across time
    strides = (96, 96)          # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (48, 48)         # overlap between pathes (size of patch strides+overlaps)
    splits_els = 10             # for parallelization split the movies in  num_splits chuncks across time
    upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
    max_deviation_rigid = 3     # maximum deviation allowed for patch with respect to rigid shifts
    
    # parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thresh = 0.8          # merging threshold, max correlation allowed
    rf = 25                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 10            # amount of overlap between the patches in pixels
    K = 25                       # number of components per patch
    
    if dend:
        gSig = [1, 1]               # expected half size of neurons
        init_method = 'sparse_nmf'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf =  1e-6          # sparsity penalty for dendritic data analysis through sparse NMF
    else:
        gSig = [3, 3]               # expected half size of neurons
        init_method = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf = None           # sparsity penalty for dendritic data analysis through sparse NMF
    
    # parameters for component evaluation
    min_SNR = 2.5               # signal to noise ratio for accepting a component
    rval_thr = 0.8              # space correlation threshold for accepting a component
    cnn_thr = 0.8               # threshold for CNN based classifier

    dview = None # parallel processing keeps crashing. 
    
    print('***************Starting motion correction*************')
    print('files:')
    print(fnames)
    
    
    # %% start a cluster for parallel processing
    #c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    
    #%%% MOTION CORRECTION
    # first we create a motion correction object with the parameters specified
    min_mov = cm.load(fnames[0]).min()
    # this will be subtracted from the movie to make it non-negative
    
    mc = MotionCorrect(fnames, min_mov,
                       dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
                       splits_rig=splits_rig,
                       strides=strides, overlaps=overlaps, splits_els=splits_els,
                       upsample_factor_grid=upsample_factor_grid,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=True, nonneg_movie=True)
    # note that the file is not loaded in memory
    
    #%% Run piecewise-rigid motion correction using NoRMCorre
    mc.motion_correct_pwrigid(save_movie=True)
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
                            
    totdes = [np.nansum(mc.x_shifts_els), np.nansum(mc.y_shifts_els)]
    print('***************Motion correction has ended*************')
    # maximum shift to be used for trimming against NaNs

    #%% MEMORY MAPPING
    # memory map the file in order 'C'
    fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C', border_to_0=bord_px_els)  # exclude borders
    
    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)
    
    
    # %% restart cluster to clean up memory
    #cm.stop_server(dview=dview)
    #c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        
    
    #%% RUN CNMF ON PATCHES
    print('***************Running CNMF...*************')
    
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    
    cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf,
                    only_init_patch=False, gnb=gnb, border_pix=bord_px_els)
    cnm = cnm.fit(images)
    
    
    #%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    
    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
        estimate_components_quality_auto(images, cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f,
                                         cnm.estimates.YrA, fr, decay_time, gSig, dims,
                                         dview=dview, min_SNR=min_SNR,
                                         r_values_min=rval_thr, use_cnn=False,
                                         thresh_cnn_min=cnn_thr)
    
     
    if display_images:
        plt.figure()
        plt.subplot(131)

        auxb = np.transpose(np.reshape(cnm.estimates.b[:,0], [int(np.sqrt(cnm.estimates.b.shape[0])), int(np.sqrt(cnm.estimates.b.shape[0]))]))
        plt.imshow(auxb)
        plt.title('Raw mean')
        plt.subplot(132)
        crd_good = cm.utils.visualization.plot_contours(
            cnm.estimates.A[:, idx_components], auxb, thr=.8)
        plt.title('Contour plots of accepted components')
        plt.subplot(133)
        crd_bad = cm.utils.visualization.plot_contours(
            cnm.estimates.A[:, idx_components_bad], auxb, thr=.8, vmax=0.2)
        plt.title('Contour plots of rejected components')
        plt.savefig(fpath + 'comp.png', bbox_inches="tight")
         
        plt.close('all')
    
    
    #%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    A_in, C_in, b_in, f_in = cnm.estimates.A[:, idx_components], cnm.estimates.C[idx_components], cnm.estimates.b, cnm.estimates.f
    cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                     merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                     f_in=f_in, rf=None, stride=None, gnb=gnb,
                     method_deconvolution='oasis', check_nan=True)
    
    print('***************Fit*************')
    cnm2 = cnm2.fit(images)
    
    print('***************Extractind DFFs*************')
    #%% Extract DF/F values
    
    #cm.stop_server(dview=dview)
    try:
        F_dff = detrend_df_f(cnm2.estimates.A, cnm2.estimates.b, cnm2.estimates.C, cnm2.estimates.f, YrA=cnm2.estimates.YrA, quantileMin=8, frames_window=250)
        #F_dff = detrend_df_f(cnm.A, cnm.b, cnm.C, cnm.f, YrA=cnm.YrA, quantileMin=8, frames_window=250)
    except:
        F_dff = cnm2.estimates.C * np.nan
        print ('WHAAT went wrong again?')
    
    print ('***************stopping cluster*************')
    #%% STOP CLUSTER and clean up log files
    #cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
           
        
    #***************************************************************************************    
    # Preparing output data
    # F_dff  -> DFF values,  is a matrix [number of neurons, length recording]
    
    # com  --> center of mass,  is a matrix [number of neurons, 2]
    print ('***************preparing output data*************')

    if len(dims)<=2:
        if len(z)==1:
            com = np.concatenate((cm.base.rois.com(cnm2.estimates.A,dims[0],dims[1]), np.zeros((cnm2.estimates.A.shape[1], 1))+z),1)
        elif len(z)==dims[0]:
            auxcom = cm.base.rois.com(cnm2.estimates.A,dims[0],dims[1])
            zy = np.zeros((auxcom.shape[0],1))
            for y in np.arange(auxcom.shape[0]):
                zy[y,0] = z[int(auxcom[y,0])]
            com = np.concatenate((auxcom, zy),1)
        else:
            print('WARNING: Z value was not correctly defined, only X and Y values on file, z==zeros')
            print(['length of z was: ' + str(len(z))])
            com = np.concatenate((cm.base.rois.com(cnm2.estimates.A,dims[0],dims[1]), np.zeros((cnm2.estimates.A.shape[1], 1))),1)
    else:
        com = cm.base.rois.com(cnm2.estimates.A,dims[0],dims[1], dims[2])        
        
    return F_dff, com, cnm2, totdes, SNR_comp[idx_components]


def get_best_e2_combo(ens_neur, online_data, cursor, trial_start, trial_end, len_base, number_planes_total=6):
    """
    Finds the most likely E2 pairing by simulating the cursor with different
    ensemble neuron pairings and finding the pairing with the highest correlation
    with the real cursor

    Args
        ens_neur: A (4,) numpy array containing the indices of the ensemble neurons.
            For instance, this array may look like [100, 452, 78, 94]
        online_data: A Pandas dataframe containing the neural activity of
            each activity over time. ENS_NEUR will index into this matrix.
        cursor: A (time,) numpy array containing the cursor activity. Assumed to be
            the same length in time as EXP_DATA
    Returns
        A (2,) numpy array containing the indices of the ensemble neurons of the
            most likely E2 pair. For instance, if ENS_NEUR is [100, 452, 78, 94],
            this function may return something like [78, 94]
    Raises
        ValueError: if cursor and exp_data are mismatched in time.
    """

    # Contains the keys for each ens_neur to index into the online data
    ens = (online_data.keys())[2:]
    frames = (np.asarray(online_data['frameNumber']) / number_planes_total).astype('int')
    online_data = online_data[ens].to_numpy().T
    cursor = cursor[frames]
    trial_end = trial_end[trial_end<(frames[-1]+len_base)]
    trial_start = trial_start[trial_start<(frames[-1]+len_base)]
    
    if online_data.shape[1] != cursor.size:
        raise ValueError("Data and cursor appear to be mismatched in time.")

    # Generate all possible pairwise combinations. We will find the combination
    # with the maximal correlation value
    e2_possibilities = list(combinations(np.arange(ens.size), 2))
    best_e2_combo = None
    best_e2_combo_val = 0.0
    all_corrs = []
    # Loop over each possible E2 combination. Simulate a cursor using the
    # data matrix, with the current combination as the simulated E2 neurons
    # The correlation of this simulated cursor with the real cursor will be the
    # score assigned to this particular E2 combination.
    for e2 in e2_possibilities:
        e1 = [i for i in range(ens.size) if i not in e2]
        correlation = 0
        for i in range(trial_end.size):
            start_idx = np.where(frames>(trial_start[i] - len_base))[0][0]
            aux_end = trial_end[i] - len_base
            if aux_end > frames[-1]:
                aux_end = frames[-1]
            end_idx = np.where(frames>=aux_end)[0][0]
            simulated_cursor = \
                np.sum(online_data[e2,start_idx:end_idx], axis=0) - \
                np.sum(online_data[e1,start_idx:end_idx], axis=0)
        trial_corr = np.nansum(
            cursor[start_idx:end_idx]*simulated_cursor
        )
        correlation += trial_corr
        # If this is the best E2 combo so far, record it
        all_corrs.append(correlation)
        if correlation > best_e2_combo_val:
            best_e2_combo_val = correlation
            best_e2_combo = e2

    # Sometimes we only get negative correlations, so we should return None
    if best_e2_combo is None:
        best_e2_neurons = [np.nan, np.nan]
    else:
        best_e2_neurons = [ens_neur[best_e2_combo[0]], ens_neur[best_e2_combo[1]]]
    return np.array(best_e2_neurons)


def test_copy_finish(remote, local):
    missed = []
    for animal in get_all_animals(remote):
        animal_path = os.path.join(remote, animal)
        for day in get_animal_days(animal_path):
            print(f'{animal} {day}')
            hf1 = h5py.File(encode_to_filename(remote, animal, day), 'r')
            hf1.close()
            try:
                hf2 = h5py.File(encode_to_filename(local, animal, day), 'r')
                hf2.close()
            except (OSError, FileNotFoundError) as e:
                missed.append((animal, day))
    return missed




def novel_detrend_df_f(A, b, C, f, YrA=None, quantileMin=8, frames_window=250,
                 flag_auto=True, use_fast=False, scaleC=False):
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
    oC = C
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
            if scaleC:
                F_df = C / (Df + Fd)
            else:
                F_df = oC / (Df + Fd)
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
