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
import bokeh.plotting as bpl

from skimage.feature import peak_local_max
from scipy import ndimage
import copy
from matplotlib import interactive
import sys, traceback
import imp
interactive(True)


def all_run(folder, animal, day, number_planes=4, number_planes_total=6):
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
    ffull = [folder_path + matinfo['fname'][0]]            # filename to be processed
    fbase = [folder_path + matinfo['fbase'][0]] 
    
    try:
        num_files, len_bmi = separate_planes(folder, animal, day, ffull, 'bmi', number_planes, number_planes_total)
        num_files_b, len_base = separate_planes(folder, animal, day, fbase, 'baseline', number_planes, number_planes_total)
    except Exception as e:
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit('Error in separate planes')
        
        
    nam = folder_path + 'readme.txt'
    readme = open(nam, 'w+')
    readme.write("num_files_b = " + str(num_files_b) + '\n')
    readme.write("num_files = " + str(num_files)+ '\n')
    readme.write("len_base = " + str(len_base)+ '\n')
    readme.write("len_bmi = " + str(len_bmi)+ '\n')
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
    fpath = folder + 'raw/' + animal + '/' + day + '/analysis/' 
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    err_file = open("errlog.txt", 'a+')  # ERROR HANDLING
    try:
        print("Trying to swapoff")
        os.system('swapoff /home/lab/Nuria/Swap/swapfile.img')  
        print("SUCCESS")
    except Exception as e:
        print("Swap Off Failed!", str(e.args))
    
    # create the swap to be able to allocate the file in memory
    print('Swapping on')
    os.system('swapon /home/lab/Nuria/Swap/swapfile.img') 
    # load the big file we will need to separate
    print('loading image...')
    ims = tifffile.TiffFile(ffull[0])
    dims = [len(ims)] + list(ims[0].shape)  
    print('Image loaded')
    len_im = int(dims[0]/number_planes_total)
    num_files = int(np.ceil(len_im/lim_bf))
    
    for plane in np.arange(number_planes):
        len_im = int(dims[0]/number_planes_total)
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
                new_img = ims.pages[int((ind + lim_bf*nf)*number_planes_total + plane), :, :].asarray()
                big_file[:, ind] = np.reshape(new_img, np.prod(dims[1:]), order=order)
            
            #to plot the image before closing big_file (as a checkup that everything went smoothly)
            if not os.path.exists(fpath + str(plane) + '/'):
                os.makedirs(fpath + str(plane) + '/')
            imgtosave = np.transpose(np.reshape(np.nanmean(big_file,1), [dims[1],dims[2]])) # careful here if the big_file is too big to average
            plt.imshow(imgtosave)
            plt.savefig(fpath + str(plane) + '/' + 'nf' + str(nf) + '_rawmean.png', bbox_inches="tight")
            plt.close()
            
            big_file.flush()
            del big_file
    
    # clean memory    
    ims.close()
    del ims

    
    # save the mmaps as tiff-files for caiman
    for plane in np.arange(number_planes):
        len_im = int(dims[0]/number_planes_total)
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
    
    len_im = int(dims[0]/number_planes_total)
    
    try:
        print('Swapping off')
        os.system('swapoff /home/lab/Nuria/Swap/swapfile.img')  
    except Exception as e:
        print('Error swapping off')
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit()
    err_file.close()
    
    return num_files, len_im


def separate_planes_multiple_baseline(folder, animal, day, ffull, ffull2, var='baseline', number_planes=4, number_planes_total=6, order='F', lim_bf=10000):
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
    fpath = folder + 'raw/' + animal + '/' + day + '/analysis/' 
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        
    print('Swapping on')
    os.system('swapon /home/lab/Nuria/Swap/swapfile.img') 
    # load the big file we will need to separate
    print('loading image...')
    imb1 = tifffile.TiffFile(ffull[0]).pages
    imb2 = tifffile.TiffFile(fful2l[0]).pages
    print('Images loaded')
    dims1 = [len(imb1)] + list(imb1[0].shape)
    dims2 = [len(imb2)] + list(imb2[0].shape)
    dims = dims1 + dims2
    len_im = int(dims[0]/number_planes_total)
    num_files = int(np.ceil(len_im/lim_bf))
    
    for plane in np.arange(number_planes):
        first_images_left = int(dims1[0]/number_planes_total)
        len_im = int(dims[0]/number_planes_total)
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
                    new_img = imb1.pages[int(plane+(ind + lim_bf*nf)*number_planes_total), :, :].asarray()
                    first_images_left -= 1
                else:
                    new_img = imb2.pages[int(plane+(ind - int(imb1.shape[0]/number_planes_total) + lim_bf*nf)*number_planes_total), :, :].asarray()
                    
                big_file[:, ind] = np.reshape(new_img, np.prod(dims[1:]), order=order)
                
            #to plot the image before closing big_file (as a checkup that everything went smoothly)
            if not os.path.exists(fpath + str(plane) + '/'):
                os.makedirs(fpath + str(plane) + '/')
            imgtosave = np.transpose(np.reshape(np.nanmean(big_file,1), [dims[1],dims[2]]))
            plt.imshow(imgtosave)
            plt.savefig(fpath + str(plane) + '/' + 'nf' + str(nf) + '_rawmean.png', bbox_inches="tight")
            plt.close()
            
            big_file.flush()
            del big_file
    
    # clean memory    
    del imb1
    del imb2
    
    
    # save the mmaps as tiff-files for caiman
    for plane in np.arange(number_planes):
        len_im = int(dims[0]/number_planes_total)
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
    
    print('Swapping off')
    os.system('swapoff /home/lab/Nuria/Swap/swapfile.img')  
    
    return num_files, len_im


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
    initialZ = int(matinfo['initialZ'][0][0])
    fr = matinfo['fr'][0][0]
    
    if dend:
        sec_var = 'Dend'
    else:
        sec_var = ''
    
    print('*************Starting with analysis*************')
    neuron_mats = []

    for plane in np.arange(number_planes):
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
        dff, com, cnm2, totdes = caiman_main(fpath, fr, fnames, zval, dend, display_images)
        print ('Caiman done: saving ... plane: ' + str(plane) + ' file: ' + str(nf)) 
        
        Asparse = scipy.sparse.csr_matrix(cnm2.estimates.A)
        f.create_dataset('dff', data = dff)                   #activity
        f.create_dataset('com', data = com)                         #distance
        g = f.create_group('Nsparse')                               #neuron shape
        g.create_dataset('data', data = Asparse.data)
        g.create_dataset('indptr', data = Asparse.indptr)
        g.create_dataset('indices', data = Asparse.indices)
        g.attrs['shape'] = Asparse.shape
        f.create_dataset('neuron_act', data =  cnm2.estimates.S)          #spikes
        f.create_dataset('C', data =  cnm2.estimates.C)                   #temporal activity
        f.create_dataset('base_im', data = cnm2.estimates.b)                 #baseline image
        f.create_dataset('tot_des', data = totdes)                          # total desplacement
        f.close()  
        
    print('... done') 
     

        
def put_together(folder, animal, day, number_planes=4, number_planes_total=6, sec_var='', toplot=True):       
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
    ffull = [folder_path + matinfo['fname'][0]]
    metadata = tifffile.TiffFile(ffull[0]).scanimage_metadata
    fr = matinfo['fr'][0][0]   
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
        auxb = np.asarray(f['base_im'])[:,1]
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
    auxZ[:,2] = np.repeat(matinfo['initialZ'][0][0],all_com.shape[0])
    all_com += auxZ
    
    
    # Reorganize sparse matrix of spatial components
    dims = all_neuron_shape.shape  
    dims = [int(np.sqrt(dims[0])), int(np.sqrt(dims[0])), all_neuron_shape.shape[1]]
    Asparse = scipy.sparse.csr_matrix(all_neuron_shape)
    Afull = np.reshape(all_neuron_shape.toarray(),dims)
    
    # separates "real" neurons from dendrites
    print ('finding neurons')
    pred, _ = evaluate_components_CNN(all_neuron_shape, dims[:2], [3,3])
    nerden = np.zeros(Afull.shape[2]).astype('bool')
    nerden[np.where(pred[:,1]>0.75)] = True
    #nerden = neurons_vs_dend(all_neuron_shape) # True is a neuron
    
    # obtain the real position of components A
    new_com = obtain_real_com(fanal, Afull, all_com, nerden)
    
    # sanity check of the neuron's quality
    plot_Cs(fanal, all_C, nerden)
    
    print('success!!')
    
    # identify ens_neur (it already plots sanity check in raw/analysis
    online_data = pd.read_csv(folder_path + matinfo['fcsv'][0])
    mask = matinfo['allmask']
    
    print('finding ensemble neurons')
    
    ens_neur = detect_ensemble_neurons(fanal, all_C, online_data, len(online_data.keys())-2,
                                             new_com, metadata, neuron_plane, number_planes_total, vars.len_base)
    
    
    # obtain trials hits and miss
    trial_end = (matinfo['trialEnd'][0] + vars.len_base).astype('int')
    trial_start = (matinfo['trialStart'][0] + vars.len_base).astype('int')
    if len(matinfo['hits']) > 0 : 
        hits = (matinfo['hits'][0] + vars.len_base).astype('float')
    else:
        hits = []
    if len(matinfo['miss']) > 0 : 
        miss = (matinfo['miss'][0] + vars.len_base).astype('float')
    else:
        miss = []
    # to remove false end of trials
    if trial_end.shape[0] > trial_start.shape[0]:
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
    
    print('finding red neurons')
    
    # obtain the neurons label as red (controlling for dendrites)
    redlabel = red_channel(red, neuron_plane, nerden, new_com, all_red_im, all_base_im, fanal, number_planes)
    redlabel[ens_neur.astype('int')] = True    
    
    # obtain the frequency
    frequency = obtainfreq(matinfo['frequency'][0], vars.len_bmi)
    
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
        plt.close('all')

    #fill the file with all the correct data!
    try:
        fall = h5py.File(folder_dest + 'full_' + animal + '_' + day + '_' + sec_var + '_data.hdf5', 'w-')
    except IOError:
        print(" OOPS!: The file already existed please try with another file, no results will be saved!!!")
        
        
    print('saviiiiiing')
        
    fall.create_dataset('dff', data = all_dff) # (array) (Ft - Fo)/Fo . Increment of fluorescence
    fall.create_dataset('C', data = all_C)  # (array) Relative fluorescence of each component
    fall.create_dataset('SNR', data = all_SNR)  # (array) Signal to noise ratio of each component
    fall.create_dataset('com_cm', data = all_com) # (array) Position of the components as given by caiman 
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
    fall.create_dataset('trial_end', data = trial_end) # (array) When a trial ended. Can be a hit or a miss
    fall.create_dataset('trial_start', data = trial_start) # (array) When a trial started
    fall.attrs['fr'] =  matinfo['fr'][0][0] # (int) Framerate
    fall.create_dataset('redlabel', data = redlabel) # (array-bool) True labels neurons as red
    fall.create_dataset('nerden', data = nerden) # (array-bool) True labels components as neurons
    fall.create_dataset('hits', data = hits) # (array) When the animal hit the target 
    fall.create_dataset('miss', data = miss) # (array) When the animal miss the target
    fall.create_dataset('array_t1', data = array_t1) # (array) index of the trials that ended in hit
    fall.create_dataset('array_miss', data = array_miss) # (array) Index of the trials that ended in miss
    fall.create_dataset('cursor', data = matinfo['cursor'][0]) # (array) Online cursor of the BMI
    fall.create_dataset('freq', data = frequency) # (array) Frenquency resulting of the online cursor.
    
    fall.close()
    
    print('all done!!')


def red_channel(red, neuron_plane, nerden, new_com, all_red_im, all_base_im, fanal, number_planes=4, maxdist=4, toplot=True):  
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
        # for some reason the motion correction does not work the other way around (first red)
        _, _, shift, _ = motion_correct_iteration(all_base_im[:,:,plane].astype('float32'), all_red_im[:,:,plane].astype('float32'),1)
        maskred = copy.deepcopy(np.transpose(red[plane]))
        mm = np.sum(maskred,1)
        maskred = maskred[~np.isnan(mm),:].astype('float32')
        if np.nansum(abs(np.asarray(shift))) < 20:  # hopefully the motion correction worked
            maskred[:,0] -= shift[1].astype('float32')
            maskred[:,1] -= shift[0].astype('float32')
        else:
            print ('There was an issue with the motion correction during red_channel comparison. Please check!')
            
        # creates a new image with the shifts found
        M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
        min_, max_ = np.min(all_red_im[:,:,plane]), np.max(all_red_im[:,:,plane])
        new_img = np.clip(cv2.warpAffine(all_red_im[:,:,plane], M, (all_red_im.shape[0], all_red_im.shape[1]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), min_, max_)
        
        # find distances
        
        neur_plane = neuron_plane[plane].astype('int')
        aux_nc = np.zeros(neur_plane)
        aux_nc = new_com[ind_neur:neur_plane+ind_neur, :2]
        aux_nerden = nerden[ind_neur:neur_plane+ind_neur]
        redlabel = np.zeros(neuron_plane.shape[0]).astype('bool')
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
        if nerden[neur]:
            center_mass = scipy.ndimage.measurements.center_of_mass(Afull[:,:,neur]>thres)
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
                
        else:
            new_com[neur,:] = [all_com[neur, 1], all_com[neur, 0], all_com[neur, 2]]
            x1 = int(all_com[neur,0]-img_size)
            x2 = int(all_com[neur,0]+img_size)
            y1 = int(all_com[neur,1]-img_size)
            y2 = int(all_com[neur,1]+img_size)
            
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
        
                
def detect_ensemble_neurons(fanal, all_C, online_data, units, com, metadata, neuron_plane, number_planes_total, len_base, auxtol=6, cormin=0.5):
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
    neurcor = np.ones((units, all_C.shape[0])) * np.nan
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
    
    for npro in np.arange(all_C.shape[0]):
        for non in np.arange(units): 
            ens = (online_data.keys())[2+non]
            frames = (np.asarray(online_data['frameNumber']) / number_planes_total).astype('int') + len_base 
            auxonline = (np.asarray(online_data[ens]) - np.nanmean(online_data[ens]))/np.nanmean(online_data[ens]) 
            auxC = all_C[npro,frames]/10000
            neurcor[non, npro] = pd.DataFrame(np.transpose([auxC[~np.isnan(auxonline)], auxonline[~np.isnan(auxonline)]])).corr()[0][1]
    
    neurcor[neurcor<cormin] = np.nan    
    auxneur = copy.deepcopy(neurcor)
    
    
    for un in np.arange(units):
        print(['finding neuron: ' + str(un)])
        tol = auxtol
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
                    print('Error couldnt find neuron' + str(un) + ' with this tolerance. Reducing tolerance')
                    if iter > 0:
                        neurcor = auxneur
                        tol *= 1.5
                        iter -= 1
                    else:
                        print ('where are my neurons??')
                        break
            else:
                print('No neurons that were correlated')
                auxcom = com[:,:2]
                dist = scipy.spatial.distance.cdist(centermass, auxcom)[0]
                if plane == 0:
                    dist[ind_neuron_plane[0]:] = np.nan
                else:
                    dist[:ind_neuron_plane[plane-1]] = np.nan
                    dist[ind_neuron_plane[plane]:] = np.nan
                indx = np.where(dist==np.nanmin(dist))[0][0]
                finalcorr[un] = np.nan
                finalneur[un] = indx
                finaldist[un] = np.nanmin(dist)
                not_good_enough = False
        print('tol value at: ', str(tol))
        print('Correlated with value: ', str(finalcorr[un]), ' with a distance: ', str(finaldist[un]))

                
        auxp = com[finalneur[un].astype(int),:2].astype(int)
        pmask[auxp[1], auxp[0]] = 2   #to detect
        pmask[centermass[0,1], centermass[0,0]] = 1   #to detect
    
    
    plt.figure()
    plt.imshow(pmask)
    plt.savefig(fanal + 'ens_masks.png', bbox_inches="tight")
    
    
    fig1 = plt.figure(figsize=(16,6))
    for non in np.arange(units): 
        ax = fig1.add_subplot(units, 1, non + 1)
        ens = (online_data.keys())[2+non]
        frames = (np.asarray(online_data['frameNumber']) / number_planes_total).astype('int') + len_base 
        auxonline = (np.asarray(online_data[ens]) - np.nanmean(online_data[ens]))/np.nanmean(online_data[ens])
        auxC = all_C[finalneur[non].astype('int'), frames] 
        
        ax.plot(auxonline[-5000:])
        ax.plot(auxC[-5000:]/10000)
        
    plt.savefig(fanal + 'ens_online_offline.png', bbox_inches="tight")

    return finalneur.astype('int')

    

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
    K = 4                       # number of components per patch
    
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
    print('***************Motion correction has ended*************')
    # maximum shift to be used for trimming against NaNs
    
    totdes = [np.nansum(mc.x_shifts_els), np.nansum(mc.y_shifts_els)]

    #%% MEMORY MAPPING
    # memory map the file in order 'C'
    fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
    border_to_0 = bord_px_els     # number of pixels to exclude
    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                               border_to_0=bord_px_els)  # exclude borders
    
    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)
        
    
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
    
    cm.stop_server(dview=dview)
    try:
        F_dff = detrend_df_f(cnm2.estimates.A, cnm2.estimates.b, cnm2.estimates.C, cnm2.estimates.f, YrA=cnm2.estimates.YrA, quantileMin=8, frames_window=250)
        #F_dff = detrend_df_f(cnm.A, cnm.b, cnm.C, cnm.f, YrA=cnm.YrA, quantileMin=8, frames_window=250)
    except:
        F_dff = cnm2.estimates.C * np.nan
        print ('WHAAT went wrong again?')
    
    
    print ('***************stopping cluster*************')
    #%% STOP CLUSTER and clean up log files
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
        
    return F_dff, com, cnm2, totdes  

