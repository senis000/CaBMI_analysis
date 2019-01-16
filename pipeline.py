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
import bokeh.plotting as bpl

from skimage.feature import peak_local_max
from scipy import ndimage
import copy
from matplotlib import interactive
import sys, traceback
interactive(True)


def all_run(folder, animal, day, numplanes_useful=4, numplanes_tot=6):
    folder_path = folder + 'raw/' + animal + '/' + day + '/'
    folder_final = folder + 'processed/' + animal + '/' + day + '/'
    err_file = open("errlog.txt", 'a+')  # ERROR HANDLING
    if not os.path.exists(folder_final):
        os.makedirs(folder_final)
    
    finfo = folder_path +  'wmat.mat'  #file name of the mat 
    matinfo = scipy.io.loadmat(finfo)
    ffull = [folder_path + matinfo['fname'][0]]            # filename to be processed
    fbase = [folder_path + matinfo['fbase'][0]] 
    
    try:
        num_files, len_bmi = separate_planes(folder, animal, day, ffull, 'bmi', numplanes_useful, numplanes_tot)
        num_files_b, len_base = separate_planes(folder, animal, day, fbase, 'baseline', numplanes_useful, numplanes_tot)
    except Exception as e:
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit('Error in separate planes')
        
        
    try:
        analyze_raw_planes(folder, animal, day, num_files, num_files_b, numplanes_useful, False)
    except Exception as e:
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit('Error in analyze raw')
        
    try:  
        put_together(folder, animal, day, len_base, len_bmi, numplanes_useful, numplanes_tot)
    except Exception as e:
        tb = sys.exc_info()[2]
        err_file.write("\n{}\n".format(folder_path))
        err_file.write("{}\n".format(str(e.args)))
        traceback.print_tb(tb, file=err_file)
        err_file.close()
        sys.exit('Error in put together')
    
    err_file.close()
    return num_files_b, len_base, num_files, len_bmi  
    #return num_files_b, len_base # Separates base only delete later
    

def separate_planes(folder, animal, day, ffull, var='bmi', num_planes_useful=4, num_planes_total=6, order='F', lim_bf=9000):
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
    im = tifffile.imread(ffull[0])
    dims = im.shape
    print('Image loaded')
    len_im = int(dims[0]/num_planes_total)
    num_files = int(np.ceil(len_im/lim_bf))
    
    for plane in np.arange(num_planes_useful):
        len_im = int(dims[0]/num_planes_total)
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
                new_img = im[int((ind + lim_bf*nf)*num_planes_total + plane), :, :]
                big_file[:, ind] = np.reshape(new_img, np.prod(dims[1:]), order=order)
            
            #to plot the image before closing big_file (as a checkup that everything went smoothly)
            imgtosave = np.transpose(np.reshape(np.nanmean(big_file,1), [dims[1],dims[2]]))
            plt.imshow(imgtosave)
            plt.savefig(fpath + str(plane) + '/' + 'nf' + str(nf) + '_rawmean.png', bbox_inches="tight")
            plt.close()
            
            big_file.flush()
            del big_file
    
    # clean memory    
    del im

    
    # save the mmaps as tiff-files for caiman
    for plane in np.arange(num_planes_useful):
        len_im = int(dims[0]/num_planes_total)
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
    
    len_im = int(dims[0]/num_planes_total)
    
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


def separate_planes_multiple_baseline(folder, animal, day, ffull, ffull2, var='baseline', num_planes_useful=4, num_planes_total=6, order='F', lim_bf=10000):
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
    imb1 = io.imread(ffull[0])
    imb2 = io.imread(ffull2[0])
    dims = [imb1.shape[0] + imb2.shape[0], imb1.shape[1], imb1.shape[2]]
    print('Images loaded')
    len_im = int(dims[0]/num_planes_total)
    num_files = int(np.ceil(len_im/lim_bf))
    
    for plane in np.arange(num_planes_useful):
        first_images_left = int(imb1.shape[0]/num_planes_total)
        len_im = int(dims[0]/num_planes_total)
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
                    new_img = imb1[int(plane+(ind + lim_bf*nf)*num_planes_total), :, :]
                    first_images_left -= 1
                else:
                    new_img = imb2[int(plane+(ind - int(imb1.shape[0]/num_planes_total) + lim_bf*nf)*num_planes_total), :, :]
                    
                big_file[:, ind] = np.reshape(new_img, np.prod(dims[1:]), order=order)
                
            #to plot the image before closing big_file (as a checkup that everything went smoothly)
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
    for plane in np.arange(num_planes_useful):
        len_im = int(dims[0]/num_planes_total)
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


def analyze_raw_planes(folder, animal, day, num_files, num_files_b, num_planes=4, dend=False, display_images=True, save_results = False):
    # folder: full address not including project/animal/day
    # dend: boolean if we are studying dendrites
    folder_path = folder + 'raw/' + animal + '/' + day + '/separated/'
    finfo = folder + 'raw/' + animal + '/' + day + '/wmat.mat'  #file name of the mat 
    matinfo = scipy.io.loadmat(finfo)
    planes_val = np.asarray(matinfo['Zactual'][0])
    initialZ = int(matinfo['initialZ'][0][0])
    fr = matinfo['fr'][0][0]
    
    if dend:
        sec_var = 'Dend'
    else:
        sec_var = ''
    
    print('*************Starting with analysis*************')
    neuron_mats = []

    for plane in np.arange(num_planes):
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
            
        zval = calculate_zvalues(folder, plane, planes_val)
        print(fnames)
        dff, com, cnm2 = caiman_main(fpath, fr, fnames, zval, dend, display_images, save_results)
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
        f.close()  
        
    print('... done') 
        
        
def put_together(folder, animal, day, len_base, len_bmi, number_planes=4, num_planes_total=6, sec_var='', toplot=True):       
    # Folder to load/save
    folder_path = folder + 'raw/' + animal + '/' + day + '/'
    folder_dest = folder + 'processed/' + animal + '/'
    folder_dest_anal = folder + 'processed/' + animal + '/analysis/'
    if not os.path.exists(folder_dest):
        os.makedirs(folder_dest)
    if not os.path.exists(folder_dest_anal):
        os.makedirs(folder_dest)
    
    # Load information
    finfo = folder_path +  'wmat.mat'  #file name of the mat 
    matinfo = scipy.io.loadmat(finfo)
    fr = matinfo['fr'][0][0]   
    planes_val = np.asarray(matinfo['Zplanes'][0])
    folder_red = folder + 'raw/' + animal + '/' + day + '/'
    fmat = folder_red + 'red.mat' 
    redinfo = scipy.io.loadmat(fmat)
    red = redinfo['red'][0]
    com_list = []
    for plane in np.arange(number_planes): 
        try:
            f = h5py.File(folder_path + 'bmi_' + sec_var + '_' + str(plane) + '.hdf5', 'r')
        except OSError:
            break
        base_im = np.asarray(f['base_im'])
        if plane == 0:
            all_dff = np.asarray(f['dff'])
            all_C = np.asarray(f['C'])
            all_com = np.asarray(f['com'])
            com_list.append(np.asarray(f['com']))
            g = f['Nsparse']
            all_neuron_shape = scipy.sparse.csr_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), g.attrs['shape'])
            all_base_im = np.ones((base_im.shape[0], base_im.shape[1], number_planes)) *np.nan
            all_neuron_act = np.asarray(f['neuron_act'])
        else:
            all_dff = np.concatenate((all_dff, np.asarray(f['dff'])), 0)
            all_C = np.concatenate((all_C, np.asarray(f['C'])), 0)
            all_com = np.concatenate((all_com, np.asarray(f['com'])))
            com_list.append(np.asarray(f['com']))
            g = f['Nsparse']
            gaux = scipy.sparse.csr_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), g.attrs['shape'])
            all_neuron_shape = scipy.sparse.hstack([all_neuron_shape, gaux])
            all_neuron_act = np.concatenate((all_neuron_act, np.asarray(f['neuron_act'])), 0)
        all_base_im[:, :, plane] = base_im
        f.close()
    
    auxZ = np.zeros((all_com.shape))
    auxZ[:,2] = np.repeat(matinfo['initialZ'][0][0],all_com.shape[0])
    all_com += auxZ
    
    fanal = folder + 'raw/' + animal + '/' + day + '/analysis/'
    Asparse = scipy.sparse.csr_matrix(all_neuron_shape)
    dims = [int(np.sqrt(dims[0])), int(np.sqrt(dims[0])), Asparse.shape[1]]
    Afull = np.reshape(A.toarray(),dims)
    
    # obtain the real position of components A
    com = obtain_real_com(fanal, Afull, all_com)
    
    # identify ens_neur (it already plots sanity check in raw/analysis
    online_data = pd.read_csv(folder_path + matinfo['fcsv'][0])
    mask = matinfo['allmask']
    
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    ens_neur = detect_ensemble_neurons(fanal, all_dff, online_data, len(online_data.keys())-2,
                                             all_com,matinfo['allmask'], num_planes_total, len_base)
    
    
    # obtain trials hits and miss
    trial_end = matinfo['trialEnd'][0] + len_base
    trial_start = matinfo['trialStart'][0] + len_base
    if len(matinfo['hits']) > 0 : 
        hits = matinfo['hits'][0] + len_base
    else:
        hits = []
    if len(matinfo['miss']) > 0 : 
        miss = matinfo['miss'][0] + len_base
    else:
        miss = []
    array_t1 = np.zeros(hits.shape[0], dtype=int)
    array_miss = np.zeros(miss.shape[0], dtype=int)
    for hh, hit in enumerate(hits): array_t1[hh] = np.where(trial_end==hit)[0][0]
    for mm, mi in enumerate(miss): array_miss[mm] = np.where(trial_end==mi)[0][0]
    
    
    # separates "real" neurons from dendrites
    nerden = neurons_vs_dend(all_neuron_shape) # True is a neuron
    
    # obtain the neurons label as red (controlling for dendrites)
    redlabel = red_channel(red, com_list, number_planes)*nerden
    
    # obtain the frequency
    frequency = obtainfreq(matinfo['frequency'][0], len_bmi)
    
    # sanity checks
    if toplot:
        plt.plot(np.nanmean(dff,0))
        plt.title('DFFs')
        plt.savefig(folder_dest_anal + animal + '_' + day + 'dff.png', bbox_inches="tight")
        plt.plot(matinfo['cursor'][0])
        plt.title('cursor')
        plt.savefig(folder_dest_anal + animal + '_' + day + 'cursor.png', bbox_inches="tight")



    #fill the file with all the correct data!
    try:
        fall = h5py.File(folder_dest + 'full_' + animal + '_' + day + '_' + sec_var + '_data.hdf5', 'w-')
    except IOError:
        print(" OOPS!: The file already existed please try with another file, no results will be saved!!!")
        
    fall.create_dataset('dff', data = all_dff)
    fall.create_dataset('C', data = all_C)
    fall.create_dataset('com_cm', data = all_com)
    fall.attrs['blen'] = len_base
    gall = fall.create_group('Nsparse')
    gall.create_dataset('data', data = Asparse.data)
    gall.create_dataset('indptr', data = Asparse.indptr)
    gall.create_dataset('indices', data = Asparse.indices)
    gall.attrs['shape'] = Asparse.shape
    fall.create_dataset('neuron_act', data = all_neuron_act)
    fall.create_dataset('base_im', data = all_base_im)
    fall.create_dataset('online_data', data = online_data)
    fall.create_dataset('ens_neur', data = ens_neur)    
    fall.create_dataset('trial_end', data = trial_end)
    fall.create_dataset('trial_start', data = trial_start)
    fall.attrs['fr'] =  matinfo['fr'][0][0]
    fall.create_dataset('redlabel', data = redlabel)
    fall.create_dataset('nerden', data = nerden)
    fall.create_dataset('hits', data = hits)
    fall.create_dataset('miss', data = miss)
    fall.create_dataset('array_t1', data = array_t1)
    fall.create_dataset('array_miss', data = array_miss)
    fall.create_dataset('cursor', data = matinfo['cursor'][0])
    fall.create_dataset('freq', data = frequency)
    
    fall.close()
    

def red_channel(red, com_list, all_red_im, folder_path, num_planes=4, maxdist=8):  
    #function to identify red neurons
    all_red = []

    if len(red) < num_planes:
        num_planes = len(red)
    for plane in np.arange(num_planes):
        maskred = np.transpose(red[plane])
        mm = np.sum(maskred,1)
        maskred = maskred[~np.isnan(mm),:]
        red_im = all_red_im[:,:,plane]
        toplot = np.zeros((red_im.shape[0], red_im.shape[1]))
        com = com_list[plane][:,0:2]
        redlabel = np.zeros(com.shape[0]).astype('bool')
        dists = scipy.spatial.distance.cdist(com, maskred)
        for neur in np.arange(com.shape[0]):
            redlabel[neur] = np.sum(dists[neur,:]<maxdist)
        redlabel[redlabel>0]=True
        all_red.append(redlabel)
        auxtoplot = com[redlabel,:]

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1,2,1)
        for ind in np.arange(maskred.shape[0]):
            auxlocx = maskred[ind,1].astype('int')
            auxlocy = maskred[ind,0].astype('int')
            toplot[auxlocx-1:auxlocx+1,auxlocy-1:auxlocy+1] = np.nanmax(red_im)
        ax1.imshow(red_im + toplot, vmax=np.nanmax(red_im))
        
        toplot = np.zeros((red_im.shape[0], red_im.shape[1]))
        ax2 = fig1.add_subplot(1,2,2)
        for ind in np.arange(auxtoplot.shape[0]):
            auxlocx = auxtoplot[ind,1].astype('int')
            auxlocy = auxtoplot[ind,0].astype('int')
            toplot[auxlocx-1:auxlocx+1,auxlocy-1:auxlocy+1] = np.nanmax(red_im)
        ax2.imshow(red_im + toplot, vmax=np.nanmax(red_im))
        plt.savefig(folder_path + 'analysis/' + str(plane) + '/redneurmask.png', bbox_inches="tight")
        plt.close("all")
        

        
    all_red = np.concatenate(all_red)
    return all_red



def neurons_vs_dend(A, tol=0.1, minsize=25): 
    #function to distinguish "real" neurons from dendrite activity
    Asize = int(np.sqrt(A.shape[0]))
    Afull = np.reshape(A.toarray(),[Asize,Asize,A.shape[1]])
    nerden = np.zeros(A.shape[1]).astype('bool')
    for ind in np.arange(A.shape[1]):
        auxA = Afull[:,:,ind]>tol
        if np.nansum(auxA)>minsize:
            nerden[ind] = True
    return nerden


def obtain_real_com(fanal, Afull, all_com, toplot=True, img_size = 20):
    #function to obtain the real values of com
    folder_path = fanal + '/Aplot/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    new_com = np.zeros((Afull.shape[2], 3))
    for neur in np.arange(Afull.shape[2]):
        center_mass = scipy.ndimage.measurements.center_of_mass(Afull[:,:,neur]>0.1)
        new_com[neur,:] = [center_mass[1], center_mass[0], all_com[neur,2]]
        img = Afull[int(center_mass[0]-img_size):int(center_mass[0]+img_size),int(center_mass[1]-img_size):int(center_mass[1]+img_size),neur]
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax1.imshow(img)
        ax2 = fig1.add_subplot(122)
        ax2.imshow(img>0.1)
                

def caiman_main(folder_path, fr, fnames, z=0, dend=False, display_images=False, save_results=False):
    
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

    dview = None
    
    #%% start a cluster for parallel processing
    """
    if 'dview' in locals():
        dview.terminate()
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)"""
    
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
    #m_els = cm.load(mc.fname_tot_els)   #it will crush if there is parallalization
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    print('***************Motion correction has ended*************')
    # maximum shift to be used for trimming against NaNs

    #%% MEMORY MAPPING
    # memory map the file in order 'C'
    fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
    border_to_0 = bord_px_els     # number of pixels to exclude
    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                               border_to_0=bord_px_els)  # exclude borders
    
    #%% restart cluster to clean up memory
    """
    dview.terminate()
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)"""

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)
        
    
   #if we are doing the main image       
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
    
#     #%% PLOT COMPONENTS (do not use during parallalization)
#      
#     if display_images:
#         plt.figure()
#         plt.subplot(221)
#         plt.imshow(np.nanmean(m_els,0))
#         plt.colorbar()
#         plt.subplot(222)
#         auxb = np.transpose(np.reshape(cnm.estimates.b[:,0], [int(np.sqrt(cnm.estimates.b.shape[0])), int(np.sqrt(cnm.estimates.b.shape[0]))]))
#         plt.imshow(auxb)
#         plt.title('Raw mean')
#         plt.subplot(223)
#         crd_good = cm.utils.visualization.plot_contours(
#             cnm.estimates.A[:, idx_components], np.nanmean(m_els,0), thr=.8)
#         plt.title('Contour plots of accepted components')
#         plt.subplot(224)
#         crd_bad = cm.utils.visualization.plot_contours(
#             cnm.estimates.A[:, idx_components_bad], Cn, thr=.8, vmax=0.2)
#         plt.title('Contour plots of rejected components')
#         plt.savefig(folder_path + 'comp.png', bbox_inches="tight")
#          
#         plt.close('all')

    #%% PLOT COMPONENTS 
     
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
        plt.savefig(folder_path + 'comp.png', bbox_inches="tight")
         
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
        print ('WHAAT went wrong again?')
    
    
    print ('***************stopping cluster*************')
    #%% STOP CLUSTER and clean up log files
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
    
#     #%% play along side original data (do not use during parallalization
#     if save_results:
#         #%% reconstruct denoised movie
#         denoised = cm.movie(cnm2.estimates.A.dot(cnm2.estimates.C) +
#                         cnm2.estimates.b.dot(cnm2.estimates.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
#         downsample_ratio = .2  # motion can be perceived better when downsampling in time
#         moviehandle = cm.concatenate([m_els.resize(1, 1, downsample_ratio),
#                         denoised.resize(1, 1, downsample_ratio)],
#                        axis=2)
#     
#         moviehandle.save(folder_path + 'mov.tif')
        
        
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
                zy[y,0] = z[int(auxcom[y,1])]
            com = np.concatenate((auxcom, zy),1)
        else:
            print('WARNING: Z value was not correctly defined, only X and Y values on file, z==zeros')
            print(['length of z was: ' + str(len(z))])
            com = np.concatenate((cm.base.rois.com(cnm2.estimates.A,dims[0],dims[1]), np.zeros((cnm2.estimates.A.shape[1], 1))),1)
    else:
        com = cm.base.rois.com(cnm2.estimates.A,dims[0],dims[1], dims[2])        
        
    return F_dff, com, cnm2


def detect_ensemble_neurons(folder_path, dff, online_data, units, com, mask, num_planes_total, len_base, auxtol=10, cormin=0.1):
    neurcor = np.ones((units, dff.shape[0])) * np.nan
    finalcorr = np.zeros(units)
    finalneur = np.zeros(units)
    maxmask = np.nanmax(mask)
    pmask = np.nansum(mask,2)
    iter = 40
    
    for npro in np.arange(dff.shape[0]):
        for non in np.arange(units): 
            ens = (online_data.keys())[2+non]
            frames = (np.asarray(online_data['frameNumber']) / num_planes_total).astype('int') + len_base -1
            neurcor[non, npro] = pd.DataFrame(np.transpose([dff[npro,frames], np.asarray(online_data[ens])])).corr()[0][1]
    
    neurcor[neurcor<cormin] = np.nan    
    auxneur = copy.deepcopy(neurcor)
    
    
    for un in np.arange(units):
        print(['finding neuron: ' + str(un)])
        tol = auxtol
        centermass = np.reshape(np.asarray(scipy.ndimage.measurements.center_of_mass(mask[:,:, un])),[1,2])
        not_good_enough = True
        while not_good_enough:
            if np.nansum(neurcor[un,:]) != 0:
                if np.nansum(np.abs(neurcor[un, :])) > 0 :
                    maxcor = np.nanmax(neurcor[un, :])
                    indx = np.where(neurcor[un, :]==maxcor)[0][0]
                    auxcom = np.reshape(com[indx,:2],[1,2])
                    dist = scipy.spatial.distance.cdist(centermass, auxcom)[0][0]
                    not_good_enough =  dist > tol
    
                    finalcorr[un] = neurcor[un, indx]
                    finalneur[un] = indx
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
                auxcom = com[:,:2]
                dist = scipy.spatial.distance.cdist(centermass, auxcom)[0]
                indx = np.where(dist==np.nanmin(dist))[0][0]
                finalcorr[un] = np.nan
                finalneur[un] = indx
                not_good_enough = False
        print('tol value at: ', str(tol))

                
        auxp = com[finalneur[un].astype(int),:2].astype(int)
        pmask[auxp[0], auxp[1]] = 2   #to detect
    
    print('Correlated with value', str(finalcorr) )
    plt.figure()
    plt.imshow(pmask)
    plt.savefig(folder_path + 'ens_masks.png', bbox_inches="tight")
    return finalneur
    

def calculate_zvalues(folder, plane, planes_val):
#     Zmatrix = np.asarray([[np.linspace(planes_val[0],planes_val[1],256)][0],
#                          [np.linspace(planes_val[1],planes_val[2],256)][0],
#                          [np.linspace(planes_val[2],planes_val[3],256)][0],
#                          [np.linspace(planes_val[3],planes_val[4],256)][0]])
#     z = Zmatrix[plane, :]
    finfo = folder + 'actuator.mat'  #file name of the mat 
    actuator = scipy.io.loadmat(finfo)
    options={ 0: 'yd1i',
              1: 'yd2i',
              2: 'yd3i',
              3: 'yd4i'}
    z = actuator[options[plane]][0]
    
    return z


def obtainfreq(origfreq, len_bmi=36000, iterat=2):
    freq = copy.deepcopy(origfreq)
    freq[:np.where(~np.isnan(origfreq))[0][0]] = 0
    if len_bmi<freq.shape[0]: freq = freq[:len_bmi]
    for it in np.arange(iterat):
        nanarray = np.where(np.isnan(freq))[0]
        for inan in nanarray[-1::-1]:
            freq[inan] = freq[inan-1]
    return freq
   

def load_movie(fname):    
    fname = [download_demo(fname)]     # the file will be downloaded if it doesn't already exist
    m_orig = cm.load_movie_chain(fname)
    downsample_ratio = .2  # motion can be perceived better when downsampling in time
    offset_mov = np.min(m_orig[:100])  # if the data has very negative values compute an offset value
    m_orig.resize(1, 1, 0.2).play(gain=10, offset = -offset_mov, fr=30, magnification=2)   # play movie (press q to exit)


