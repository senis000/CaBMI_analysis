
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
import seaborn as sns
from matplotlib import interactive
import xgboost
import shap

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
    
    if len(ens_neur)>0:
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
            
            # diff in xy
            auxdxy = []
            for nn in np.arange(ens_neur.shape[0]):
                for nns in np.arange(nn+1, ens_neur.shape[0]):
                    auxdxy.append(scipy.spatial.distance.euclidean(com_ens[nn,:2], com_ens[nns,:2]))
            
            diffxy_mean = np.nanmean(auxdxy)
            diffxy_max = np.nanmax(auxdxy)  
        else:
            dist_mean = np.nan
            dist_max = np.nan
            diffdepth_mean = np.nan
            diffdepth_max = np.nan
            diffxy_mean = np.nan
            diffxy_max = np.nan
        
        # dynamic range
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
    
    else:
        depth_mean = np.nan
        depth_max = np.nan
        dist_mean = np.nan
        dist_max = np.nan
        diffdepth_mean = np.nan
        diffdepth_max = np.nan
        diffxy_mean = np.nan
        diffxy_max = np.nan
        post_whole_std_mean = np.nan
        post_whole_std_max = np.nan
        post_base_std_mean = np.nan
        post_base_std_max = np.nan
    
    
    auxonstd = []
    for nn in np.arange(online_data.shape[1]):
        auxonstd.append(np.nanstd(online_data[:,nn]))
        
    onstd_mean = np.nanmean(auxonstd)
    onstd_max = np.nanmax(auxonstd)
    

    auxcursorstd = []
    cursor_std = np.nanstd(cursor)
    
    
    
    row_entry = np.asarray([depth_mean, depth_max, dist_mean, dist_max, diffdepth_mean, diffdepth_max, \
                             diffxy_mean, diffxy_max, \
                             onstd_mean, onstd_max, post_whole_std_mean, post_whole_std_max, post_base_std_mean, \
                             post_base_std_max, cursor_std])
    
    return row_entry


def plot_results(folder_plots, df_aux, first_ind=0, single_animal=True, mode='basic'):
    '''
    Function to plot all the basic results in terms of linear regression 
    '''
    columns = df_aux.columns.tolist()
    # depth mean
    columns_ler = columns[4:10]

    if mode =='basic':
        columns_aux = columns[10:25]  ## basic
        figsiz = (12, 20)
        sbx = 4
        sby = 4
    elif mode == 'SNR':
        columns_aux = columns[25:27]  ## SNR
        figsiz = (8, 4)
        sbx = 1
        sby = 2
        
    for cc, col in enumerate(columns_ler):
        fig1 = plt.figure(figsize=figsiz)
        for tt, btest in enumerate(columns_aux):
            ax = fig1.add_subplot(sbx, sby, tt+1)
            ax = sns.regplot(x=btest, y=col, data=df_aux)
            ax.set_xticks([])
            if single_animal:
                plotname = os.path.join(folder_plots, 'per_animal', df_aux.loc[first_ind][0] + '_' + mode + '_' + col)
            else:
                plotname = os.path.join(folder_plots, mode + '_' + col)
            fig1.savefig(plotname + '.png', bbox_inches="tight")
            fig1.savefig(plotname + '.eps', bbox_inches="tight")
            
    plt.close('all')
            


def create_dataframe(folder_main, file_csv, to_plot=True):
    # ineficient way to create the dataframe, but I want to be f* sure that each entry is correct.
    folder = os.path.join(folder_main, 'processed')
    folder_plots = os.path.join(folder_main, 'plots', 'learning_regressions')
    folder_snr = os.path.join(folder_main, 'onlineSNR')
    to_save_df = os.path.join(folder_main, 'df_all.hdf5')
    animals = os.listdir(folder)
    df_results = pd.read_csv(file_csv)
    columns_res = df_results.columns.tolist()
    columns_basic = ['depth_mean', 'depth_max', 'dist_mean', 'dist_max', 'diffdepth_mean', 'diffdepth_max', \
                         'diffxy_mean', 'diffxy_max', \
                         'onstd_mean', 'onstd_max', 'post_whole_std_mean', 'post_whole_std_max', 'post_base_std_mean', \
                         'post_base_std_max', 'cursor_std']
    columns = columns_res + columns_basic
    columns.insert(3,'label')
    df = pd.DataFrame(columns=columns)

    # obtain basic features
    print('obtaining basic features!')
    for aa,animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        mat_animal = np.zeros((len(filenames), len(columns_basic))) + np.nan
        for dd,filename in enumerate(filenames):
            day = filename[-17:-11]
            print ('Analyzing animal: ' + animal + ' day: ' + day)
            try:
                mat_animal[dd, :] = basic_entry (folder, animal, day)
            except OSError:
                print('day: ' + day + ' not obtained. ERRRRROOOOOOOOOOOOOOR')    
                break               
        df_res_animal = df_results.loc[df_results['animal']==animal]
        if len(filenames) != len(df_res_animal):
            print('Experiment number missmatch!!. STOPING')
#             break
        if animal[:2] == 'IT':
            aux_label = 0
        elif animal[:2] == 'PT':
            aux_label = 1
        df_res_animal.insert(3, 'ITPTlabel', np.repeat([aux_label], len(df_res_animal)))
        df_basic = pd.DataFrame(mat_animal, columns=columns_basic, index=df_res_animal.index)
        df_animal = df_res_animal.join(df_basic)
        if to_plot:
            plot_results(folder_plots, df_animal,  df_res_animal.index[0])
        df = df.append(df_animal)
    if to_plot:
        plot_results(folder_plots, df, single_animal=False)
    
    #obtain snr features
    print('obtaining snrs')
    snr_vector_mean = []#np.zeros(len(df))
    snr_vector_max = []#np.zeros(len(df))
    for aa,animal in enumerate(animals):
        folder_path = os.path.join(folder_snr, animal)
        filenames = os.listdir(folder_path)
        for dd,filename in enumerate(filenames):
            if (animal=='IT8') & (filename=='190301'):
                continue
            else:
                print ('Analyzing animal: ' + animal + ' day: ' + filename)
                f = h5py.File(os.path.join(folder_path, filename,'onlineSNR.hdf5'), 'r')
                aux_snr = np.asarray(f['SNR_ens'])
                f.close()
                snr_vector_mean.append(np.nanmean(aux_snr))
                snr_vector_max.append(np.nanmax(aux_snr))
    try:
        df['onlineSNRmean'] = snr_vector_mean
        df['onlineSNRmax'] = snr_vector_max
    except ValueError:
        print('BIG ERROR!!! sizes dont match for SNR and basics')
    if to_plot:
        plot_results(folder_plots, df, single_animal=False, mode='SNR')
        
    df.to_hdf(to_save_df, key='df', mode='w')
        
    
    # XGBOOOOOST MADAFACA!
    df['ITPTlabel'] = pd.to_numeric(df['ITPTlabel'])
    labels_to_study = [columns[3]] +  columns[10:].tolist()
    X_df = df.loc[:, labels_to_study]
    Y_df = df.iloc[:, 7]
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_df, label=Y_df), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_df.iloc[0,:], matplotlib=True)
    shap.summary_plot(shap_values, X_df)
    shap.summary_plot(shap_values, X_df, plot_type="bar")
    
    
        
        



    
    
