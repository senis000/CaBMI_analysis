
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import copy
import seaborn as sns
from matplotlib import interactive
import utils_cabmi as uc
import xgboost
import shap
from matplotlib.ticker import LogitLocator

interactive(True)



def basic_entry (folder, animal, day):
    '''
    to generate an entry to the pd dataframe with the basic features
    Needs to have a folder with all_processed together
    '''
    
    file_template = "full_{}_{}__data.hdf5"
    file_name = os.path.join(folder, animal, file_template.format(animal, day))
    folder_proc = os.path.join(folder, 'processed')
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
        depth_min = np.nanmin(com_ens[:,2])
        
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
            dist_min = np.nanmin(auxdist)
    
        
            # diff of depth
            auxddepth = []
            for nn in np.arange(ens_neur.shape[0]):
                for nns in np.arange(nn+1, ens_neur.shape[0]):
                    auxddepth.append(scipy.spatial.distance.euclidean(com_ens[nn,2], com_ens[nns,2]))
            
            diffdepth_mean = np.nanmean(auxddepth)
            diffdepth_max = np.nanmax(auxddepth) 
            diffdepth_min = np.nanmin(auxddepth)   
            
            # diff in xy
            auxdxy = []
            for nn in np.arange(ens_neur.shape[0]):
                for nns in np.arange(nn+1, ens_neur.shape[0]):
                    auxdxy.append(scipy.spatial.distance.euclidean(com_ens[nn,:2], com_ens[nns,:2]))
            
            diffxy_mean = np.nanmean(auxdxy)
            diffxy_max = np.nanmax(auxdxy)  
            diffxy_min = np.nanmin(auxdxy)  
        else:
            dist_mean = np.nan
            dist_max = np.nan
            dist_min = np.nan
            diffdepth_mean = np.nan
            diffdepth_max = np.nan
            diffdepth_min = np.nan
            diffxy_mean = np.nan
            diffxy_max = np.nan
            diffxy_min = np.nan
        
        # dynamic range
        # there are some dff that are crazy weird, to avoid them for deteriorating the dataset any std>1 will be ignored
        auxpostwhostd = []
        for nn in np.arange(dff_ens.shape[0]):
            aux_std = np.nanstd(dff_ens[nn,:])
            if aux_std<1:
                auxpostwhostd.append(aux_std)
            
        post_whole_std_mean = np.nanmean(auxpostwhostd)
        post_whole_std_max = np.nanmax(auxpostwhostd)
        post_whole_std_min = np.nanmin(auxpostwhostd)
        
        auxpostbasestd = []
        for nn in np.arange(dff_ens.shape[0]):
            aux_std = np.nanstd(dff_ens[nn,:startBMI])
            if aux_std<1:
                auxpostbasestd.append(aux_std)
            
        post_base_std_mean = np.nanmean(auxpostbasestd)
        post_base_std_max = np.nanmax(auxpostbasestd)
        post_base_std_min = np.nanmin(auxpostbasestd)
    
    else:
        depth_mean = np.nan
        depth_max = np.nan
        depth_min = np.nan
        dist_mean = np.nan
        dist_max = np.nan
        dist_min = np.nan
        diffdepth_mean = np.nan
        diffdepth_max = np.nan
        diffdepth_min = np.nan
        diffxy_mean = np.nan
        diffxy_max = np.nan
        diffxy_min = np.nan
        post_whole_std_mean = np.nan
        post_whole_std_max = np.nan
        post_whole_std_min = np.nan
        post_base_std_mean = np.nan
        post_base_std_max = np.nan
        post_base_std_min = np.nan
    
    auxonstd = []
    for nn in np.arange(online_data.shape[1]):
        pseudo_dff = (online_data[:,nn] - np.nanmean(online_data[:,nn]))/online_data[:,nn]
        auxonstd.append(np.nanstd(pseudo_dff))
        
    onstd_mean = np.nanmean(auxonstd)
    onstd_max = np.nanmax(auxonstd)
    onstd_min = np.nanmin(auxonstd)

    auxcursorstd = []
    cursor_std = np.nanstd(cursor)
    
    row_entry = np.asarray([depth_mean, depth_max, depth_min, dist_mean, dist_max, dist_min, diffdepth_mean, diffdepth_max, \
                             diffdepth_min, diffxy_mean, diffxy_max, diffxy_min, \
                             onstd_mean, onstd_max, onstd_min, post_whole_std_mean, post_whole_std_max, post_whole_std_min, \
                             post_base_std_mean, post_base_std_max, post_base_std_min, cursor_std])
    
    return row_entry, ens_neur


def plot_results(folder_plots, df_aux, first_ind=0, single_animal=True, mode='basic'):
    '''
    Function to plot all the basic results in terms of linear regression 
    '''
    columns = df_aux.columns.tolist()
    # depth mean
    columns_ler = columns[4:10]

    if mode =='basic':
        columns_aux = columns[10:32]  ## basic
        figsiz = (12, 20)
        sbx = 5
        sby = 5
    elif mode == 'SNR':
        columns_aux = columns[32:34]  ## SNR
        figsiz = (8, 4)
        sbx = 1
        sby = 2
    elif mode == 'ce':
        columns_aux = [columns[34]]  ## SNR
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
    '''
    create pandas dataframe to include all the results together and save it
    folder_main = main folder where everything will be stored in subfolders
    file_csv = excel file where the learning results were stored
    to_plot = boolean to plot or not
    saves df
    '''
    # ineficient way to create the dataframe, but I want to be f* sure that each entry is correct.
    folder = os.path.join(folder_main, 'processed')
    folder_plots = os.path.join(folder_main, 'plots', 'learning_regressions')
    folder_snr = os.path.join(folder_main, 'onlineSNR')
    to_save_df = os.path.join(folder_main, 'df_all.hdf5')
    to_load_pick = os.path.join(folder_main, 'cursor_engagement.p')
    animals = os.listdir(folder)
    df_results = pd.read_csv(file_csv)
    df_ce = pd.read_pickle(to_load_pick)
    columns_res = df_results.columns.tolist()
    columns_basic = ['depth_mean', 'depth_max', 'depth_min', 'dist_mean', 'dist_max', 'dist_min', 'diffdepth_mean', 'diffdepth_max', \
                         'diffdepth_min', 'diffxy_mean', 'diffxy_max', 'diffxy_min', \
                         'onstd_mean', 'onstd_max', 'onstd_min', 'post_whole_std_mean', 'post_whole_std_max', 'post_whole_std_min',
                         'post_base_std_mean', 'post_base_std_max', 'post_base_std_min', 'cursor_std']
    columns = columns_res + columns_basic
    columns.insert(3,'ITPTlabel')
    df = pd.DataFrame(columns=columns)

    # obtain basic features
    print('obtaining basic features!')
    mat_ens_ind = np.zeros((len(animals), 25, 4)) + np.nan
    for aa,animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        mat_animal = np.zeros((len(filenames), len(columns_basic))) + np.nan
        for dd,filename in enumerate(filenames):
            day = filename[-17:-11]
            print ('Analyzing animal: ' + animal + ' day: ' + day)
            try:
                mat_animal[dd, :], ens_neur  = basic_entry (folder, animal, day)
                mat_ens_ind[aa,dd,:len(ens_neur)] = ens_neur
            except OSError:
                print('day: ' + day + ' not obtained. ERRRRROOOOOOOOOOOOOOR')    
                break               
        df_res_animal = df_results.loc[df_results['animal']==animal]
        if len(filenames) != len(df_res_animal):
            print('Experiment number missmatch!!. STOPING')
            break
        if animal[:2] == 'IT':
            aux_label = 0
        elif animal[:2] == 'PT':
            aux_label = 1
        df_res_animal.insert(3, 'ITPTlabel', np.repeat([aux_label], len(df_res_animal)))
        df_basic = pd.DataFrame(mat_animal, columns=columns_basic, index=df_res_animal.index)
        df_animal = df_res_animal.join(df_basic)
        if to_plot:
            plot_results(folder_plots, df_animal,  df_res_animal.index[0])
        df = df.append(df_animal, ignore_index=True)
    if to_plot:
        plot_results(folder_plots, df, single_animal=False)
    df['ITPTlabel'] = pd.to_numeric(df['ITPTlabel'])
    
    # obtain snr features
    print('obtaining snrs')
    snr_vector_mean = []#np.zeros(len(df))
    snr_vector_max = []#np.zeros(len(df))
    snr_vector_min = []#np.zeros(len(df))
    number_snrs = np.zeros(len(animals))
    for aa,animal in enumerate(animals):
        folder_path = os.path.join(folder_snr, animal)
        filenames = os.listdir(folder_path)
        number_snrs[aa] = len(filenames)
        for dd,filename in enumerate(filenames):
            print ('Analyzing animal: ' + animal + ' day: ' + filename)
            f = h5py.File(os.path.join(folder_path, filename), 'r')
            aux_snr = np.asarray(f['SNR_ens'])
            f.close()
            snr_vector_mean.append(np.nanmean(aux_snr))
            snr_vector_max.append(np.nanmax(aux_snr))
            snr_vector_min.append(np.nanmin(aux_snr))
    try:
        df['onlineSNRmean'] = snr_vector_mean
        df['onlineSNRmax'] = snr_vector_max
        df['onlineSNRmin'] = snr_vector_min
    except ValueError:
        print('BIG ERROR!!! sizes dont match for SNR and basics')
    if to_plot:
        plot_results(folder_plots, df, single_animal=False, mode='SNR')
    
    # obtain cursor engagement
    print('obtaining cursor engagement')
    df['cursor_eng'] = np.nan
    for aa,animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        andf = df_ce.loc[df_ce['Animal']==animal]
        for dd,filename in enumerate(filenames):
            day = filename[-17:-11]
            auxr2 = andf[andf['Date']==day]['R2 Values'].values
            if len(auxr2)>0:
                dfind = df.index[(df['animal']==animal)&(df['day']==int(day))]
                df.loc[dfind,'cursor_eng']=auxr2
    if to_plot:
        plot_results(folder_plots, df, single_animal=False, mode='ce')
        

    # obtain connectivity values
    df['GC_raw_ratio_ens_x'] = np.nan
    df['GC_raw_ratio_x_ens'] = np.nan
    df['GC_per_ratio_ens_x'] = np.nan
    df['GC_per_ratio_x_ens'] = np.nan
    df['GC_raw_ens_red'] = np.nan
    df['GC_raw_red_ens'] = np.nan
    df['GC_per_ens_red'] = np.nan
    df['GC_per_red_ens'] = np.nan
    df['GC_raw_ens_ind'] = np.nan
    df['GC_raw_ind_ens'] = np.nan
    df['GC_per_ens_ind'] = np.nan
    df['GC_per_ind_ens'] = np.nan
    
    for aa,animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        for dd,filename in enumerate(filenames): 
            day = filename[-17:-11] 
            dfind = df.index[(df['animal']==animal)&(df['day']==int(day))]
            to_load_FC = os.path.join(folder_main, 'FC', 'statsmodel' , animal, day, 'baseline_red_ens-indirect_dff_order_auto.p')
            try:
                df_FC = pd.read_pickle(to_load_FC)  
            except FileNotFoundError:
                print ('Animal: ' + animal + ' or day: ' + day  + ' doesnt exist on the Granger causality analysis')
                break
            #extract the granger causality values
            FC_red = df_FC['FC_red']
            FC_pval_red = df_FC['FC_pval_red']
            ind_red = df_FC['indices_red']
            FC_ens_indirect = df_FC['FC_ens-indirect']
            FC_indirect_ens = df_FC['FC_indirect-ens']
            FC_pval_ens_indirect = df_FC['FC_pval_ens-indirect']
            FC_pval_indirect_ens = df_FC['FC_pval_indirect-ens']
            ind_ens_ind = df_FC['indices_ens-indirect']
            ind_ind_ens = df_FC['indices_indirect-ens']
            ens_neur = mat_ens_ind[aa,dd,~np.isnan(mat_ens_ind[aa,dd,:])].astype(int)
            ind_mat_red = np.zeros(ens_neur.shape[0], dtype=np.int16)
            for ee, eneur in enumerate(ens_neur):
                ind_mat_red[ee] = np.where(ind_red==eneur)[0][0]
            FC_ens_red = FC_red[ind_mat_red,:]
            FC_pval_ens_red = FC_pval_red[ind_mat_red,:]
            FC_red_ens = FC_red[:,ind_mat_red]
            FC_pval_red_ens = FC_pval_red[:,ind_mat_red]
            # obtain GC values, raw are values GC, per are number of connections with p<0.05
            # obtain red connectivity
            raw_ens_red = np.nanmean(FC_ens_red[FC_pval_ens_red<0.05 ])
            raw_red_ens = np.nanmean(FC_red_ens[FC_pval_red_ens<0.05 ])
            per_ens_red = np.nansum(FC_pval_ens_red<0.05)/np.prod(FC_pval_ens_red.shape)
            per_red_ens = np.nansum(FC_pval_red_ens<0.05)/np.prod(FC_pval_red_ens.shape)
            # obtain green connectivity
            raw_ens_ind = np.nanmean(FC_ens_indirect[FC_pval_ens_indirect<0.05])
            raw_ind_ens = np.nanmean(FC_indirect_ens[FC_pval_indirect_ens<0.05])
            per_ens_ind = np.nansum(FC_pval_ens_indirect<0.05)/np.prod(FC_pval_ens_indirect.shape)
            per_ind_ens = np.nansum(FC_pval_indirect_ens<0.05)/np.prod(FC_pval_indirect_ens.shape)

            # df red-ind
            df.loc[dfind,'GC_raw_ens_red'] = raw_ens_red
            df.loc[dfind,'GC_raw_red_ens'] = raw_red_ens
            df.loc[dfind,'GC_per_ens_red'] = per_ens_red
            df.loc[dfind,'GC_per_red_ens'] = per_red_ens
            df.loc[dfind,'GC_raw_ens_ind'] = raw_ens_ind
            df.loc[dfind,'GC_raw_ind_ens'] = raw_ind_ens
            df.loc[dfind,'GC_per_ens_ind'] = per_ens_ind
            df.loc[dfind,'GC_per_ind_ens'] = per_ind_ens
            # df ratio
            df.loc[dfind,'GC_raw_ratio_ens_x'] = raw_ens_red/raw_ens_ind
            df.loc[dfind,'GC_raw_ratio_x_ens'] = raw_red_ens/raw_ind_ens
            df.loc[dfind,'GC_per_ratio_ens_x'] = per_ens_red/per_ens_ind
            df.loc[dfind,'GC_per_ratio_x_ens'] = per_red_ens/per_ind_ens
            
    # save!
    df.to_hdf(to_save_df, key='df', mode='w')
        
    

    
    
def bootstrap_pandas(len_df, X_df, Y_df, bts_n=1000):
    ''' Bootstrap  pd 
    df = panda dataframe
    n = size of bootstrap matrix (how many values)

    returns dfbst
    '''
    
    if bts_n == 0:
        bts_n = len_df
        
    bootst_ind = np.floor(np.random.rand(bts_n)*len_df).astype(int)
    X_df_bst = X_df.iloc[bootst_ind]
    Y_df_bst = Y_df.iloc[bootst_ind]
    
    #dfbst = df.iloc[bootst_ind]
    return bootst_ind, X_df_bst, Y_df_bst


def split_df(df, bts_n=1000, learn_stat_colum='totalPC', size_split_test=0.2):
    '''
    Function to calculate the xgboost model
    '''
    # select X and Y
    columns = df.columns
    labels_to_study = [columns[3]] +  columns[10:].tolist()
    X_df = df.loc[:, labels_to_study]
    Y_df = df.loc[:, learn_stat_colum]
    if np.isinf(np.nansum(Y_df)):
        X_df = X_df[~np.isinf(Y_df)]
        Y_df = Y_df[~np.isinf(Y_df)]
    if np.sum(np.isnan(Y_df))>0:
        X_df = X_df[~np.isnan(Y_df)]
        Y_df = Y_df[~np.isnan(Y_df)]
    
    # split in train/test
    X_df_train, X_df_test, Y_df_train, Y_df_test = train_test_split(X_df, Y_df, test_size=size_split_test)
   
    return X_df_train, X_df_test, Y_df_train, Y_df_test
    
    
def calculate_model(X_df, Y_df, learning_rate=0.1, xgboost_rep=100):
    
    model = xgboost.train({"learning_rate": learning_rate}, xgboost.DMatrix(X_df, label=Y_df), xgboost_rep)
    
    return model

    
def calculate_bst632 (model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test):
    '''
    function to calculate the error of the bootstrap by rule 0.632
    '''
    
    # calculate errors
    error_samp = mean_squared_error(model.predict(xgboost.DMatrix(X_df_train_bst, label=Y_df_train_bst)), Y_df_train_bst)
    error_test = mean_squared_error(model.predict(xgboost.DMatrix(X_df_test, label=Y_df_test)), Y_df_test)
    error_bst = 0.632*error_test + 0.368*error_samp
    
    return error_bst

    
def calculate_bst_length_optimal(df, rep=100, bst_num=None):
    '''
    function to calculate the best length for the bootstrap depending on the rule 0.632
    '''
    # some values for HPM
    #0.200 for 40/60 split (minimum error 0.15)
    #0.204 for 20/80 (0.133 for error<0.15)
    #0.188 for 1000 xgboost_rep (0.124 for error<0.15) 20/80
    #0.190 for learning 0.1 (0.123 for error<0.15)
    #0.198 for learning 0.3 (0.126 for error<0.15)

    if bst_num == None:
        bst_num = np.arange(0,10000,1000)
        bst_num[0] = 100
    error_bst = np.zeros((rep,bst_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, bn in enumerate(bst_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, bn)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            error_bst[rr,ind] = calculate_bst632 (model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)  
    return error_bst

            
def calculate_size_train_optimal(df, rep=100, size_num=None):
    '''
    function to calculate the best size for the XGboost depending on the rule 0.632
    '''

    if size_num == None:
        size_num = np.arange(0.1,0.6,0.1)
    error_bst = np.zeros((rep,size_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, sn in enumerate(size_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, size_split_test=sn)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n, )
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            error_bst[rr,ind] = calculate_bst632 (model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst    


def calculate_learning_optimal(df, rep=100, learn_num=None):
    '''
    function to calculate the learning_rate for the XGboost depending on the rule 0.632
    '''

    if learn_num == None:
        learn_num = np.arange(0,0.4,0.05)
        learn_num[0] = 0.01
    error_bst = np.zeros((rep,learn_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, ln in enumerate(learn_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst, learning_rate=ln)
            error_bst[rr,ind] = calculate_bst632 (model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst   


def calculate_xgbrep_optimal(df, rep=100, xgrep_num=None):
    '''
    function to calculate the best number of trees for the XGboost depending on the rule 0.632
    '''

    if xgrep_num == None:
        xgrep_num = np.asarray([20,50,100,200,500,1000,2000])
    error_bst = np.zeros((rep,xgrep_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, xn in enumerate(xgrep_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst, xgboost_rep=xn)
            error_bst[rr,ind] = calculate_bst632 (model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst  


def calculate_learn_stat_optimal(df, rep=100, bts_n=1000):
    '''
    function to calculate the error for each learning stat for the XGboost depending on the rule 0.632
    '''
    columns = df.columns.tolist()
    columns_ler = columns[4:10]
    error_bst = np.zeros((rep,len(columns_ler))) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, cn in enumerate(columns_ler):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, learn_stat_colum=cn)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            error_bst[rr,ind] = calculate_bst632 (model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst  


def calculate_learn_stat_mseoptimal(df, rep=100, bts_n=1000):
    '''
    function to calculate the error for each learning stat for the XGboost depending on the rule 0.632
    '''
    columns = df.columns.tolist()
    columns_ler = columns[4:10]
    error_bst = np.zeros((rep,len(columns_ler))) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, cn in enumerate(columns_ler):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, learn_stat_colum=cn)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            aux_predict = model.predict(xgboost.DMatrix(X_df_test, label=Y_df_test))
            error_bst[rr,ind] = mean_squared_error(Y_df_test, aux_predict)
    return error_bst 


def calculate_all_errors(df, folder_main, rep=100):
    '''
    function to calculate and store all the errors to obtain the optimal parameters for the model
    '''
    f = h5py.File(os.path.join(folder_main, 'plots', 'utils', 'tpc_error_optimal.h5py'), 'w-')
    error_length = calculate_bst_length_optimal(df, rep)
    error_size = calculate_size_train_optimal(df, rep)
    error_learning = calculate_learning_optimal(df, rep)
    error_xgrep = calculate_xgbrep_optimal(df, rep)
    error_learn_stat = calculate_learn_stat_optimal(df, rep)
    f.create_dataset('error_length', data = error_length) 
    f.create_dataset('error_size', data = error_size) 
    f.create_dataset('error_learning', data = error_learning) 
    f.create_dataset('error_xgrep', data = error_xgrep) 
    f.close()
    

def obtain_shap_iter(df, folder_main, bts_n=1000, mod_n=1000, mod_x=100, error_bstmax=[0.02,0.2], \
                     error_msemax=[0.03,0.3], size_split_test=0.2, max_iter=40, stability_var=0.7, toplot=True):
    '''
    obtain shap values of mod_n different XGboost model if the conditions for error of the model and stability are set
    obtain stabitlity of feature: correlation of original shap values and bootstrap values to see if values are miningful or noise
    '''
    columns = df.columns.tolist()
    columns_ler = columns[6:8] # columns[4:10] #[columns[6]]# 
    labels_to_study = [columns[3]] +  columns[10:]
    
    test_size = np.ceil(len(df)*(size_split_test)).astype(int)
    train_size = np.floor(len(df)*(1-size_split_test)).astype(int)
    
    all_shap = np.zeros((len(columns_ler), mod_n, test_size, len(labels_to_study))) + np.nan
    all_y_pred = np.zeros((len(columns_ler), mod_n, test_size)) + np.nan
    all_mse = np.zeros((len(columns_ler), mod_n)) + np.nan
    shap_correlations = np.zeros((len(columns_ler), mod_n, mod_x, len(labels_to_study))) + np.nan
    explainer_val = np.zeros((len(columns_ler), mod_n)) + np.nan
    all_df = np.zeros((len(columns_ler), mod_n, test_size)) + np.nan
    
    for cc, col_ler in enumerate(columns_ler):
        i = 0
        iteri = 0
        while i < mod_n:
            if iteri < max_iter:
                all_shap_train_aux = np.zeros((mod_x+1, train_size, len(labels_to_study))) + np.nan
                shap_cor_aux = np.zeros((mod_x, len(labels_to_study))) + np.nan
                j = 1
                iterj = 0   
                # make splits for original model
                X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, bts_n, col_ler, size_split_test=size_split_test)
                # calculate original
                model_original = calculate_model(X_df_train, Y_df_train)
                
                # first check for the model (using bst632)
                error_bst = calculate_bst632 (model_original, X_df_train, X_df_test, Y_df_train, Y_df_test)
                # second check for the model (using mse)
                aux_predict = model_original.predict(xgboost.DMatrix(X_df_test, label=Y_df_test))
                aux_mse = mean_squared_error(Y_df_test, aux_predict)
                
                # if the model is good enough (mse/bst error low)               
                if (error_bst  < error_bstmax[cc]) & (aux_mse < error_msemax[cc]):
                    explainer_train = shap.TreeExplainer(model_original, data=X_df_test, feature_perturbation='interventional')
                    # just in case the size differs we will take only the size of X_df
                    all_shap_train_aux[0, :len(X_df_train), :] = explainer_train.shap_values(X_df_train)
                    
                    # bootstrap check for stability of features
                    while j<(mod_x+1):
                        if iterj > max_iter:
                            print('too many iterations, check that the maximum error is not too restrictive or split was just bad')
                            break
                        print('repetition: ' + str(i) + ':' + str(j) + ', with iterj: ' + str(iterj))
                        _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
                        model_bst = calculate_model(X_df_train_bst, Y_df_train_bst)
                        error_bst = calculate_bst632 (model_bst, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
                        if error_bst  < error_bstmax[cc]:
                            explainer_bst = shap.TreeExplainer(model_bst, data=X_df_test, feature_perturbation='interventional')
                            all_shap_train_aux[j, :len(X_df_train), :] = explainer_bst.shap_values(X_df_train)
                            
                            # check correlation of features with features from original model
                            for ll, label_ts in enumerate(labels_to_study):
                                shap_cor_aux[j-1, ll] = np.corrcoef(all_shap_train_aux[0, :len(X_df_train), ll], \
                                                                    all_shap_train_aux[j, :len(X_df_train), ll])[0,1]
                            j += 1
                            iterj = 0
                        else:
                            iterj += 1 
                
                    aaux = np.nanmean(all_shap_train_aux,0)  
                    
                    # if the model had a relatively low error and the features were not only noise
                    if (np.nansum(aaux) != 0) & (np.nanmean(shap_cor_aux)>stability_var):
                        # first store everything we used to calculate stability
                        shap_correlations[cc, i, :, :] = shap_cor_aux
                        
                        # keep info of the xgboost performance
                        all_y_pred[cc, i, :len(X_df_test)] = aux_predict
                        all_mse[cc,i] = aux_mse
                        
                        # now calculate the shap_values with the X_df_test FINALLY!
                        explainer_test = shap.TreeExplainer(model_original, data=X_df_train, feature_perturbation='interventional')
                        all_shap[cc, i, :len(X_df_test), :] = explainer_test.shap_values(X_df_test)
                        explainer_val[cc, i] = explainer_test.expected_value
                        ind_aux = np.zeros(X_df_test.shape[0], dtype=np.int16) 
                        for xx in np.arange(X_df_test.shape[0]):
                            ind_aux[xx] = df.index.get_loc(X_df_test.index[xx])
                        all_df[cc,i,:len(X_df_test)] = ind_aux
                        
                        # update
                        i+=1 
                        iteri = 0
                        print('new model processed: ' + str(i) + '/' + str(mod_n))

                    else:
                        print('model not up to specs, repeting model. Iteri: ' + str(iteri) )
                        iteri += 1
                else:
                    print('too high bst ' + str(~(error_bst  < error_bstmax[cc])) + ' or mse error ' + \
                           str(~(aux_mse < error_msemax[cc]))  + ', repeting split. Iteri: ' + str(iteri))
                    iteri += 1
                    
            else:
                print('Error too high to continue. Maxiter reached for ' + col_ler)
                print('   ')
                print('I REPEAT!!!!! Error to high to continue. Maxiter reached')
                break
            
            
    # check mean shap values.
    all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
    all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])]) 
    
    shap_experiment_mean = np.zeros((len(columns_ler), len(df), len(labels_to_study))) + np.nan 
    shap_experiment_std = np.zeros((len(columns_ler), len(df), len(labels_to_study))) + np.nan 
    shap_experiment_sem = np.zeros((len(columns_ler), len(df), len(labels_to_study))) + np.nan 
    bins_zscore = np.arange(-2,2,0.1)
    spread = np.zeros((len(columns_ler), len(df), len(labels_to_study), len(bins_zscore)-1)) + np.nan 
    for cc, col_ler in enumerate(columns_ler):
        for ind in np.arange(len(df)):
            aux_ind = np.where(all_df_reshape[cc,:]==ind)[0]
            if len(aux_ind) > 0:
                aux_shap = all_shap_reshape[cc,aux_ind,:]
                shap_experiment_mean[cc,ind,:] = np.nanmean(aux_shap,0)
                shap_experiment_std[cc,ind,:] = np.nanstd(aux_shap,0)
                shap_experiment_sem[cc,ind,:] = np.nanstd(aux_shap,0)/np.sqrt(len(aux_ind))
                for ll, label_ts in enumerate(labels_to_study):
                    [h,b] = np.histogram(zscore(aux_shap[:,ll]), bins_zscore)
                    spread[cc,ind,ll,:] = h
    # the spread was gaussian
    
    
    # lets get plotting!!!
    
    if toplot:
        sizesubpl = np.ceil(len(labels_to_study)/6).astype('int')
        
        # check stability of features
        folder_plots_sta = os.path.join(folder_main, 'plots', 'XGBoost', 'feature_stability')
        bins_cor = np.arange(stability_var-0.2,1,0.01)
        for cc, col_ler in enumerate(columns_ler):
            fig1 = plt.figure(figsize=(17,9))
            for ll, label_ts in enumerate(labels_to_study):
                ax0 = fig1.add_subplot(sizesubpl, 6, ll+1)
                [h,b] = np.histogram(shap_correlations[cc,:,:,ll], bins_cor)
                ax0.bar(b[1:], h, width=0.01)
                ax0.set_xlabel(label_ts)
                
            fig1.tight_layout()
            fig1.savefig(os.path.join(folder_plots_sta, col_ler + '_stability_features.png'), bbox_inches="tight")
            fig1.savefig(os.path.join(folder_plots_sta, col_ler + '_stability_features.eps'), bbox_inches="tight")
            plt.close('all')
    
    
        # check IT/PT shap values
        folder_plots_ITPT = os.path.join(folder_main, 'plots', 'XGBoost', 'ITPT')
        all_IT = np.zeros(len(columns_ler)) + np.nan
        all_PT = np.zeros(len(columns_ler)) + np.nan
        bins_shap = np.arange(-0.05,0.05,0.001)
        
        
        for cc, col_ler in enumerate(columns_ler):
            aux_IT = shap_experiment_mean[cc,df['ITPTlabel']==0,0]
            aux_PT = shap_experiment_mean[cc,df['ITPTlabel']==1,0]
            all_IT[cc] = np.nanmean(aux_IT)
            all_PT[cc] = np.nanmean(aux_PT)
            
            fig2 = plt.figure(figsize=(12,4))
            ax1 = fig2.add_subplot(1, 2, 1)
            [h_IT,b] = np.histogram(aux_IT, bins_shap)
            [h_PT,b] = np.histogram(aux_PT, bins_shap)
            ax1.bar(b[1:], h_IT, width=0.001, label='IT')
            ax1.bar(b[1:], h_PT, width=0.001, label='PT')
            ax1.legend()
            
            ax2 = fig2.add_subplot(1, 2, 2)
            ax2.bar([0.4,1.4], [all_IT[cc], all_PT[cc]], width=0.8, \
                    yerr=[pd.DataFrame(aux_IT).sem(0).values[0], pd.DataFrame(aux_PT).sem(0).values[0]], \
                    error_kw=dict(ecolor='k'))
            ax2.set_xticks([0.4,1.4])
            ax2.set_xticklabels(['IT', 'PT'])
            _, p_value = stats.ttest_ind(aux_IT, aux_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.8, 0.008, p, color='grey', alpha=0.6)
            ax2.set_ylim([-0.03, 0.03])
            
            fig2.savefig(os.path.join(folder_plots_ITPT, col_ler + '_ITPT_shap_val.png'), bbox_inches="tight")
            fig2.savefig(os.path.join(folder_plots_ITPT, col_ler + '_ITPT_shap_val.eps'), bbox_inches="tight")
            plt.close('all')
            
        
        #check shap summary plot
        folder_plots_shap = os.path.join(folder_main, 'plots', 'XGBoost', 'shap')
        bins_shap = np.arange(-0.05,0.05,0.001)      

        for cc, col_ler in enumerate(columns_ler):
            fig3 = plt.figure()
            aux_df = np.where(~np.isnan(np.sum(shap_experiment_mean[cc,:,:],1)))[0]
            shap.summary_plot(shap_experiment_mean[cc,aux_df,:], df.iloc[aux_df][labels_to_study])
            
            fig3.savefig(os.path.join(folder_plots_shap, col_ler + '_summary.png'), bbox_inches="tight")
            fig3.savefig(os.path.join(folder_plots_shap, col_ler + '_summary.eps'), bbox_inches="tight")
            plt.close('all')
            
            
        # check dependencies
        folder_plots_depend = os.path.join(folder_main, 'plots', 'XGBoost', 'dependencies')
        sizesubpl = np.ceil(len(labels_to_study)/6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            for ll, label_ts in enumerate(labels_to_study):
                fig4 = plt.figure(figsize=(17,9))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax3 = fig4.add_subplot(sizesubpl, 6, llsec+1)
                    aux_df = np.where(~np.isnan(np.sum(shap_experiment_mean[cc,:,:],1)))[0]
                    shap.dependence_plot(label_ts, shap_experiment_mean[cc,aux_df,:], df.iloc[aux_df][labels_to_study], interaction_index=llsec, ax=ax3)
                fig4.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig4.savefig(os.path.join(folder_plots_depend, col_ler + '_' + label_ts + '_depen.png'), bbox_inches="tight")
                fig4.savefig(os.path.join(folder_plots_depend, col_ler + '_' + label_ts + '_depen.eps'), bbox_inches="tight")
                plt.close('all')
                
                
        # check regression inter feature       
        folder_plots_reg_feat = os.path.join(folder_main, 'plots', 'XGBoost', 'regression_feat')
        sizesubpl = np.ceil(len(labels_to_study)/6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            for ll, label_ts in enumerate(labels_to_study):
                fig5 = plt.figure(figsize=(17,9))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax4 = fig5.add_subplot(sizesubpl, 6, llsec+1)
                    sns.regplot(df[label_ts], df[label_tsec], ax=ax4)
                    ax4.set_ylabel(label_tsec)
                fig5.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig5.savefig(os.path.join(folder_plots_reg_feat, col_ler + '_' + label_ts + '_regfet.png'), bbox_inches="tight")
                fig5.savefig(os.path.join(folder_plots_reg_feat, col_ler + '_' + label_ts + '_regfet.eps'), bbox_inches="tight")
                plt.close('all')                
                
        
        # labels to study regression to shap
        folder_plots_reg = os.path.join(folder_main, 'plots', 'XGBoost', 'regression_shap')
        sizesubpl = np.ceil(len(labels_to_study)/6).astype('int')

        for cc, col_ler in enumerate(columns_ler):
            fig6 = plt.figure(figsize=(17,9))
            for ll, label_ts in enumerate(labels_to_study):
                ax6 = fig6.add_subplot(sizesubpl, 6, ll+1)
                aux_df = np.where(~np.isnan(np.sum(shap_experiment_mean[cc,:,:],1)))[0]
                sns.regplot(df.iloc[aux_df][label_ts], shap_experiment_mean[cc,aux_df,ll], ax=ax6)
                ax6.set_ylabel('shap_val')
            fig6.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig6.savefig(os.path.join(folder_plots_reg, col_ler + '_reg.png'), bbox_inches="tight")
            fig6.savefig(os.path.join(folder_plots_reg, col_ler + '_reg.eps'), bbox_inches="tight")
            plt.close('all')
            
            
        # check for confidence interval on features
        folder_plots_ci = os.path.join(folder_main, 'plots', 'XGBoost', 'confidence_interval')
        for cc, col_ler in enumerate(columns_ler):
            fig7 = plt.figure(figsize=(17,9))
            for ll, label_ts in enumerate(labels_to_study):
                ax7 = fig7.add_subplot(sizesubpl, 6, ll+1)
                aux_spread = np.nansum(spread[cc,:,ll,:],0)
                ax7.bar(bins_zscore[1:], aux_spread)
                ax7.set_ylabel(label_ts)
            fig7.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig7.savefig(os.path.join(folder_plots_ci, col_ler + '_ci.png'), bbox_inches="tight")
            fig7.savefig(os.path.join(folder_plots_ci, col_ler + '_ci.eps'), bbox_inches="tight")
            plt.close('all')
                
                
            
            
        # check for groups of labels
#         groups_labels = ['ITPT', 'Pos', 'STD', 'SNR']       
#         shap_group = np.stack((shap_experiment_mean[:,:,0], np.nansum(shap_experiment_mean[:,:,1:9],2), \
#                                          np.nansum(shap_experiment_mean[:,:,9:16],2), np.nansum(shap_experiment_mean[:,:,16:],2)),axis=2) 
#         bins_grshap = np.arange(-0.1,0.1,0.001)      
#         for cc, col_ler in enumerate(columns_ler):
#             for gr, group in enumerate(groups_labels):
#                 [h,b] = np.histogram(shap_group[cc,:,gr], bins_grshap)
#                 plt.bar(b[1:], h, width=0.001, label=group)
#             plt.legend()
#             plt.close('all')

        
        
        
        #************************************************************************************************************************
        # check for all models together! ***************************************************************************************
        #**************************************************************************************************************************
        folder_plots_ITPTm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'ITPT')
        all_ITm = np.zeros((len(columns_ler), mod_n)) + np.nan
        all_PTm = np.zeros((len(columns_ler), mod_n)) + np.nan
        for cc, col_ler in enumerate(columns_ler):
            aux_ind_IT = np.zeros((mod_n, all_shap.shape[2]), dtype='bool')
            aux_ind_PT = np.zeros((mod_n, all_shap.shape[2]), dtype='bool')
            for i in np.arange(mod_n):
                aux_df = all_df[cc,i,~np.isnan(all_df[cc,i,:])]
                aux_ind_IT[i, :aux_df.shape[0]] = df.iloc[aux_df]['ITPTlabel']==0
                aux_ind_PT[i, :aux_df.shape[0]] = df.iloc[aux_df]['ITPTlabel']==1
                all_ITm[cc,i] = np.nanmean(all_shap[cc,i,aux_ind_IT[i,:],0])
                all_PTm[cc,i] = np.nanmean(all_shap[cc,i,aux_ind_PT[i,:],0])
            
            fig2 = plt.figure(figsize=(12,4))
            ax1 = fig2.add_subplot(1, 2, 1)
            [h_IT,b] = np.histogram(all_shap[cc,aux_ind_IT,0], bins_shap)
            [h_PT,b] = np.histogram(all_shap[cc,aux_ind_PT,0], bins_shap)
            ax1.bar(b[1:], h_IT, width=0.001, label='IT')
            ax1.bar(b[1:], h_PT, width=0.001, label='PT')
            ax1.legend()
            
            ax2 = fig2.add_subplot(1, 2, 2)
            ax2.bar([0.4,1.4], [np.nanmean(all_ITm[cc,:]), np.nanmean(all_PTm[cc,:])], width=0.8, \
                    yerr=[pd.DataFrame(all_ITm[cc,:]).sem(0).values[0], pd.DataFrame(all_PTm[cc,:]).sem(0).values[0]], \
                    error_kw=dict(ecolor='k'))
            ax2.set_xticks([0.4,1.4])
            ax2.set_xticklabels(['IT', 'PT'])
            _, p_value = stats.ttest_ind(all_ITm[cc,:], all_PTm[cc,:], nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.8, 0.008, p, color='grey', alpha=0.6)
            ax2.set_ylim([-0.03, 0.03])
            
            fig2.savefig(os.path.join(folder_plots_ITPTm, col_ler + '_ITPT_shap_val.png'), bbox_inches="tight")
            fig2.savefig(os.path.join(folder_plots_ITPTm, col_ler + '_ITPT_shap_val.eps'), bbox_inches="tight")
            plt.close('all')
            
        
        folder_plots_shapm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'shap')
        bins_shap = np.arange(-0.05,0.05,0.001)      

        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc,:])
            aux_df = all_df_reshape[cc,aux_ind] 
            
            shap_reshape = all_shap_reshape[cc,aux_ind,:]
            fig3 = plt.figure()
            shap.summary_plot(shap_reshape, df.iloc[aux_df][labels_to_study])
            
            fig3.savefig(os.path.join(folder_plots_shapm, col_ler + '_summary.png'), bbox_inches="tight")
            fig3.savefig(os.path.join(folder_plots_shapm, col_ler + '_summary.eps'), bbox_inches="tight")
            plt.close('all')
            
            
        # check dependencies
        all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
        all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])])  
         
        folder_plots_dependm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'dependencies')
        sizesubpl = np.ceil(len(labels_to_study)/6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc,:])
            aux_df = all_df_reshape[cc,aux_ind] 
            shap_reshape = all_shap_reshape[cc,aux_ind,:]
            for ll, label_ts in enumerate(labels_to_study):
                fig4 = plt.figure(figsize=(17,9))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax3 = fig4.add_subplot(sizesubpl, 6, llsec+1)
                    shap.dependence_plot(label_ts, shap_reshape, df.iloc[aux_df][labels_to_study], interaction_index=llsec, ax=ax3)
                fig4.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig4.savefig(os.path.join(folder_plots_dependm, col_ler + '_' + label_ts + '_depen.png'), bbox_inches="tight")
                fig4.savefig(os.path.join(folder_plots_dependm, col_ler + '_' + label_ts + '_depen.eps'), bbox_inches="tight")
                plt.close('all')
                
                
        # check regression inter feature
        all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
        all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])])  
         
        folder_plots_reg_featm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'regression_feat')
        sizesubpl = np.ceil(len(labels_to_study)/6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc,:])
            aux_df = all_df_reshape[cc,aux_ind] 
            shap_reshape = all_shap_reshape[cc,aux_ind,:]
            for ll, label_ts in enumerate(labels_to_study):
                fig5 = plt.figure(figsize=(17,9))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax4 = fig5.add_subplot(sizesubpl, 6, llsec+1)
                    sns.regplot(df.iloc[aux_df][label_ts], df.iloc[aux_df][label_tsec], ax=ax4)
                    ax4.set_ylabel(label_tsec)
                fig5.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig5.savefig(os.path.join(folder_plots_reg_featm, col_ler + '_' + label_ts + '_regfet.png'), bbox_inches="tight")
                fig5.savefig(os.path.join(folder_plots_reg_featm, col_ler + '_' + label_ts + '_regfet.eps'), bbox_inches="tight")
                plt.close('all')                
                
        
        # labels to study regression to shap
        all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
        all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])])  
         
        folder_plots_regm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'regression_shap')
        sizesubpl = np.ceil(len(labels_to_study)/6).astype('int')

        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc,:])
            aux_df = all_df_reshape[cc,aux_ind] 
            shap_reshape = all_shap_reshape[cc,aux_ind,:]
            fig6 = plt.figure(figsize=(17,9))
            for ll, label_ts in enumerate(labels_to_study):
                ax6 = fig6.add_subplot(sizesubpl, 6, ll+1)
                sns.regplot(df.iloc[aux_df][label_ts], shap_reshape[:,ll], ax=ax6)
                ax6.set_ylabel('shap_val')
            fig6.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig6.savefig(os.path.join(folder_plots_regm, col_ler + '_reg.png'), bbox_inches="tight")
            fig6.savefig(os.path.join(folder_plots_regm, col_ler + '_reg.eps'), bbox_inches="tight")
            plt.close('all')
            
            
#         # check for groups of labels
#         groups_labels = ['ITPT', 'Pos', 'STD', 'SNR']
#         all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
#         all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])])  
#         
#         all_shap_group = np.stack((all_shap_reshape[:,:,0], np.nansum(all_shap_reshape[:,:,1:9],2), \
#                                          np.nansum(all_shap_reshape[:,:,9:16],2), np.nansum(all_shap_reshape[:,:,16:],2)),axis=2) 
#         bins_grshap = np.arange(-0.1,0.1,0.001)      
#         for cc, col_ler in enumerate(columns_ler):
#             shap_reshape = all_shap_group[cc,:,:]
#             for gr, group in enumerate(groups_labels):
#                 [h,b] = np.histogram(all_shap_group[cc,:,gr], bins_grshap)
#                 plt.bar(b[1:], h, width=0.01, label=group)
#             plt.legend()
            

        
        
        
        
            
        
            