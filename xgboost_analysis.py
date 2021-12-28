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


def basic_entry(folder, animal, day, df_e2, speed=False):
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
    e2_neur = np.asarray(df_e2[animal][day])
    e1_neur = copy.deepcopy(ens_neur)

    online_data = np.asarray(f['online_data'])[:, 2:]
    dff = np.asarray(f['dff'])
    cursor = np.asarray(f['cursor'])

    startBMI = np.asarray(f['trial_start'])[0]
    f.close()
    if np.isnan(np.sum(ens_neur)):
        print('Only ' + str(4 - np.sum(np.isnan(ens_neur))) + ' ensemble neurons')
    if np.isnan(np.sum(e2_neur)):
        print('Only ' + str(2 - np.sum(np.isnan(e2_neur))) + ' e2 neurons')

    if np.nansum(e2_neur) > 0:
        for i in np.arange(len(e2_neur)):
            e1_neur[np.where(ens_neur == e2_neur[i])[0]] = np.nan

    e1_neur = np.int16(e1_neur[~np.isnan(e1_neur)])
    ens_neur = np.int16(ens_neur[~np.isnan(ens_neur)])
    e2_neur = np.int16(e2_neur[~np.isnan(e2_neur)])

    if len(ens_neur) > 0:
        com_ens = com[ens_neur, :]
        com_e2 = com[e2_neur, :]
        com_e1 = com[e1_neur, :]
        dff_ens = dff[ens_neur, :]

        # xyz position
        depth_mean = np.nanmean(com_ens[:, 2])
        depth_max = np.nanmax(com_ens[:, 2])
        depth_min = np.nanmin(com_ens[:, 2])

        #     if len(e2_neur) > 0:
        #         depth_mean_e2 = np.nanmean(com_e2[:,2])
        #         depth_max_e2 = np.nanmax(com_e2[:,2])
        #     else:
        #         depth_mean_e2 = np.nan
        #         depth_max_e2 = np.nan

        # distance
        if ens_neur.shape[0] > 1:
            if (e1_neur.shape[0] > 0) & (e2_neur.shape[0] > 0):
                auxdist = []
                for nne2 in np.arange(e2_neur.shape[0]):
                    for nne1 in np.arange(e1_neur.shape[0]):
                        auxdist.append(scipy.spatial.distance.euclidean(com_e2[nne2, :], com_e1[nne1, :]))

                dist_mean = np.nanmean(auxdist)
                dist_max = np.nanmax(auxdist)
                dist_min = np.nanmin(auxdist)
            else:
                dist_mean = np.nan
                dist_max = np.nan
                dist_min = np.nan

            # diff of depth
            auxddepth = []
            for nn in np.arange(ens_neur.shape[0]):
                for nns in np.arange(nn + 1, ens_neur.shape[0]):
                    auxddepth.append(scipy.spatial.distance.euclidean(com_ens[nn, 2], com_ens[nns, 2]))

            diffdepth_mean = np.nanmean(auxddepth)
            diffdepth_max = np.nanmax(auxddepth)
            diffdepth_min = np.nanmin(auxddepth)

            # diff in xy
            auxdxy = []
            for nn in np.arange(ens_neur.shape[0]):
                for nns in np.arange(nn + 1, ens_neur.shape[0]):
                    auxdxy.append(scipy.spatial.distance.euclidean(com_ens[nn, :2], com_ens[nns, :2]))

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
            aux_std = np.nanstd(dff_ens[nn, startBMI:])
            if aux_std < 1:
                auxpostwhostd.append(aux_std)

        post_whole_std_mean = np.nanmean(auxpostwhostd)
        post_whole_std_max = np.nanmax(auxpostwhostd)
        post_whole_std_min = np.nanmin(auxpostwhostd)

        auxpostbasestd = []
        for nn in np.arange(dff_ens.shape[0]):
            aux_std = np.nanstd(dff_ens[nn, :startBMI])
            if aux_std < 1:
                auxpostbasestd.append(aux_std)

        post_base_std_mean = np.nanmean(auxpostbasestd)
        post_base_std_max = np.nanmax(auxpostbasestd)
        post_base_std_min = np.nanmin(auxpostbasestd)

    else:
        depth_mean = np.nan
        depth_max = np.nan
        depth_min = np.nan
        x_mean = np.nan
        x_max = np.nan
        x_min = np.nan
        y_mean = np.nan
        y_max = np.nan
        y_min = np.nan
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
        pseudo_dff = (online_data[:, nn] - np.nanmean(online_data[:, nn])) / np.nanmean(online_data[:, nn])
        auxonstd.append(np.nanstd(pseudo_dff))

    onstd_mean = np.nanmean(auxonstd)
    onstd_max = np.nanmax(auxonstd)
    onstd_min = np.nanmin(auxonstd)

    auxcursorstd = []
    cursor_std = np.nanstd(cursor)

    row_entry = np.asarray([depth_mean, depth_max, depth_min, \
                            dist_mean, dist_max, dist_min, diffdepth_mean, diffdepth_max, \
                            diffdepth_min, diffxy_mean, diffxy_max, diffxy_min, \
                            onstd_mean, onstd_max, onstd_min, post_whole_std_mean, post_whole_std_max,
                            post_whole_std_min, \
                            post_base_std_mean, post_base_std_max, post_base_std_min, cursor_std])

    return row_entry, ens_neur, e2_neur, e1_neur


def plot_results(folder_plots, df_aux, first_ind=0, single_animal=True, mode='basic'):
    '''
    Function to plot all the basic results in terms of linear regression 
    '''
    columns = df_aux.columns.tolist()
    # depth mean
    columns_ler = columns[4:10]

    if mode == 'basic':
        columns_aux = columns[10:32]  ## basic
        figsiz = (12, 20)
        sbx = 5
        sby = 5
    elif mode == 'SNR':
        columns_aux = columns[32:35]  ## SNR
        figsiz = (8, 4)
        sbx = 1
        sby = 3
    elif mode == 'ce':
        columns_aux = [columns[35]]  ## SNR
        figsiz = (4, 4)
        sbx = 1
        sby = 1
    elif mode == 'GC':
        columns_aux = columns[36:]  ## SNR
        figsiz = (8, 8)
        sbx = 4
        sby = 4

    for cc, col in enumerate(columns_ler):
        fig1 = plt.figure(figsize=figsiz)
        for tt, btest in enumerate(columns_aux):
            ax = fig1.add_subplot(sbx, sby, tt + 1)
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
    if not os.path.exists(folder_plots):
        os.makedirs(folder_plots)
    folder_snr = os.path.join(folder_main, 'onlineSNR')
    to_save_df = os.path.join(folder_main, 'df_all.hdf5')
    to_load_pick = os.path.join(folder_main, 'cursor_engagement.p')
    to_load_e2 = os.path.join(folder_main, 'e2_neurs.p')
    animals = os.listdir(folder)
    df_results = pd.read_csv(file_csv)
    df_ce = pd.read_pickle(to_load_pick)
    df_e2 = pd.read_pickle(to_load_e2)
    columns_res = df_results.columns.tolist()
    columns_basic = ['depth_mean', 'depth_max', 'depth_min', \
                     'dist_mean', 'dist_max', 'dist_min', 'diffdepth_mean', 'diffdepth_max', \
                     'diffdepth_min', 'diffxy_mean', 'diffxy_max', 'diffxy_min', \
                     'onstd_mean', 'onstd_max', 'onstd_min', 'post_whole_std_mean', 'post_whole_std_max',
                     'post_whole_std_min',
                     'post_base_std_mean', 'post_base_std_max', 'post_base_std_min', 'cursor_std']
    columns = columns_res + columns_basic
    columns.insert(3, 'ITPTlabel')
    df = pd.DataFrame(columns=columns)

    # obtain basic features
    print('obtaining basic features!')
    mat_ens_ind = np.zeros((len(animals), 25, 4)) + np.nan
    mat_e2_ind = np.zeros((len(animals), 25, 2)) + np.nan
    mat_e1_ind = np.zeros((len(animals), 25, 2)) + np.nan
    for aa, animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        mat_animal = np.zeros((len(filenames), len(columns_basic))) + np.nan
        for dd, filename in enumerate(filenames):
            day = filename[-17:-11]
            print('Analyzing animal: ' + animal + ' day: ' + day)
            try:
                mat_animal[dd, :], ens_neur, e2_neur, e1_neur = basic_entry(folder, animal, day, df_e2)
                mat_ens_ind[aa, dd, :len(ens_neur)] = ens_neur
                mat_e2_ind[aa, dd, :len(e2_neur)] = e2_neur
                mat_e1_ind[aa, dd, :len(e1_neur)] = e1_neur
            except OSError:
                print('day: ' + day + ' not obtained. ERRRRROOOOOOOOOOOOOOR')
                break
        df_res_animal = df_results.loc[df_results['animal'] == animal]
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
            plot_results(folder_plots, df_animal, df_res_animal.index[0])
        df = df.append(df_animal, ignore_index=True)
    if to_plot:
        plot_results(folder_plots, df, single_animal=False)
    df['ITPTlabel'] = pd.to_numeric(df['ITPTlabel'])

    # obtain snr features
    print('obtaining snrs')
    snr_vector_mean = []  # np.zeros(len(df))
    snr_vector_max = []  # np.zeros(len(df))
    snr_vector_min = []  # np.zeros(len(df))
    number_snrs = np.zeros(len(animals))
    for aa, animal in enumerate(animals):
        folder_path = os.path.join(folder_snr, animal)
        filenames = os.listdir(folder_path)
        number_snrs[aa] = len(filenames)
        for dd, filename in enumerate(filenames):
            print('Analyzing animal: ' + animal + ' day: ' + filename)
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
    for aa, animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        andf = df_ce.loc[df_ce['Animal'] == animal]
        for dd, filename in enumerate(filenames):
            day = filename[-17:-11]
            auxr2 = andf[andf['Date'] == day]['R2 Values'].values
            if len(auxr2) > 0:
                dfind = df.index[(df['animal'] == animal) & (df['day'] == int(day))]
                df.loc[dfind, 'cursor_eng'] = auxr2
    if to_plot:
        plot_results(folder_plots, df, single_animal=False, mode='ce')

    # obtain connectivity values
    df['GC_raw_ens_ens'] = np.nan
    df['GC_per_ens_ens'] = np.nan
    df['GC_raw_e2_e1'] = np.nan
    df['GC_per_e2_e1'] = np.nan
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

    for aa, animal in enumerate(animals):
        folder_path = os.path.join(folder, animal)
        filenames = os.listdir(folder_path)
        for dd, filename in enumerate(filenames):
            day = filename[-17:-11]
            dfind = df.index[(df['animal'] == animal) & (df['day'] == int(day))]
            to_load_FC = os.path.join(folder_main, 'FC', 'statsmodel', animal, day,
                                      'baseline_red_ens-indirect_dff_order_auto.p')
            try:
                df_FC = pd.read_pickle(to_load_FC)
            except FileNotFoundError:
                print('Animal: ' + animal + ' or day: ' + day + ' doesnt exist on the Granger causality analysis')
                break
            # extract the granger causality values
            FC_red = df_FC['FC_red']
            FC_pval_red = df_FC['FC_pval_red']
            ind_red = df_FC['indices_red']
            FC_ens_indirect = df_FC['FC_ens-indirect']
            FC_indirect_ens = df_FC['FC_indirect-ens']
            FC_pval_ens_indirect = df_FC['FC_pval_ens-indirect']
            FC_pval_indirect_ens = df_FC['FC_pval_indirect-ens']
            ind_ens_ind = df_FC['indices_ens-indirect']
            ind_ind_ens = df_FC['indices_indirect-ens']
            ens_neur = mat_ens_ind[aa, dd, ~np.isnan(mat_ens_ind[aa, dd, :])].astype(int)
            e2_neur = mat_e2_ind[aa, dd, ~np.isnan(mat_e2_ind[aa, dd, :])].astype(int)
            e1_neur = mat_e1_ind[aa, dd, ~np.isnan(mat_e1_ind[aa, dd, :])].astype(int)
            ind_mat_red = np.zeros(ens_neur.shape[0], dtype=np.int16)
            ind_mat_e2 = np.zeros(e2_neur.shape[0], dtype=np.int16)
            ind_mat_e1 = np.zeros(e1_neur.shape[0], dtype=np.int16)
            for ee, eneur in enumerate(ens_neur):
                ind_mat_red[ee] = np.where(ind_red == eneur)[0][0]
            for ee, eneur in enumerate(e2_neur):
                ind_mat_e2[ee] = np.where(ind_red == eneur)[0][0]
            for ee, eneur in enumerate(e1_neur):
                ind_mat_e1[ee] = np.where(ind_red == eneur)[0][0]
            FC_ens_red = FC_red[ind_mat_red, :]
            FC_ens_red[:, ind_mat_red] = np.nan
            FC_pval_ens_red = FC_pval_red[ind_mat_red, :]
            FC_pval_ens_red[:, ind_mat_red] = 1
            FC_red_ens = FC_red[:, ind_mat_red]
            FC_red_ens[ind_mat_red, :] = np.nan
            FC_pval_red_ens = FC_pval_red[:, ind_mat_red]
            FC_pval_red_ens[ind_mat_red, :] = 1
            if (len(ind_mat_e2) == 2) & (len(ind_mat_e1) == 2):
                FC_e2_e1 = np.concatenate((FC_red[ind_mat_e2, ind_mat_e1], FC_red[ind_mat_e1, ind_mat_e2]))
                FC_pval_e2_e1 = np.concatenate(
                    (FC_pval_red[ind_mat_e2, ind_mat_e1], FC_pval_red[ind_mat_e1, ind_mat_e2]))
                FC_ens_ens = np.concatenate(
                    ([FC_red[ind_mat_e2[0], ind_mat_e2[1]]], [FC_red[ind_mat_e1[0], ind_mat_e1[1]]]))
                FC_pval_ens_ens = np.concatenate(
                    ([FC_pval_red[ind_mat_e2[0], ind_mat_e2[1]]], [FC_pval_red[ind_mat_e1[0], ind_mat_e1[1]]]))
                calculate_ens = 0
            elif (len(ind_mat_e2) > 0) & (len(ind_mat_e1) > 0):
                FC_e2_e1 = np.concatenate((FC_red[ind_mat_e2, ind_mat_e1], FC_red[ind_mat_e1, ind_mat_e2]))
                FC_pval_e2_e1 = np.concatenate(
                    (FC_pval_red[ind_mat_e2, ind_mat_e1], FC_pval_red[ind_mat_e1, ind_mat_e2]))
                calculate_ens = 1
            else:
                FC_e2_e1 = np.zeros(1)
                FC_pval_e2_e1 = np.ones(1)
                calculate_ens = 1
            if calculate_ens == 1:
                if len(ind_mat_e2) == 2:
                    FC_ens_ens = FC_red[ind_mat_e2[0], ind_mat_e2[1]]
                    FC_pval_ens_ens = FC_pval_red[ind_mat_e2[0], ind_mat_e2[1]]
                elif len(ind_mat_e1) == 2:
                    FC_ens_ens = FC_red[ind_mat_e1[0], ind_mat_e1[1]]
                    FC_pval_ens_ens = FC_pval_red[ind_mat_e1[0], ind_mat_e1[1]]
                else:
                    FC_ens_ens = np.zeros(1)
                    FC_pval_ens_ens = np.ones(1)

            # obtain GC values, raw are values GC, per are number of connections with p<0.05
            # obtain ens ens connectivity
            raw_ens_ens = np.nanmean(FC_ens_ens[FC_pval_ens_ens < 0.05])
            if np.nansum(raw_ens_ens) == 0:
                raw_ens_ens = 0
            per_ens_ens = np.nansum(FC_pval_ens_ens < 0.05) / np.prod(FC_pval_ens_ens.shape)
            raw_e2_e1 = np.nanmean(FC_e2_e1[FC_pval_e2_e1 < 0.05])
            if np.nansum(raw_e2_e1) == 0:
                raw_e2_e1 = 0
            per_e2_e1 = np.nansum(FC_pval_e2_e1 < 0.05) / np.prod(FC_pval_e2_e1.shape)
            # obtain red connectivity
            raw_ens_red = np.nanmean(FC_ens_red[FC_pval_ens_red < 0.05])
            if np.nansum(raw_ens_red) == 0:
                raw_ens_red = 0
            raw_red_ens = np.nanmean(FC_red_ens[FC_pval_red_ens < 0.05])
            if np.nansum(raw_red_ens) == 0:
                raw_red_ens = 0
            per_ens_red = np.nansum(FC_pval_ens_red < 0.05) / np.prod(FC_pval_ens_red.shape)
            per_red_ens = np.nansum(FC_pval_red_ens < 0.05) / np.prod(FC_pval_red_ens.shape)
            # obtain green connectivity
            raw_ens_ind = np.nanmean(FC_ens_indirect[FC_pval_ens_indirect < 0.05])
            if np.nansum(raw_ens_ind) == 0:
                raw_ens_ind = 0
            raw_ind_ens = np.nanmean(FC_indirect_ens[FC_pval_indirect_ens < 0.05])
            if np.nansum(raw_ind_ens) == 0:
                raw_ind_ens = 0
            per_ens_ind = np.nansum(FC_pval_ens_indirect < 0.05) / np.prod(FC_pval_ens_indirect.shape)
            per_ind_ens = np.nansum(FC_pval_indirect_ens < 0.05) / np.prod(FC_pval_indirect_ens.shape)

            # df ens-ens
            df.loc[dfind, 'GC_raw_ens_ens'] = raw_ens_ens
            df.loc[dfind, 'GC_per_ens_ens'] = per_ens_ens
            df.loc[dfind, 'GC_raw_e2_e1'] = raw_e2_e1
            df.loc[dfind, 'GC_per_e2_e1'] = per_e2_e1
            # df red-ind            
            df.loc[dfind, 'GC_raw_ens_red'] = raw_ens_red
            df.loc[dfind, 'GC_raw_red_ens'] = raw_red_ens
            df.loc[dfind, 'GC_per_ens_red'] = per_ens_red
            df.loc[dfind, 'GC_per_red_ens'] = per_red_ens
            df.loc[dfind, 'GC_raw_ens_ind'] = raw_ens_ind
            df.loc[dfind, 'GC_raw_ind_ens'] = raw_ind_ens
            df.loc[dfind, 'GC_per_ens_ind'] = per_ens_ind
            df.loc[dfind, 'GC_per_ind_ens'] = per_ind_ens
            # df ratio
            if raw_ens_ind > 0:
                df.loc[dfind, 'GC_raw_ratio_ens_x'] = raw_ens_red / raw_ens_ind
            else:
                df.loc[dfind, 'GC_raw_ratio_ens_x'] = np.nan
            if raw_ind_ens > 0:
                df.loc[dfind, 'GC_raw_ratio_x_ens'] = raw_red_ens / raw_ind_ens
            else:
                df.loc[dfind, 'GC_raw_ratio_x_ens'] = np.nan
            if per_ens_ind > 0:
                df.loc[dfind, 'GC_per_ratio_ens_x'] = per_ens_red / per_ens_ind
            else:
                df.loc[dfind, 'GC_per_ratio_ens_x'] = np.nan
            if per_ind_ens > 0:
                df.loc[dfind, 'GC_per_ratio_x_ens'] = per_red_ens / per_ind_ens
            else:
                df.loc[dfind, 'GC_per_ratio_x_ens'] = np.nan
    if to_plot:
        plot_results(folder_plots, df, single_animal=False, mode='GC')

    # save!
    df.to_hdf(to_save_df, key='df', mode='w')


def modify_df_speed(df, folder_proc, time_learning=10):
    '''
    to modify an entry to the pd dataframe based on experimental time of learning in min. time_learning (min)
    '''

    animals = os.listdir(folder_proc)
    for aa, animal in enumerate(animals):
        folder_path = os.path.join(folder_proc, animal)
        filenames = os.listdir(folder_path)
        for dd, filename in enumerate(filenames):

            day = filename[-17:-11]
            print('processing animal: ' + animal + ' and day: ' + day)
            dfind = df.index[(df['animal'] == animal) & (df['day'] == int(day))]

            file_template = "full_{}_{}__data.hdf5"
            file_name = os.path.join(folder_proc, animal, file_template.format(animal, day))
            f = h5py.File(file_name, 'r')

            ens_neur = np.asarray(f['ens_neur'])

            online_data = np.asarray(f['online_data'])
            dff = np.asarray(f['dff'])
            cursor = np.asarray(f['cursor'])

            startBMI = np.asarray(f['trial_start'])[0]
            frame_time = np.int(startBMI / 15 * time_learning)
            online_data = online_data[online_data[:, 1] < frame_time, 2:]
            f.close()

            ens_neur = np.int16(ens_neur[~np.isnan(ens_neur)])

            if len(ens_neur) > 0:

                dff_ens = dff[ens_neur, :]

                # dynamic range
                # there are some dff that are crazy weird, to avoid them for deteriorating the dataset any std>1 will be ignored
                auxpostwhostd = []
                for nn in np.arange(dff_ens.shape[0]):
                    aux_std = np.nanstd(dff_ens[nn, startBMI:startBMI + frame_time])
                    if aux_std < 1:
                        auxpostwhostd.append(aux_std)

                df.loc[dfind, 'post_whole_std_mean'] = np.nanmean(auxpostwhostd)
                df.loc[dfind, 'post_whole_std_max'] = np.nanmax(auxpostwhostd)
                df.loc[dfind, 'post_whole_std_min'] = np.nanmin(auxpostwhostd)

            auxonstd = []
            for nn in np.arange(online_data.shape[1]):
                pseudo_dff = (online_data[:, nn] - np.nanmean(online_data[:, nn])) / np.nanmean(online_data[:, nn])
                auxonstd.append(np.nanstd(pseudo_dff))

            df.loc[dfind, 'onstd_mean'] = np.nanmean(auxonstd)
            df.loc[dfind, 'onstd_max'] = np.nanmax(auxonstd)
            df.loc[dfind, 'onstd_min'] = np.nanmin(auxonstd)

            df.loc[dfind, 'cursor_std'] = np.nanstd(cursor[:frame_time])
    to_save_df = os.path.join(folder_main, 'df_all.hdf5')
    df.to_hdf(to_save_df, key='df', mode='w')

    return df


def bootstrap_pandas(len_df, X_df, Y_df, bts_n=1000):
    ''' Bootstrap  pd 
    df = panda dataframe
    n = size of bootstrap matrix (how many values)

    returns dfbst
    '''

    if bts_n == 0:
        bts_n = len_df

    bootst_ind = np.floor(np.random.rand(bts_n) * len_df).astype(int)
    X_df_bst = X_df.iloc[bootst_ind]
    Y_df_bst = Y_df.iloc[bootst_ind]

    # dfbst = df.iloc[bootst_ind]
    return bootst_ind, X_df_bst, Y_df_bst


def split_df(df, bts_n=1000, learn_stat_colum='totalPC', size_split_test=0.2, classif=False, synthetic=False):
    '''
    Function to calculate the xgboost model
    '''
    # select X and Y
    columns = df.columns
    if classif:
        labels_to_study = columns[10:].tolist()
        Y_df = df['ITPTlabel']
    elif synthetic:
        Y_df = df.loc[:, 'PC_fake']
        labels_to_study = columns[1:]
    else:
        labels_to_study = [columns[3]] + columns[10:].tolist()
        Y_df = df.loc[:, learn_stat_colum]
    X_df = df.loc[:, labels_to_study]

    if np.isinf(np.nansum(Y_df)):
        X_df = X_df[~np.isinf(Y_df)]
        Y_df = Y_df[~np.isinf(Y_df)]
    if np.sum(np.isnan(Y_df)) > 0:
        X_df = X_df[~np.isnan(Y_df)]
        Y_df = Y_df[~np.isnan(Y_df)]

    # split in train/test
    X_df_train, X_df_test, Y_df_train, Y_df_test = train_test_split(X_df, Y_df, test_size=size_split_test)

    return X_df_train, X_df_test, Y_df_train, Y_df_test


def calculate_model(X_df, Y_df, learning_rate=0.1, xgboost_rep=100):
    model = xgboost.train({"learning_rate": learning_rate}, xgboost.DMatrix(X_df, label=Y_df), xgboost_rep)

    return model


def calculate_bst632(model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test):
    '''
    function to calculate the error of the bootstrap by rule 0.632
    '''

    # calculate errors
    error_samp = mean_squared_error(model.predict(xgboost.DMatrix(X_df_train_bst, label=Y_df_train_bst)),
                                    Y_df_train_bst)
    error_test = mean_squared_error(model.predict(xgboost.DMatrix(X_df_test, label=Y_df_test)), Y_df_test)
    error_bst = 0.632 * error_test + 0.368 * error_samp

    return error_bst


def calculate_bst_length_optimal(df, rep=100, bst_num=None):
    '''
    function to calculate the best length for the bootstrap depending on the rule 0.632
    '''
    # some values for HPM
    # 0.200 for 40/60 split (minimum error 0.15)
    # 0.204 for 20/80 (0.133 for error<0.15)
    # 0.188 for 1000 xgboost_rep (0.124 for error<0.15) 20/80
    # 0.190 for learning 0.1 (0.123 for error<0.15)
    # 0.198 for learning 0.3 (0.126 for error<0.15)

    if bst_num == None:
        bst_num = np.arange(0, 10000, 1000)
        bst_num[0] = 100
    error_bst = np.zeros((rep, bst_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, bn in enumerate(bst_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, bn)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            error_bst[rr, ind] = calculate_bst632(model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst


def calculate_size_train_optimal(df, rep=100, size_num=None):
    '''
    function to calculate the best size for the XGboost depending on the rule 0.632
    '''

    if size_num == None:
        size_num = np.arange(0.1, 0.6, 0.1)
    error_bst = np.zeros((rep, size_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, sn in enumerate(size_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, size_split_test=sn)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            error_bst[rr, ind] = calculate_bst632(model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst


def calculate_learning_optimal(df, rep=100, learn_num=None, classif=False):
    '''
    function to calculate the learning_rate for the XGboost depending on the rule 0.632
    '''

    if learn_num == None:
        learn_num = np.arange(0, 0.4, 0.05)
        learn_num[0] = 0.01
    error_bst = np.zeros((rep, learn_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, ln in enumerate(learn_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, classif=classif)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst, learning_rate=ln)
            error_bst[rr, ind] = calculate_bst632(model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst


def calculate_xgbrep_optimal(df, rep=100, xgrep_num=None):
    '''
    function to calculate the best number of trees for the XGboost depending on the rule 0.632
    '''

    if xgrep_num == None:
        xgrep_num = np.asarray([20, 50, 100, 200, 500, 1000, 2000])
    error_bst = np.zeros((rep, xgrep_num.shape[0])) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, xn in enumerate(xgrep_num):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst, xgboost_rep=xn)
            error_bst[rr, ind] = calculate_bst632(model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst


def calculate_learn_stat_optimal(df, rep=100, bts_n=1000, classif=False):
    '''
    function to calculate the error for each learning stat for the XGboost depending on the rule 0.632
    '''
    columns = df.columns.tolist()
    columns_ler = [columns[6]]  # columns[4:10]
    error_bst = np.zeros((rep, len(columns_ler))) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, cn in enumerate(columns_ler):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, learn_stat_colum=cn, classif=classif)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            error_bst[rr, ind] = calculate_bst632(model, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
    return error_bst


def calculate_learn_stat_mseoptimal(df, rep=100, bts_n=1000, classif=False):
    '''
    function to calculate the error for each learning stat for the XGboost depending on the rule 0.632
    '''
    columns = df.columns.tolist()
    columns_ler = [columns[6]]  # columns[4:10]
    error_bst = np.zeros((rep, len(columns_ler))) + np.nan
    for rr in np.arange(rep):
        print('repetition: ' + str(rr))
        for ind, cn in enumerate(columns_ler):
            X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, learn_stat_colum=cn, classif=classif)
            _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train, bts_n)
            model = calculate_model(X_df_train_bst, Y_df_train_bst)
            aux_predict = model.predict(xgboost.DMatrix(X_df_test, label=Y_df_test))
            error_bst[rr, ind] = mean_squared_error(Y_df_test, aux_predict)
    return error_bst


def calculate_all_errors(df, folder_main, rep=1000):
    '''
    function to calculate and store all the errors to obtain the optimal parameters for the model
    '''
    f = h5py.File(os.path.join(folder_main, 'plots', 'utils', 'tpc_error_optimal.h5py'), 'w-')
    error_length = calculate_bst_length_optimal(df, rep)
    error_size = calculate_size_train_optimal(df, rep)
    error_learning = calculate_learning_optimal(df, rep)
    error_xgrep = calculate_xgbrep_optimal(df, rep)
    error_learn_stat = calculate_learn_stat_optimal(df, rep)
    error_learn_statmse = calculate_learn_stat_mseoptimal(df, rep)
    f.create_dataset('error_length', data=error_length)
    f.create_dataset('error_size', data=error_size)
    f.create_dataset('error_learning', data=error_learning)
    f.create_dataset('error_xgrep', data=error_xgrep)
    f.create_dataset('error_learn_stat', data=error_learn_stat)
    f.create_dataset('error_learn_statmse', data=error_learn_statmse)
    f.close()


def obtain_shap_iter(df, folder_main, bts_n=1000, mod_n=10000, mod_x=100, error_bstmax=[0.02, 0.2], \
                     error_msemax=[0.03, 0.3], size_split_test=0.2, max_iter=40, stability_var=0.6, classif=False,
                     synthetic=False, toplot=True):
    '''
    obtain shap values of mod_n different XGboost model if the conditions for error of the model and stability are set
    obtain stabitlity of feature: correlation of original shap values and bootstrap values to see if values are miningful or noise
    '''
    columns = df.columns.tolist()
    if classif:
        # this is to classify IT vs PT
        columns_ler = [columns[3]]
        labels_to_study = columns[10:]  # columns[10:]
    elif synthetic:
        columns_ler = [columns[0]]
        labels_to_study = columns[1:]
    else:
        # this is to study the learning stats on columns_ler
        columns_ler = [columns[6]]  # columns[4:10] #[columns[6]]# [columns[3]]
        labels_to_study = [columns[3]] + columns[10:]  # columns[10:]

    test_size = np.ceil(len(df) * (size_split_test)).astype(int)
    train_size = np.floor(len(df) * (1 - size_split_test)).astype(int)

    all_shap = np.zeros((len(columns_ler), mod_n, test_size, len(labels_to_study))) + np.nan
    all_y_pred = np.zeros((len(columns_ler), mod_n, test_size)) + np.nan
    all_mse = np.zeros((len(columns_ler), mod_n)) + np.nan
    shap_correlations = np.zeros((len(columns_ler), mod_n, mod_x, len(labels_to_study))) + np.nan
    explainer_val = np.zeros((len(columns_ler), mod_n)) + np.nan
    all_df = np.zeros((len(columns_ler), mod_n, test_size)) + np.nan
    number_models = np.zeros(len(columns_ler), dtype=int)

    for cc, col_ler in enumerate(columns_ler):
        i = 0
        iteri = 0
        while i < mod_n:
            if iteri < max_iter:
                all_shap_train_aux = np.zeros((mod_x + 1, train_size, len(labels_to_study))) + np.nan
                shap_cor_aux = np.zeros((mod_x, len(labels_to_study))) + np.nan
                j = 1
                iterj = 0
                # make splits for original model
                X_df_train, X_df_test, Y_df_train, Y_df_test = split_df(df, bts_n, col_ler,
                                                                        size_split_test=size_split_test,
                                                                        classif=classif, synthetic=synthetic)
                # calculate original
                model_original = calculate_model(X_df_train, Y_df_train)
                number_models[cc] += 1

                # first check for the model (using bst632)
                error_bst = calculate_bst632(model_original, X_df_train, X_df_test, Y_df_train, Y_df_test)
                # second check for the model (using mse)
                aux_predict = model_original.predict(xgboost.DMatrix(X_df_test, label=Y_df_test))
                aux_mse = mean_squared_error(Y_df_test, aux_predict)

                # if the model is good enough (mse/bst error low)               
                if (error_bst < error_bstmax[cc]) & (aux_mse < error_msemax[cc]):
                    explainer_train = shap.TreeExplainer(model_original,
                                                         feature_perturbation='tree_path_dependent')  # True to data!, data=X_df_test, feature_perturbation='interventional')
                    # just in case the size differs we will take only the size of X_df
                    all_shap_train_aux[0, :len(X_df_train), :] = explainer_train.shap_values(X_df_train)

                    # bootstrap check for stability of features
                    while j < (mod_x + 1):
                        if iterj > max_iter:
                            print(
                                'too many iterations, check that the maximum error is not too restrictive or split was just bad')
                            break
                        print('repetition: ' + str(i) + ':' + str(j) + ', with iterj: ' + str(iterj))
                        _, X_df_train_bst, Y_df_train_bst = bootstrap_pandas(len(X_df_train), X_df_train, Y_df_train,
                                                                             bts_n)
                        model_bst = calculate_model(X_df_train_bst, Y_df_train_bst)
                        error_bst = calculate_bst632(model_bst, X_df_train_bst, X_df_test, Y_df_train_bst, Y_df_test)
                        if error_bst < error_bstmax[cc]:
                            explainer_bst = shap.TreeExplainer(model_bst,
                                                               feature_perturbation='tree_path_dependent')  # True to data!, data=X_df_test, feature_perturbation='interventional')
                            all_shap_train_aux[j, :len(X_df_train), :] = explainer_bst.shap_values(X_df_train)

                            # check correlation of features with features from original model
                            for ll, label_ts in enumerate(labels_to_study):
                                if np.nansum(all_shap_train_aux[j, :len(X_df_train), ll]) > 0:
                                    shap_cor_aux[j - 1, ll] = np.corrcoef(all_shap_train_aux[0, :len(X_df_train), ll], \
                                                                          all_shap_train_aux[j, :len(X_df_train), ll])[
                                        0, 1]
                            j += 1
                            iterj = 0
                        else:
                            iterj += 1

                    aaux = np.nanmean(all_shap_train_aux, 0)

                    # if the model had a relatively low error and the features were not only noise
                    if (np.nansum(aaux) != 0) & (np.nanmean(shap_cor_aux) > stability_var):
                        # first store everything we used to calculate stability
                        shap_correlations[cc, i, :, :] = shap_cor_aux

                        # keep info of the xgboost performance
                        all_y_pred[cc, i, :len(X_df_test)] = aux_predict
                        all_mse[cc, i] = aux_mse

                        # now calculate the shap_values with the X_df_test FINALLY!
                        explainer_test = shap.TreeExplainer(model_original,
                                                            feature_perturbation='tree_path_dependent')  # True to data!, data=X_df_train, feature_perturbation='interventional')
                        all_shap[cc, i, :len(X_df_test), :] = explainer_test.shap_values(X_df_test)
                        explainer_val[cc, i] = explainer_test.expected_value
                        ind_aux = np.zeros(X_df_test.shape[0], dtype=np.int16)
                        for xx in np.arange(X_df_test.shape[0]):
                            ind_aux[xx] = df.index.get_loc(X_df_test.index[xx])
                        all_df[cc, i, :len(X_df_test)] = ind_aux

                        # update
                        i += 1
                        iteri = 0
                        print('new model processed: ' + str(i) + '/' + str(mod_n))

                    else:
                        print('model not up to specs, repeting model. Iteri: ' + str(iteri))
                        print(np.nanmean(shap_cor_aux))
                        iteri += 1
                else:
                    print('too high bst ' + str(~(error_bst < error_bstmax[cc])) + ' or mse error ' + \
                          str(~(aux_mse < error_msemax[cc])) + ', repeting split. Iteri: ' + str(iteri))
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
    bins_zscore = np.arange(-2, 2, 0.1)
    spread = np.zeros((len(columns_ler), len(df), len(labels_to_study), len(bins_zscore) - 1)) + np.nan
    all_ind = np.zeros((len(columns_ler), len(df))) + np.nan
    for cc, col_ler in enumerate(columns_ler):
        for ind in np.arange(len(df)):
            aux_ind = np.where(all_df_reshape[cc, :] == ind)[0]
            all_ind[cc, ind] = len(aux_ind)
            if all_ind[cc, ind] > 0:
                aux_shap = all_shap_reshape[cc, aux_ind, :]
                shap_experiment_mean[cc, ind, :] = np.nanmean(aux_shap, 0)
                shap_experiment_std[cc, ind, :] = np.nanstd(aux_shap, 0)
                shap_experiment_sem[cc, ind, :] = np.nanstd(aux_shap, 0) / np.sqrt(len(aux_ind))
                for ll, label_ts in enumerate(labels_to_study):
                    [h, b] = np.histogram(zscore(aux_shap[:, ll]), bins_zscore)
                    spread[cc, ind, ll, :] = h
    # the spread was gaussian

    if classif:
        f = h5py.File(os.path.join(folder_main, 'XGShap_classif_model.h5py'), 'w-')
    else:
        f = h5py.File(os.path.join(folder_main, 'XGShap_model.h5py'), 'w-')

    for key in ['labels_to_study', 'test_size', 'train_size', 'all_shap', 'all_y_pred', 'all_mse', 'shap_correlations', \
                'explainer_val', 'all_df', 'number_models', 'all_shap_reshape', 'all_df_reshape',
                'shap_experiment_mean', \
                'shap_experiment_std', 'shap_experiment_sem', 'spread']:
        try:
            f.create_dataset(key, data=eval(key))
        except TypeError:
            try:
                f.attrs[key] = eval(key)
            except:
                print('ERROR ' + key)
    f.close()

    # lets get plotting!!!

    if toplot:
        sizesubpl = np.ceil(len(labels_to_study) / 6).astype('int')

        # check stability of features
        folder_plots_sta = os.path.join(folder_main, 'plots', 'XGBoost', 'feature_stability')
        print('plotting feature stability')
        if not os.path.exists(folder_plots_sta):
            os.makedirs(folder_plots_sta)
        bins_cor = np.arange(stability_var - 0.2, 1, 0.01)
        for cc, col_ler in enumerate(columns_ler):
            fig1 = plt.figure(figsize=(17, 9))
            for ll, label_ts in enumerate(labels_to_study):
                ax0 = fig1.add_subplot(sizesubpl, 6, ll + 1)
                if np.nansum(shap_correlations[cc, :, :, ll]) > 0:
                    [h, b] = np.histogram(shap_correlations[cc, :, :, ll], bins_cor)
                    ax0.bar(b[1:], h, width=0.01)
                    ax0.set_xlabel(label_ts)

            fig1.tight_layout()
            fig1.savefig(os.path.join(folder_plots_sta, col_ler + '_stability_features.png'), bbox_inches="tight")
            fig1.savefig(os.path.join(folder_plots_sta, col_ler + '_stability_features.eps'), bbox_inches="tight")
            plt.close('all')

        # check IT/PT shap values
        folder_plots_ITPT = os.path.join(folder_main, 'plots', 'XGBoost', 'ITPT')
        print('ITPT stuff')
        if not os.path.exists(folder_plots_ITPT):
            os.makedirs(folder_plots_ITPT)
        all_IT = np.zeros(len(columns_ler)) + np.nan
        all_PT = np.zeros(len(columns_ler)) + np.nan
        bins_shap = np.arange(-0.002, 0.002, 0.00005)

        for cc, col_ler in enumerate(columns_ler):
            aux_IT = shap_experiment_mean[cc, df['ITPTlabel'] == 0, 0]
            aux_PT = shap_experiment_mean[cc, df['ITPTlabel'] == 1, 0]
            all_IT[cc] = np.nanmean(aux_IT)
            all_PT[cc] = np.nanmean(aux_PT)

            fig2 = plt.figure(figsize=(12, 4))
            ax1 = fig2.add_subplot(1, 2, 1)
            [h_IT, b] = np.histogram(aux_IT, bins_shap)
            [h_PT, b] = np.histogram(aux_PT, bins_shap)
            ax1.bar(b[1:], h_IT, width=0.00005, label='IT')
            ax1.bar(b[1:], h_PT, width=0.00005, label='PT')
            ax1.legend()

            ax2 = fig2.add_subplot(1, 2, 2)
            ax2.bar([0.4, 1.4], [all_IT[cc], all_PT[cc]], width=0.8, \
                    yerr=[pd.DataFrame(aux_IT).sem(0).values[0], pd.DataFrame(aux_PT).sem(0).values[0]], \
                    error_kw=dict(ecolor='k'))
            ax2.set_xticks([0.4, 1.4])
            ax2.set_xticklabels(['IT', 'PT'])
            _, p_value = stats.ttest_ind(aux_IT, aux_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.8, 0.001, p, color='grey', alpha=0.6)
            ax2.set_ylim([-0.001, 0.001])

            fig2.savefig(os.path.join(folder_plots_ITPT, col_ler + '_ITPT_shap_val.png'), bbox_inches="tight")
            fig2.savefig(os.path.join(folder_plots_ITPT, col_ler + '_ITPT_shap_val.eps'), bbox_inches="tight")
            plt.close('all')

        # check shap summary plot
        folder_plots_shap = os.path.join(folder_main, 'plots', 'XGBoost', 'shap')
        print('plotting shap')
        if not os.path.exists(folder_plots_shap):
            os.makedirs(folder_plots_shap)
        bins_shap = np.arange(-0.05, 0.05, 0.001)

        for cc, col_ler in enumerate(columns_ler):
            fig3 = plt.figure()
            aux_df = np.where(~np.isnan(np.sum(shap_experiment_mean[cc, :, :], 1)))[0]
            shap.summary_plot(shap_experiment_mean[cc, aux_df, :], df.iloc[aux_df][labels_to_study])

            fig3.savefig(os.path.join(folder_plots_shap, col_ler + '_summary.png'), bbox_inches="tight")
            fig3.savefig(os.path.join(folder_plots_shap, col_ler + '_summary.svg'), format='svg', bbox_inches="tight")
            plt.close('all')

        # check dependencies
        folder_plots_depend = os.path.join(folder_main, 'plots', 'XGBoost', 'dependencies')
        print('plotting dependencies')
        if not os.path.exists(folder_plots_depend):
            os.makedirs(folder_plots_depend)
        sizesubpl = np.ceil(len(labels_to_study) / 6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            for ll, label_ts in enumerate(labels_to_study):
                fig4 = plt.figure(figsize=(24, 12))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax3 = fig4.add_subplot(sizesubpl, 6, llsec + 1)
                    aux_df = np.where(~np.isnan(np.sum(shap_experiment_mean[cc, :, :], 1)))[0]
                    shap.dependence_plot(label_ts, shap_experiment_mean[cc, aux_df, :],
                                         df.iloc[aux_df][labels_to_study], interaction_index=llsec, ax=ax3)
                fig4.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig4.savefig(os.path.join(folder_plots_depend, col_ler + '_' + label_ts + '_depen.png'),
                             bbox_inches="tight")
                fig4.savefig(os.path.join(folder_plots_depend, col_ler + '_' + label_ts + '_depen.eps'),
                             bbox_inches="tight")
                plt.close('all')

        # check regression inter feature       
        folder_plots_reg_feat = os.path.join(folder_main, 'plots', 'XGBoost', 'regression_feat')
        print('plotting regressions')
        if not os.path.exists(folder_plots_reg_feat):
            os.makedirs(folder_plots_reg_feat)
        sizesubpl = np.ceil(len(labels_to_study) / 6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            for ll, label_ts in enumerate(labels_to_study):
                fig5 = plt.figure(figsize=(24, 12))
                fig5b = plt.figure(figsize=(24, 12))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax4 = fig5.add_subplot(sizesubpl, 6, llsec + 1)
                    ax4b = fig5b.add_subplot(sizesubpl, 6, llsec + 1)
                    sns.regplot(df[label_ts], df[label_tsec], ax=ax4)
                    sns.regplot(df[label_ts][df['ITPTlabel'] == 0], df[label_tsec][df['ITPTlabel'] == 0], ax=ax4b,
                                label='IT')
                    sns.regplot(df[label_ts][df['ITPTlabel'] == 1], df[label_tsec][df['ITPTlabel'] == 1], ax=ax4b,
                                label='PT')
                    plt.legend()
                    ax4.set_ylabel(label_tsec)
                    ax4b.set_ylabel(label_tsec)
                fig5.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig5.savefig(os.path.join(folder_plots_reg_feat, col_ler + '_' + label_ts + '_regfet.png'),
                             bbox_inches="tight")
                fig5.savefig(os.path.join(folder_plots_reg_feat, col_ler + '_' + label_ts + '_regfet.eps'),
                             bbox_inches="tight")
                fig5b.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig5b.savefig(os.path.join(folder_plots_reg_feat, col_ler + '_' + label_ts + '_regfet_ITPT.png'),
                              bbox_inches="tight")
                fig5b.savefig(os.path.join(folder_plots_reg_feat, col_ler + '_' + label_ts + '_regfet_ITPT.eps'),
                              bbox_inches="tight")
                plt.close('all')

        # labels to study regression to shap
        folder_plots_reg = os.path.join(folder_main, 'plots', 'XGBoost', 'regression_shap')
        print('plotting more regressions')
        if not os.path.exists(folder_plots_reg):
            os.makedirs(folder_plots_reg)
        sizesubpl = np.ceil(len(labels_to_study) / 6).astype('int')

        for cc, col_ler in enumerate(columns_ler):
            fig6 = plt.figure(figsize=(24, 12))
            for ll, label_ts in enumerate(labels_to_study):
                ax6 = fig6.add_subplot(sizesubpl, 6, ll + 1)
                aux_df = np.where(~np.isnan(np.sum(shap_experiment_mean[cc, :, :], 1)))[0]
                sns.regplot(df.iloc[aux_df][label_ts], shap_experiment_mean[cc, aux_df, ll], ax=ax6)
                ax6.set_ylabel('shap_val')
            fig6.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig6.savefig(os.path.join(folder_plots_reg, col_ler + '_reg.png'), bbox_inches="tight")
            fig6.savefig(os.path.join(folder_plots_reg, col_ler + '_reg.svg'), format='svg', bbox_inches="tight")
            plt.close('all')

            aux_IT = shap_experiment_mean[cc, df['ITPTlabel'] == 0, :]
            aux_PT = shap_experiment_mean[cc, df['ITPTlabel'] == 1, :]
            df_IT = df.loc[df['ITPTlabel'] == 0]
            df_PT = df.loc[df['ITPTlabel'] == 1]

            fig61 = plt.figure(figsize=(24, 12))
            for ll, label_ts in enumerate(labels_to_study):
                ax61 = fig61.add_subplot(sizesubpl, 6, ll + 1)
                aux_df = np.where(~np.isnan(np.sum(aux_IT[:, :], 1)))[0]
                sns.regplot(df_IT.iloc[aux_df][label_ts], aux_IT[aux_df, ll], ax=ax61)
                ax61.set_ylabel('shap_val')
            fig61.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig61.savefig(os.path.join(folder_plots_reg, col_ler + '_IT_reg.png'), bbox_inches="tight")
            fig61.savefig(os.path.join(folder_plots_reg, col_ler + '_IT_reg.svg'), format='svg', bbox_inches="tight")
            plt.close('all')

            fig62 = plt.figure(figsize=(24, 12))
            for ll, label_ts in enumerate(labels_to_study):
                ax62 = fig62.add_subplot(sizesubpl, 6, ll + 1)
                aux_df = np.where(~np.isnan(np.sum(aux_PT[:, :], 1)))[0]
                sns.regplot(df_PT.iloc[aux_df][label_ts], aux_PT[aux_df, ll], ax=ax62)
                ax62.set_ylabel('shap_val')
            fig62.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig62.savefig(os.path.join(folder_plots_reg, col_ler + '_PT_reg.png'), bbox_inches="tight")
            fig62.savefig(os.path.join(folder_plots_reg, col_ler + '_PT_reg.svg'), format='svg', bbox_inches="tight")
            plt.close('all')

            for ll, label_ts in enumerate(labels_to_study):
                fig63 = plt.figure(figsize=(3, 2))
                ax63 = fig63.add_subplot(1, 1, 1)
                aux_df = np.where(~np.isnan(np.sum(aux_IT[:, :], 1)))[0]
                sns.regplot(df_IT.iloc[aux_df][label_ts], aux_IT[aux_df, ll], ax=ax63, label='IT')
                aux_df = np.where(~np.isnan(np.sum(aux_PT[:, :], 1)))[0]
                sns.regplot(df_PT.iloc[aux_df][label_ts], aux_PT[aux_df, ll], ax=ax63, label='PT')
                ax63.set_ylabel('shap_val')
                plt.legend()
                fig63.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig63.savefig(os.path.join(folder_plots_reg, col_ler + '_' + label_ts + '_tog_reg.png'),
                              bbox_inches="tight")
                fig63.savefig(os.path.join(folder_plots_reg, col_ler + '_' + label_ts + '_tog_reg.svg'),
                              format='svg', bbox_inches="tight")
                plt.close('all')

        # check for confidence interval on features
        folder_plots_ci = os.path.join(folder_main, 'plots', 'XGBoost', 'confidence interval')
        print('plotting CI')
        if not os.path.exists(folder_plots_ci):
            os.makedirs(folder_plots_ci)
        for cc, col_ler in enumerate(columns_ler):
            fig7 = plt.figure(figsize=(24, 12))
            for ll, label_ts in enumerate(labels_to_study):
                ax7 = fig7.add_subplot(sizesubpl, 6, ll + 1)
                aux_spread = np.nansum(spread[cc, :, ll, :], 0)
                ax7.bar(bins_zscore[1:], aux_spread, width=0.1)
                ax7.set_ylabel(label_ts)
            fig7.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig7.savefig(os.path.join(folder_plots_ci, col_ler + '_ci.png'), bbox_inches="tight")
            fig7.savefig(os.path.join(folder_plots_ci, col_ler + '_ci.eps'), bbox_inches="tight")
            plt.close('all')

        # check for groups of labels
        folder_plots_gro = os.path.join(folder_main, 'plots', 'XGBoost', 'groups')
        print('plotting groups shap')
        if not os.path.exists(folder_plots_gro):
            os.makedirs(folder_plots_gro)
        if classif:
            groups_labels = np.asarray(['Position', 'STD', 'SNR', 'Engagement', 'Connectivity'])
            groups_index = [np.arange(0, 12), np.arange(12, 22), np.arange(22, 25), 25, np.arange(26, 42)]
            shap_group = np.stack((np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[0]]), 2), \
                                   np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[1]]), 2), \
                                   np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[2]]), 2), \
                                   np.abs(shap_experiment_mean[:, :, groups_index[3]]), \
                                   np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[4]]), 2)), axis=2)
        else:
            groups_labels = np.asarray(['ITPT', 'Position', 'STD', 'SNR', 'Engagement', 'Connectivity'])
            groups_index = [0, np.arange(1, 13), np.arange(13, 23), np.arange(23, 26), 26, np.arange(27, 43)]
            shap_group = np.stack((shap_experiment_mean[:, :, groups_index[0]], \
                                   np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[1]]), 2), \
                                   np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[2]]), 2), \
                                   np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[3]]), 2), \
                                   np.abs(shap_experiment_mean[:, :, groups_index[4]]), \
                                   np.nanmean(np.abs(shap_experiment_mean[:, :, groups_index[5]]), 2)), axis=2)

        for cc, col_ler in enumerate(columns_ler):
            fig8 = plt.figure()
            aux_df = np.where(~np.isnan(np.sum(shap_group[cc, :, :], 1)))[0]
            aux_shap = np.nanmean(np.abs(shap_group[cc, aux_df, :]), 0)
            order_shap = np.argsort(aux_shap)
            plt.barh(np.arange(0, len(aux_shap)), np.sort(aux_shap),
                     xerr=pd.DataFrame(np.abs(shap_group[cc, aux_df, :])).sem(0))
            plt.xlabel('mean(|SHAP value|) (total impact on model output magnitud')
            plt.yticks(np.arange(0, len(aux_shap)), groups_labels[order_shap])

            fig8.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig8.savefig(os.path.join(folder_plots_gro, col_ler + '_group_bar_mean.png'), bbox_inches="tight")
            fig8.savefig(os.path.join(folder_plots_gro, col_ler + '_group_bar_mean.eps'), bbox_inches="tight")
            plt.close('all')

        for cc, col_ler in enumerate(columns_ler):
            fig9 = plt.figure()
            for gr, group in enumerate(groups_labels):
                # [h,b] = np.histogram(shap_group[cc,:,gr], bins_grshap)
                # plt.bar(b[1:], h, width=0.0005, label=group)
                sns.kdeplot(shap_group[cc, :, gr], shade=True, label=groups_labels[gr])

            plt.legend()
            plt.xlim([-0.015, 0.025])
            #             plt.ylim([0,200])
            fig9.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig9.savefig(os.path.join(folder_plots_gro, col_ler + '_group_kde_mean.png'), bbox_inches="tight")
            fig9.savefig(os.path.join(folder_plots_gro, col_ler + '_group_kde_mean.eps'), bbox_inches="tight")
            plt.close('all')

        groups_labels_sum = np.asarray(['Position', 'STD', 'SNR', 'Others', 'Connectivity'])
        if classif:
            groups_index_sum = [np.arange(0, 12), np.arange(12, 22), np.arange(22, 25), \
                                np.asarray([0, 25]), np.arange(26, 42)]
        else:
            groups_index_sum = [np.arange(1, 13), np.arange(13, 23), np.arange(23, 26), \
                                np.asarray([0, 26]), np.arange(27, 43)]

        for cc, col_ler in enumerate(columns_ler):
            for gr, group in enumerate(groups_labels_sum):
                fig10 = plt.figure()
                aux_list = list(groups_index_sum[gr])
                aux_df = np.where(~np.isnan(np.sum(shap_experiment_mean[cc, :, aux_list].T, 1)))[
                    0]  # groups_index changes the dimension position???
                shap.summary_plot(shap_experiment_mean[cc, :, groups_index_sum[gr]][:, aux_df].T, \
                                  df.iloc[aux_df][np.asarray(labels_to_study)[groups_index_sum[gr]]])

                plt.xlim([-0.04, 0.06])
                fig10.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig10.savefig(os.path.join(folder_plots_gro, col_ler + '_' + groups_labels_sum[gr] + '_group_summ.png'),
                              bbox_inches="tight")
                fig10.savefig(os.path.join(folder_plots_gro, col_ler + '_' + groups_labels_sum[gr] + '_group_summ.eps'),
                              bbox_inches="tight")
                plt.close('all')

        folder_plots_ITPT_f = os.path.join(folder_main, 'plots', 'XGBoost', 'ITPT', 'features')
        print('plotting ITPT GC features')
        if classif:
            GC_index = np.arange(26, 42)
            index = np.arange(26, 42)
        else:
            GC_index = np.arange(27, 43)
            index = np.arange(27, 43)
        for cc, col_ler in enumerate(columns_ler):
            aux_ITshap = []
            aux_PTshap = []
            df_ITshap = df.loc[df['ITPTlabel'] == 0]
            df_PTshap = df.loc[df['ITPTlabel'] == 1]
            for ind in index:
                aux_IT = shap_experiment_mean[cc, df['ITPTlabel'] == 0, ind]
                aux_PT = shap_experiment_mean[cc, df['ITPTlabel'] == 1, ind]
                aux_ITshap.append(aux_IT)
                aux_PTshap.append(aux_PT)
            aux_ITshap = np.asarray(aux_ITshap)
            aux_PTshap = np.asarray(aux_PTshap)

            fig11 = plt.figure()
            shap.summary_plot(aux_ITshap.T, df_ITshap[np.asarray(labels_to_study)[GC_index]])
            plt.xlim([-0.04, 0.06])
            fig11.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig11.savefig(os.path.join(folder_plots_gro, col_ler + '_IT_con_summ.png'), bbox_inches="tight")
            fig11.savefig(os.path.join(folder_plots_gro, col_ler + '_IT_con_summ.eps'), bbox_inches="tight")
            plt.close('all')

            fig12 = plt.figure()
            shap.summary_plot(aux_PTshap.T, df_PTshap[np.asarray(labels_to_study)[GC_index]])
            plt.xlim([-0.04, 0.06])
            fig12.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig12.savefig(os.path.join(folder_plots_gro, col_ler + '_PT_con_summ.png'), bbox_inches="tight")
            fig12.savefig(os.path.join(folder_plots_gro, col_ler + '_PT_con_summ.eps'), bbox_inches="tight")
            plt.close('all')

            labels_to_study = labels_to_study.tolist()
            indexei = labels_to_study.index('GC_per_ens_ind')
            indexer = labels_to_study.index('GC_per_ens_red')
            indexie = labels_to_study.index('GC_per_ind_ens')
            indexre = labels_to_study.index('GC_per_red_ens')
            aux_ITei = shap_experiment_mean[cc, df['ITPTlabel'] == 0, indexei]
            aux_PTei = shap_experiment_mean[cc, df['ITPTlabel'] == 1, indexei]
            aux_ITer = shap_experiment_mean[cc, df['ITPTlabel'] == 0, indexer]
            aux_PTer = shap_experiment_mean[cc, df['ITPTlabel'] == 1, indexer]
            aux_ITie = shap_experiment_mean[cc, df['ITPTlabel'] == 0, indexie]
            aux_PTie = shap_experiment_mean[cc, df['ITPTlabel'] == 1, indexie]
            aux_ITre = shap_experiment_mean[cc, df['ITPTlabel'] == 0, indexre]
            aux_PTre = shap_experiment_mean[cc, df['ITPTlabel'] == 1, indexre]
            index_IT = df['ITPTlabel'] == 0
            index_PT = df['ITPTlabel'] == 1
            aux_df_ITei = np.asarray(df['GC_per_ens_ind'][index_IT])
            aux_df_PTei = np.asarray(df['GC_per_ens_ind'][index_PT])
            aux_df_ITer = np.asarray(df['GC_per_ens_red'][index_IT])
            aux_df_PTer = np.asarray(df['GC_per_ens_red'][index_PT])
            aux_df_ITie = np.asarray(df['GC_per_ind_ens'][index_IT])
            aux_df_PTie = np.asarray(df['GC_per_ind_ens'][index_PT])
            aux_df_ITre = np.asarray(df['GC_per_red_ens'][index_IT])
            aux_df_PTre = np.asarray(df['GC_per_red_ens'][index_PT])

            fig13 = plt.figure(figsize=(12, 8))
            ax1 = fig13.add_subplot(2, 2, 1)
            ax2 = fig13.add_subplot(2, 2, 2)
            ax1.bar([0, 0.4], [np.nanmean(aux_ITei), np.nanmean(aux_ITer)], width=0.3,
                    yerr=[pd.DataFrame(aux_ITei).sem(0)[0], pd.DataFrame(aux_ITer).sem(0)[0]])
            ax1.bar([1, 1.4], [np.nanmean(aux_PTei), np.nanmean(aux_PTer)], width=0.3,
                    yerr=[pd.DataFrame(aux_PTei).sem(0)[0], pd.DataFrame(aux_PTer).sem(0)[0]])
            ax2.bar([0, 0.4], [np.nanmean(aux_ITie), np.nanmean(aux_ITre)], width=0.3,
                    yerr=[pd.DataFrame(aux_ITie).sem(0)[0], pd.DataFrame(aux_ITre).sem(0)[0]])
            ax2.bar([1, 1.4], [np.nanmean(aux_PTie), np.nanmean(aux_PTre)], width=0.3,
                    yerr=[pd.DataFrame(aux_PTie).sem(0)[0], pd.DataFrame(aux_PTre).sem(0)[0]])
            ax1.set_ylabel('GC shap Ens -> X')
            ax2.set_ylabel('GC shap X -> Ens')
            ax1.set_xlabel(['ITei', 'ITer', 'PTei', 'PTer'])
            ax2.set_xlabel(['ITie', 'ITre', 'PTie', 'PTre'])
            _, p_value = stats.ttest_ind(aux_ITei, aux_ITer, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax1.text(0.2, 0, p, color='grey')
            _, p_value = stats.ttest_ind(aux_PTei, aux_PTer, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax1.text(1.2, 0, p, color='grey')
            _, p_value = stats.ttest_ind(aux_ITie, aux_ITre, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.2, 0, p, color='grey')
            _, p_value = stats.ttest_ind(aux_PTie, aux_PTre, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(1.2, 0, p, color='grey')

            ax3 = fig13.add_subplot(2, 2, 3)
            ax4 = fig13.add_subplot(2, 2, 4)
            ax3.bar([0, 0.4], [np.nanmean(aux_df_ITei), np.nanmean(aux_df_ITer)], width=0.3,
                    yerr=[pd.DataFrame(aux_df_ITei).sem(0)[0], pd.DataFrame(aux_df_ITer).sem(0)[0]])
            ax3.bar([1, 1.4], [np.nanmean(aux_df_PTei), np.nanmean(aux_df_PTer)], width=0.3,
                    yerr=[pd.DataFrame(aux_df_PTei).sem(0)[0], pd.DataFrame(aux_df_PTer).sem(0)[0]])
            ax4.bar([0, 0.4], [np.nanmean(aux_df_ITie), np.nanmean(aux_df_ITre)], width=0.3,
                    yerr=[pd.DataFrame(aux_df_ITie).sem(0)[0], pd.DataFrame(aux_df_ITre).sem(0)[0]])
            ax4.bar([1, 1.4], [np.nanmean(aux_df_PTie), np.nanmean(aux_df_PTre)], width=0.3,
                    yerr=[pd.DataFrame(aux_df_PTie).sem(0)[0], pd.DataFrame(aux_df_PTre).sem(0)[0]])
            ax3.set_ylabel('GC per  Ens -> X')
            ax4.set_ylabel('GC per  X -> Ens')
            ax3.set_xlabel(['ITei', 'ITer', 'PTei', 'PTer'])
            ax4.set_xlabel(['ITie', 'ITre', 'PTie', 'PTre'])
            _, p_value = stats.ttest_ind(aux_df_ITei, aux_df_ITer, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax3.text(0.2, 0, p, color='grey')
            _, p_value = stats.ttest_ind(aux_df_PTei, aux_df_PTer, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax3.text(1.2, 0, p, color='grey')
            _, p_value = stats.ttest_ind(aux_df_ITie, aux_df_ITre, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax4.text(0.2, 0, p, color='grey')
            _, p_value = stats.ttest_ind(aux_df_PTie, aux_df_PTre, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax4.text(1.2, 0, p, color='grey')
            fig13.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig13.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig13.savefig(os.path.join(folder_plots_ITPT_f, col_ler + '_GC_ens_ind_red_ALPTIT.png'),
                          bbox_inches="tight")
            fig13.savefig(os.path.join(folder_plots_ITPT_f, col_ler + '_GC_ens_ind_red_ALPTIT.eps'),
                          bbox_inches="tight")
            plt.close('all')

        # ************************************************************************************************************************
        # check for all models together! ***************************************************************************************
        # **************************************************************************************************************************
        print('plotting MODEL Stuff')
        folder_plots_ITPTm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'ITPT')
        if not os.path.exists(folder_plots_ITPTm):
            os.makedirs(folder_plots_ITPTm)
        all_ITm = np.zeros((len(columns_ler), mod_n)) + np.nan
        all_PTm = np.zeros((len(columns_ler), mod_n)) + np.nan
        for cc, col_ler in enumerate(columns_ler):
            aux_ind_IT = np.zeros((mod_n, all_shap.shape[2]), dtype='bool')
            aux_ind_PT = np.zeros((mod_n, all_shap.shape[2]), dtype='bool')
            for i in np.arange(mod_n):
                aux_df = all_df[cc, i, ~np.isnan(all_df[cc, i, :])]
                aux_ind_IT[i, :aux_df.shape[0]] = df.iloc[aux_df]['ITPTlabel'] == 0
                aux_ind_PT[i, :aux_df.shape[0]] = df.iloc[aux_df]['ITPTlabel'] == 1
                all_ITm[cc, i] = np.nanmean(all_shap[cc, i, aux_ind_IT[i, :], 0])
                all_PTm[cc, i] = np.nanmean(all_shap[cc, i, aux_ind_PT[i, :], 0])

            fig2 = plt.figure(figsize=(12, 4))
            ax1 = fig2.add_subplot(1, 2, 1)
            [h_IT, b] = np.histogram(all_shap[cc, aux_ind_IT, 0], bins_shap)
            [h_PT, b] = np.histogram(all_shap[cc, aux_ind_PT, 0], bins_shap)
            ax1.bar(b[1:], h_IT, width=0.001, label='IT')
            ax1.bar(b[1:], h_PT, width=0.001, label='PT')
            ax1.legend()

            ax2 = fig2.add_subplot(1, 2, 2)
            ax2.bar([0.4, 1.4], [np.nanmean(all_ITm[cc, :]), np.nanmean(all_PTm[cc, :])], width=0.8, \
                    yerr=[pd.DataFrame(all_ITm[cc, :]).sem(0).values[0], pd.DataFrame(all_PTm[cc, :]).sem(0).values[0]], \
                    error_kw=dict(ecolor='k'))
            ax2.set_xticks([0.4, 1.4])
            ax2.set_xticklabels(['IT', 'PT'])
            _, p_value = stats.ttest_ind(all_ITm[cc, :], all_PTm[cc, :], nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.8, 0.008, p, color='grey', alpha=0.6)
            ax2.set_ylim([-0.03, 0.03])

            fig2.savefig(os.path.join(folder_plots_ITPTm, col_ler + '_ITPT_shap_val.png'), bbox_inches="tight")
            fig2.savefig(os.path.join(folder_plots_ITPTm, col_ler + '_ITPT_shap_val.eps'), bbox_inches="tight")
            plt.close('all')

        folder_plots_shapm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'shap')
        if not os.path.exists(folder_plots_shapm):
            os.makedirs(folder_plots_shapm)
        bins_shap = np.arange(-0.05, 0.05, 0.001)

        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc, :])
            aux_df = all_df_reshape[cc, aux_ind]

            shap_reshape = all_shap_reshape[cc, aux_ind, :]
            fig3 = plt.figure()
            shap.summary_plot(shap_reshape, df.iloc[aux_df][labels_to_study])

            fig3.savefig(os.path.join(folder_plots_shapm, col_ler + '_summary.png'), bbox_inches="tight")
            fig3.savefig(os.path.join(folder_plots_shapm, col_ler + '_summary.eps'), bbox_inches="tight")
            plt.close('all')

        # check dependencies
        all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
        all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])])

        folder_plots_dependm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'dependencies')
        if not os.path.exists(folder_plots_dependm):
            os.makedirs(folder_plots_dependm)
        sizesubpl = np.ceil(len(labels_to_study) / 6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc, :])
            aux_df = all_df_reshape[cc, aux_ind]
            shap_reshape = all_shap_reshape[cc, aux_ind, :]
            for ll, label_ts in enumerate(labels_to_study):
                fig4 = plt.figure(figsize=(17, 9))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax3 = fig4.add_subplot(sizesubpl, 6, llsec + 1)
                    shap.dependence_plot(label_ts, shap_reshape, df.iloc[aux_df][labels_to_study],
                                         interaction_index=llsec, ax=ax3)
                fig4.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig4.savefig(os.path.join(folder_plots_dependm, col_ler + '_' + label_ts + '_depen.png'),
                             bbox_inches="tight")
                fig4.savefig(os.path.join(folder_plots_dependm, col_ler + '_' + label_ts + '_depen.eps'),
                             bbox_inches="tight")
                plt.close('all')

        # check regression inter feature
        all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
        all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])])

        folder_plots_reg_featm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'regression_feat')
        if not os.path.exists(folder_plots_reg_featm):
            os.makedirs(folder_plots_reg_featm)
        sizesubpl = np.ceil(len(labels_to_study) / 6).astype('int')
        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc, :])
            aux_df = all_df_reshape[cc, aux_ind]
            shap_reshape = all_shap_reshape[cc, aux_ind, :]
            for ll, label_ts in enumerate(labels_to_study):
                fig5 = plt.figure(figsize=(17, 9))
                for llsec, label_tsec in enumerate(labels_to_study):
                    ax4 = fig5.add_subplot(sizesubpl, 6, llsec + 1)
                    sns.regplot(df.iloc[aux_df][label_ts], df.iloc[aux_df][label_tsec], ax=ax4)
                    ax4.set_ylabel(label_tsec)
                fig5.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
                fig5.savefig(os.path.join(folder_plots_reg_featm, col_ler + '_' + label_ts + '_regfet.png'),
                             bbox_inches="tight")
                fig5.savefig(os.path.join(folder_plots_reg_featm, col_ler + '_' + label_ts + '_regfet.eps'),
                             bbox_inches="tight")
                plt.close('all')

                # labels to study regression to shap
        all_shap_reshape = np.reshape(all_shap, [len(columns_ler), np.prod(all_shap.shape[1:3]), all_shap.shape[3]])
        all_df_reshape = np.reshape(all_df, [len(columns_ler), np.prod(all_df.shape[1:3])])

        folder_plots_regm = os.path.join(folder_main, 'plots', 'XGBoost', 'model', 'regression_shap')
        if not os.path.exists(folder_plots_regm):
            os.makedirs(folder_plots_regm)
        sizesubpl = np.ceil(len(labels_to_study) / 6).astype('int')

        for cc, col_ler in enumerate(columns_ler):
            aux_ind = ~np.isnan(all_df_reshape[cc, :])
            aux_df = all_df_reshape[cc, aux_ind]
            shap_reshape = all_shap_reshape[cc, aux_ind, :]
            fig6 = plt.figure(figsize=(17, 9))
            for ll, label_ts in enumerate(labels_to_study):
                ax6 = fig6.add_subplot(sizesubpl, 6, ll + 1)
                sns.regplot(df.iloc[aux_df][label_ts], shap_reshape[:, ll], ax=ax6)
                ax6.set_ylabel('shap_val')
            fig6.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
            fig6.savefig(os.path.join(folder_plots_regm, col_ler + '_reg.png'), bbox_inches="tight")
            fig6.savefig(os.path.join(folder_plots_regm, col_ler + '_reg.eps'), bbox_inches="tight")
            plt.close('all')


def plot_ITPT(df, folder_main, labels_to_study, shap_experiment_mean):
    ''' 
    To plot IT-PT differences
    '''
    columns = df.columns.tolist()
    columns_ler = [columns[6]]
    folder_plots_ITPT_f = os.path.join(folder_main, 'plots', 'XGBoost', 'ITPT', 'features')
    if not os.path.exists(folder_plots_ITPT_f):
        os.makedirs(folder_plots_ITPT_f)
    all_IT = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_PT = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_df_IT = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_df_PT = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    index_IT = df['ITPTlabel'] == 0
    index_PT = df['ITPTlabel'] == 1
    bins_shap = np.arange(-0.04, 0.06, 0.001)

    for cc, col_ler in enumerate(columns_ler):
        for ll, label in enumerate(labels_to_study):
            aux_IT = shap_experiment_mean[cc, index_IT, ll]
            aux_PT = shap_experiment_mean[cc, index_PT, ll]
            aux_df_IT = np.asarray(df[label][index_IT])
            aux_df_PT = np.asarray(df[label][index_PT])
            all_IT[cc, ll] = np.nanmean(aux_IT)
            all_PT[cc, ll] = np.nanmean(aux_PT)
            all_df_IT[cc, ll] = np.nanmean(aux_df_IT)
            all_df_PT[cc, ll] = np.nanmean(aux_df_PT)

            fig1 = plt.figure(figsize=(12, 8))
            ax1 = fig1.add_subplot(2, 2, 1)
            [h_IT, b] = np.histogram(aux_IT, bins_shap)
            [h_PT, b] = np.histogram(aux_PT, bins_shap)
            ax1.bar(b[1:], h_IT, width=0.001, label='IT')
            ax1.bar(b[1:], h_PT, width=0.001, label='PT')
            ax1.set_xlabel('SHAP values')
            ax1.legend()

            ax2 = fig1.add_subplot(2, 2, 2)
            ax2.bar([0.4, 1.4], [all_IT[cc, ll], all_PT[cc, ll]], width=0.8, \
                    yerr=[pd.DataFrame(aux_IT).sem(0).values[0], pd.DataFrame(aux_PT).sem(0).values[0]], \
                    error_kw=dict(ecolor='k'))
            ax2.set_xticks([0.4, 1.4])
            ax2.set_xticklabels(['IT', 'PT'])
            ax2.set_ylabel('SHAP values')
            _, p_value = stats.ttest_ind(aux_IT, aux_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.8, 0.001, p, color='grey', alpha=0.6)

            ax3 = fig1.add_subplot(2, 2, 3)
            bins_feat = np.linspace(np.nanmin([np.nanmin(aux_df_IT), np.nanmin(aux_df_PT)]), \
                                    np.nanmax([np.nanmax(aux_df_IT), np.nanmax(aux_df_PT)]), 100)
            [h_IT, b] = np.histogram(aux_df_IT, bins_feat)
            [h_PT, b] = np.histogram(aux_df_PT, bins_feat)
            ax3.bar(b[1:], h_IT, width=np.diff(bins_feat)[0], label='IT')
            ax3.bar(b[1:], h_PT, width=np.diff(bins_feat)[0], label='PT')
            ax3.set_xlabel(label)
            ax3.legend()

            ax4 = fig1.add_subplot(2, 2, 4)
            ax4.bar([0.4, 1.4], [all_df_IT[cc, ll], all_df_PT[cc, ll]], width=0.8, \
                    yerr=[pd.DataFrame(aux_df_IT).sem(0).values[0], pd.DataFrame(aux_df_PT).sem(0).values[0]], \
                    error_kw=dict(ecolor='k'))
            ax4.set_xticks([0.4, 1.4])
            ax4.set_xticklabels(['IT', 'PT'])
            _, p_value = stats.ttest_ind(aux_df_IT, aux_df_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax4.text(0.8, all_df_IT[cc, ll], p, color='grey', alpha=0.6)
            ax4.set_ylabel(label)

            fig1.savefig(os.path.join(folder_plots_ITPT_f, col_ler + '_' + label + '_ITPT.png'), bbox_inches="tight")
            fig1.savefig(os.path.join(folder_plots_ITPT_f, col_ler + '_' + label + '_ITPT.svg'), format='svg',
                         bbox_inches="tight")
            plt.close('all')


def spread_normal(spread):
    spread = np.squeeze(spread)
    fig1 = plt.figure()
    ktest = []
    ktestpval = []
    for i in np.arange(spread.shape[0]):
        for j in np.arange(spread.shape[1]):
            kt, pval = stats.kstest(spread[i, j, :], 'norm')
            ktest.append(kt)
            ktestpval.append(pval)


def ITCC_CSTR(folder_main):
    folder_plots_ITPT_cc = os.path.join(folder_main, 'plots', 'XGBoost', 'ITPT', 'CC_CSTR')
    if not os.path.exists(folder_plots_ITPT_cc):
        os.makedirs(folder_plots_ITPT_cc)
    columns = df.columns.tolist()
    columns_ler = [columns[6]]

    all_ITcc = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_ITcs = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_PT = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_df_ITcc = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_df_ITcs = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    all_df_PT = np.zeros((len(columns_ler), len(labels_to_study))) + np.nan
    index_ITcs = (df['depth_min'] > 400) & (df['ITPTlabel'] == 0)
    index_ITcc = (df['depth_max'] < 400) & (df['ITPTlabel'] == 0)
    index_PT = df['ITPTlabel'] == 1
    index_IT = df['ITPTlabel'] == 0
    bins_shap = np.arange(-0.04, 0.06, 0.001)

    for cc, col_ler in enumerate(columns_ler):
        for ll, label in enumerate(labels_to_study):
            aux_ITcc = shap_experiment_mean[cc, index_ITcc, ll]
            aux_ITcs = shap_experiment_mean[cc, index_ITcs, ll]
            aux_PT = shap_experiment_mean[cc, index_PT, ll]
            aux_df_ITcc = np.asarray(df[label][index_ITcc])
            aux_df_ITcs = np.asarray(df[label][index_ITcs])
            aux_df_PT = np.asarray(df[label][index_PT])
            all_ITcc[cc, ll] = np.nanmean(aux_ITcc)
            all_ITcs[cc, ll] = np.nanmean(aux_ITcs)
            all_PT[cc, ll] = np.nanmean(aux_PT)
            all_df_ITcc[cc, ll] = np.nanmean(aux_df_ITcc)
            all_df_ITcs[cc, ll] = np.nanmean(aux_df_ITcs)
            all_df_PT[cc, ll] = np.nanmean(aux_df_PT)

            fig1 = plt.figure(figsize=(12, 8))
            ax1 = fig1.add_subplot(2, 2, 1)
            [h_ITcc, b] = np.histogram(aux_ITcc, bins_shap)
            [h_PT, b] = np.histogram(aux_PT, bins_shap)
            [h_ITcs, b] = np.histogram(aux_ITcs, bins_shap)
            ax1.bar(b[1:], h_ITcc, width=0.001, label='ITcc')
            ax1.bar(b[1:], h_PT, width=0.001, label='PT')
            ax1.bar(b[1:], h_ITcs, width=0.001, label='ITcs')
            ax1.set_xlabel('SHAP values')
            ax1.legend()

            ax2 = fig1.add_subplot(2, 2, 2)
            ax2.bar([0.4, 1.4, 2.4], [all_ITcc[cc, ll], all_ITcs[cc, ll], all_PT[cc, ll]], width=0.8, \
                    yerr=[pd.DataFrame(aux_ITcc).sem(0).values[0], pd.DataFrame(aux_ITcs).sem(0).values[0], \
                          pd.DataFrame(aux_PT).sem(0).values[0]], error_kw=dict(ecolor='k'))
            ax2.set_xticks([0.4, 1.4, 2.4])
            ax2.set_xticklabels(['ITcc', 'ITcs', 'PT'])
            ax2.set_ylabel('SHAP values')
            _, p_value = stats.ttest_ind(aux_ITcc, aux_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.8, 0.001, p, color='grey', alpha=0.6)
            _, p_value = stats.ttest_ind(aux_ITcs, aux_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(1.8, 0.001, p, color='grey', alpha=0.6)
            _, p_value = stats.ttest_ind(aux_ITcc, aux_ITcs, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax2.text(0.4, -0.00001, p, color='grey', alpha=0.6)

            ax3 = fig1.add_subplot(2, 2, 3)
            bins_feat = np.linspace(np.nanmin([np.nanmin(aux_df_ITcc), np.nanmin(aux_df_ITcs), np.nanmin(aux_df_PT)]), \
                                    np.nanmax([np.nanmax(aux_df_ITcc), np.nanmax(aux_df_ITcs), np.nanmax(aux_df_PT)]),
                                    100)
            [h_ITcc, b] = np.histogram(aux_df_ITcc, bins_feat)
            [h_PT, b] = np.histogram(aux_df_PT, bins_feat)
            [h_ITcs, b] = np.histogram(aux_df_ITcs, bins_feat)
            ax3.bar(b[1:], h_ITcc, width=np.diff(bins_feat)[0], label='ITcc')
            ax3.bar(b[1:], h_PT, width=np.diff(bins_feat)[0], label='PT')
            ax3.bar(b[1:], h_ITcs, width=np.diff(bins_feat)[0], label='ITcs')
            ax3.set_xlabel(label)
            ax3.legend()

            ax4 = fig1.add_subplot(2, 2, 4)
            ax4.bar([0.4, 1.4, 2.4], [all_df_ITcc[cc, ll], all_df_ITcs[cc, ll], all_df_PT[cc, ll]], width=0.8, \
                    yerr=[pd.DataFrame(aux_df_ITcc).sem(0).values[0], pd.DataFrame(aux_df_ITcs).sem(0).values[0], \
                          pd.DataFrame(aux_df_PT).sem(0).values[0]], error_kw=dict(ecolor='k'))
            ax4.set_xticks([0.4, 1.4, 2.4])
            ax4.set_xticklabels(['ITcc', 'ITcs', 'PT'])
            _, p_value = stats.ttest_ind(aux_df_ITcc, aux_df_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax4.text(0.8, all_df_ITcc[cc, ll], p, color='grey', alpha=0.6)
            _, p_value = stats.ttest_ind(aux_df_ITcs, aux_df_PT, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax4.text(1.8, all_df_ITcs[cc, ll], p, color='grey', alpha=0.6)
            _, p_value = stats.ttest_ind(aux_df_ITcc, aux_df_ITcs, nan_policy='omit')
            p = uc.calc_pvalue(p_value)
            ax4.text(0.4, all_df_ITcc[cc, ll], p, color='grey', alpha=0.6)
            ax4.set_ylabel(label)

            fig1.savefig(os.path.join(folder_plots_ITPT_cc, col_ler + '_' + label + '_ITPT.png'), bbox_inches="tight")
            fig1.savefig(os.path.join(folder_plots_ITPT_cc, col_ler + '_' + label + '_ITPT.eps'), bbox_inches="tight")
            plt.close('all')
