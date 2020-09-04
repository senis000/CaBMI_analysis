#*************************************************************************
#*****************************ALL TOGETHER********************************
#*************************************************************************

__author__ = 'Nuria'


import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import interactive
interactive(True)

import utils_cabmi as ut
import analysis_cabmi as acb     



def all_learning(folder, animals=['GCP1', 'GCP2'], var='PT', sec_var='',
        maxf=36000, len_days=20, smoth=1, min_bin=5, trials=150):
    fpath = folder +  'raw/' + animal + '/'
    folder_anal = folder +  'analysis/learning/' 
    
    flag_init = True
    pc_all = np.ones((len(animals), len_days)) * np.nan
    
    for aa, animal in enumerate(animals):
        for dd, day in enumerate(os.listdir(fpath)):
            
            if flag_init:
                f = h5py.File(folder_path + 'full_' + animal + '_' + day + '_' +
                    sec_var + '_data.hdf5', 'r'
                    ) 
                fr = f['fr']
                f.close()
                bins = np.arange(0, maxf/fr, 60)
                hpm_all = np.ones((len(animals), len_days, bins.shape[0]))*np.nan
                tth_all = np.ones((len(animals), len_days, bins.shape[0]))*np.nan
                flag_init = False
            [hpm, tth, pc] = acb.learning(folder, animal, day, sec_var, toplot=False)
            pc_all[aa,dd] = pc
            if hpm.shape[0]<hpm_all.shape[1]:
                hpm_all[aa, dd, :hpm.shape[0]] = hpm
                tth_all[aa, dd, :tth.shape[0]] = tth
            else:
                hpm_all[aa, dd, :] = hpm
                tth_all[aa, dd, :] = tth
                
    fig1 = plt.figure(figsize=(18, 5))
    bx = fig1.add_subplot(131)
    # make up of figure
    bx.spines["top"].set_visible(False)
    bx.spines["right"].set_visible(False)
    bx.get_yaxis().tick_left()
    bx.set_ylim(0.4, 0.95)
    bx.set_xlim(0.8, 10.5)
    bx.set_xlabel('day', fontsize=16)
    bx.set_ylabel('percentage correct', fontsize=16)
    # end of figure makeup
    if smoth == 1:
        bx.errorbar(np.arange(1, len_days + 1), np.nanmean(pc_all, 0),
            yerr=pd.DataFrame(pc_all).sem(0).values, color='k'
            )
    else:
        bx.errorbar(np.arange(1, len_days + 1),
            sliding_mean(np.nanmean(pc_all, 0), smoth),
            yerr=pd.DataFrame(pc_all).sem(0).values, color='k'
            )

    bx0 = fig1.add_subplot(132)
    # make up of figure
    bx0.spines["top"].set_visible(False)
    bx0.spines["right"].set_visible(False)
    bx0.get_yaxis().tick_left()
    bx0.set_ylim(0.3, 7)
    bx0.set_xlim(0.8, 10.5)
    bx0.set_xlabel('day', fontsize=16)
    bx0.set_ylabel('Hits/min', fontsize=16)
    # end of figure makeup
    aux_hpm = np.nanmean(hpm_all, 2)
    if smoth == 1:
        bx0.errorbar(np.arange(1, len_days + 1),
            np.nanmean(aux_hpm, 0), yerr=pd.DataFrame(aux_hpm).sem(0).values,
            color='k'
            )
    else:
        bx0.errorbar(np.arange(1, len_days + 1),
            ut.sliding_mean(np.nanmean(aux_hpm, 0), smoth),
            yerr=pd.DataFrame(aux_hpm).sem(0).values, color='k'
            )

    bx1 = fig1.add_subplot(133)
    # make up of figure
    bx1.spines["top"].set_visible(False)
    bx1.spines["right"].set_visible(False)
    bx1.get_yaxis().tick_left()
    bx1.set_ylim(0.3, 7)
    bx1.set_xlim(-0.5, 25.1)
    bx1.set_xlabel('Time in session (min)', fontsize=16)
    bx1.set_ylabel('Hits/min', fontsize=16)
    # end of figure makeup
    aux_tothpm = np.reshape(hpm_all, [hpm_all.shape[0]*hpm_all.shape[1],
        hpm_all.shape[2]]
        )
    _, p_value = stats.ttest_ind(
        np.reshape(aux_tothpm[:, :min_bin],
        aux_tothpm.shape[0]*min_bin),
        np.reshape(aux_tothpm[:, -min_bin:], aux_tothpm.shape[0]*min_bin),
        nan_policy='omit'
        )
    p = calc_pvalue(p_value)
    bx1.errorbar(np.arange(maxf/fr), np.nanmean(aux_tothpm, 0),
        yerr=pd.DataFrame(aux_tothpm).sem(0).values, color='k'
        )

    bx1.text(len_exper/2, 1.2, p, color='grey', alpha=0.6)
    bx1.axhline(
        y=1.3, xmin=0, xmax=1/maxf/fr*min_bin,
        c='grey', linewidth=0.5, alpha=0.5
        )
    bx1.axhline(
        y=1.3, xmin=1-1/maxf/fr*min_bin, xmax=1,
        c='grey', linewidth=0.5, alpha=0.5
        )
    bx1.axhline(
        y=1.2, xmin=1/maxf/fr*min_bin/2, xmax=1-1/maxf/fr*min_bin/2,
        c='grey', linewidth=0.5, alpha=0.5
        )
    fig1.savefig(folder_path + 'percent_correct_min.png', bbox_inches="tight")
    fig1.savefig(
        folder_path + 'percent_correct_min.eps', format='eps',
        bbox_inches="tight"
        )

    fig2 = plt.figure(figsize=(18, 5))
    bx2 = fig2.add_subplot(131)
    # make up of figure
    bx2.spines["top"].set_visible(False)
    bx2.spines["right"].set_visible(False)
    bx2.get_yaxis().tick_left()
    bx2.set_xlim(0.5, 10.5)
    bx2.set_ylim(4, 18)
    bx2.set_xlabel('Day', fontsize=16)
    bx2.set_ylabel('time to hit', fontsize=16)
    bx2.set_title('Averaged days')
    aux_tit = np.nanmean(tth_all, 2)

    if smoth == 1:
        bx2.errorbar(
            np.arange(1, len_days + 1), np.nanmean(aux_tit, 0),
            yerr=pd.DataFrame(aux_tit).sem(0).values, color='k'
            )
    else:
        bx2.errorbar(
            np.arange(1, len_days + 1),
            ut.sliding_mean(np.nanmean(aux_tit, 0), smoth),
            yerr=pd.DataFrame(aux_tit).sem(0).values, color='k'
            )
    _, p_value = stats.ttest_ind(aux_tit[:, 0], aux_tit[:, -1], nan_policy='omit')
    p = ut.calc_pvalue(p_value)
    bx2.text(len_days/2, 5, p, color='grey', alpha=0.6)
    bx2.axhline(
        y=5, xmin=0, xmax=1/len_exper*min_bin,
        c='grey', linewidth=0.5, alpha=0.5
        )
    bx2.axhline(
        y=5, xmin=1-1/len_days, xmax=1,
        c='grey', linewidth=0.5, alpha=0.5
        )
    bx2.axhline(
        y=4.9, xmin=1/len_days, xmax=1-1/len_exper*min_bin/2,
        c='grey', linewidth=0.5, alpha=0.5
        )

    bx3 = fig2.add_subplot(132)
    # make up of figure
    bx3.spines["top"].set_visible(False)
    bx3.spines["right"].set_visible(False)
    bx3.get_yaxis().tick_left()
    bx3.set_xlabel('Trials', fontsize=16)
    bx3.set_ylabel('time to hit', fontsize=16)
    bx3.set_ylim(4, 18)
    bx3.set_xlim(0, 100)
    # end of figure makeup
    aux_timhit = np.reshape(tth_all, [len_days*len(animals), trials])
    sm_auxtim = ut.sliding_mean(np.nanmean(aux_timhit, 0), window=3)
    sm_error = ut.sliding_mean(pd.DataFrame(aux_timhit).sem(0).values, window=3)
    bx3.fill_between(
        np.arange(trials), sm_auxtim - sm_error, sm_auxtim + sm_error,
        color="lightgrey", alpha=0.7
        )
    bx3.plot(np.arange(trials), sm_auxtim, linewidth=2, c="k")
    _, p_value = stats.ttest_ind(
        np.reshape(aux_timhit[:, :int(trials/5)], aux_timhit.shape[0]*int(trials/5)),
        np.reshape(aux_timhit[:, int(4*trials/5):], aux_timhit.shape[0]*int(trials/5)),
        nan_policy='omit'
        )
    p = ut.calc_pvalue(p_value)
    bx3.text(50, 4.9, p, color='grey', alpha=0.6)
    bx3.axhline(y=4.9, xmin=0, xmax=0.2, c='grey', linewidth=0.5, alpha=0.5)
    bx3.axhline(y=4.9, xmin=0.8, xmax=1, c='grey', linewidth=0.5, alpha=0.5)
    bx3.axhline(y=5, xmin=0.1, xmax=0.9, c='grey', linewidth=0.5, alpha=0.5)

    bx4 = fig2.add_subplot(133)
    # make up of figure
    bx4.spines["top"].set_visible(False)
    bx4.spines["right"].set_visible(False)
    bx4.get_yaxis().tick_left()
    bx4.set_xlabel('Trials', fontsize=16)
    bx4.set_ylabel('time to hit', fontsize=16)
    bx4.set_xlim(0, 100)
    # end of figure makeup
    color_idx = np.linspace(0, 1, len_days)
    for xx in np.arange(0, len_days):
        aux_timhit = ut.sliding_mean(np.nanmean(tim_hit[:, xx, :], 0), 10)
        bx4.plot(np.arange(trials), aux_timhit, color=plt.cm.copper(color_idx[xx]))

    fig2.savefig(folder_path + 'tihit.png', bbox_inches="tight")
    fig2.savefig(folder_path + 'tihit.eps', format='eps', bbox_inches="tight")
    plt.close('all')


def all_tuning(folder, animals=['GCP1', 'GCP2'], sec_var='', len_days=20):
    fpath = folder +  'raw/' + animal + '/'
    folder_anal = folder +  'analysis/tunning/' 
    
    flag_init = True
    pc_all = np.ones((len(animals), len_days)) * np.nan
    
    for aa, animal in enumerate(animals):
        for dd, day in np.arange(len_days):
            f = h5py.File(
                folder_dest + 'tunning_' + animal + '_' + day + '_' +
                sec_var + '_freq.hdf5', 'w-'
                )
