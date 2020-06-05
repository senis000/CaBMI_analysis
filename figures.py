
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import scipy
import h5py
import pandas as pd
import copy
import seaborn as sns
import utils_cabmi as uc
from scipy import stats
from matplotlib import interactive

interactive(True)


def fig2():
    folder_main = 'I:/Nuria_data/CaBMI/Layer_project'
    out = 'I:/Nuria_data/CaBMI/Layer_project/learning_stats'
    file_csv = os.path.join(out, 'learning_stats_summary_bin_1.csv')
    file_csv_hpm = os.path.join(out, 'learning_stats_HPM_bin_1.csv')
    file_csv_PC = os.path.join(out, 'learning_stats_cumuPC_bin_1.csv')
#     file_csv_hpm = os.path.join(out, 'learning_stats_HPM_bin_5.csv')
#     file_csv_PC = os.path.join(out, 'learning_stats_PC_bin_5.csv')
    
    to_load_df = os.path.join(folder_main, 'df_all.hdf5')
    df = pd.read_hdf(to_load_df)
    df_hpm = pd.read_csv(file_csv_hpm)
    df_PC = pd.read_csv(file_csv_PC)
    df_results = pd.read_csv(file_csv)
    bins = np.arange(1,51,1)
    days = np.arange(1,16)
    
    
    #convert df_hpm to matrix same for PC
    folder = os.path.join(folder_main, 'processed')
    folder_plots = os.path.join(folder_main, 'plots', 'figures')
    animals = os.listdir(folder)
    hpm_5bin = np.zeros((len(animals), 76)) + np.nan
    PC_5bin = np.zeros((len(animals), 76)) + np.nan
    hpm_smoo = np.zeros((len(animals), 76)) + np.nan
    PC_smoo = np.zeros((len(animals), 76)) + np.nan
    hpm_gsmoo = np.zeros((len(animals), 76)) + np.nan
    PC_gsmoo = np.zeros((len(animals), 76)) + np.nan
    hpm_ses = np.zeros((len(animals), 15)) + np.nan
    PC_ses = np.zeros((len(animals), 15)) + np.nan
    hpm_g = np.zeros((len(animals), 76)) + np.nan
    PC_g = np.zeros((len(animals), 76)) + np.nan
    for aa, animal in enumerate(animals):
        # per session
        aux_hpm = df_results[df_results['animal']==animal].iloc[:,6].values
        aux_PC = df_results[df_results['animal']==animal].iloc[:,5].values
        if len(aux_hpm)<=hpm_ses.shape[1]:
            hpm_ses[aa,:len(aux_hpm)] = aux_hpm
        else:
            hpm_ses[aa,:] = aux_hpm[:hpm_ses.shape[1]]
        if len(aux_PC)<=PC_ses.shape[1]:
            PC_ses[aa,:len(aux_PC)] = aux_PC
        else:
            PC_ses[aa,:] = aux_PC[:PC_ses.shape[1]] 
        # per timebin
        hpm_5bin[aa,:] = np.nanmean(df_hpm[df_hpm['animal']==animal].iloc[:,3:].values,0)
        PC_5bin[aa,:] = np.nanmean(df_PC[df_PC['animal']==animal].iloc[:,3:].values,0)
        
        # per timebin smoothed
        hpm_smoo[aa,:] = uc.sliding_mean(hpm_5bin[aa,:], window=2)
        PC_smoo[aa,:] = uc.sliding_mean(PC_5bin[aa,:], window=2)
        # relative % increase
        hpm_g[aa,:] = (hpm_5bin[aa,:] - np.nanmean(hpm_5bin[aa,:]))/np.nanmean(hpm_5bin[aa,:])*100 - \
         (hpm_5bin[aa,0] - np.nanmean(hpm_5bin[aa,:]))/np.nanmean(hpm_5bin[aa,:])*100
        PC_g[aa,:] = (PC_5bin[aa,:] - np.nanmean(PC_5bin[aa,:]))/np.nanmean(PC_5bin[aa,:])*100 - \
         (PC_5bin[aa,0] - np.nanmean(PC_5bin[aa,:]))/np.nanmean(PC_5bin[aa,:])*100
        hpm_gsmoo[aa,:] = (hpm_smoo[aa,:] - np.nanmean(hpm_smoo[aa,:]))/np.nanmean(hpm_smoo[aa,:])*100 - \
         (hpm_smoo[aa,0] - np.nanmean(hpm_smoo[aa,:]))/np.nanmean(hpm_smoo[aa,:])*100
        PC_gsmoo[aa,:] = (PC_smoo[aa,:] - np.nanmean(PC_smoo[aa,:]))/np.nanmean(PC_smoo[aa,:])*100 - \
         (PC_smoo[aa,0] - np.nanmean(PC_smoo[aa,:]))/np.nanmean(PC_smoo[aa,:])*100
        
    #5 bin
#     fig1 = plt.figure(figsize=(8,4))
#     ax1 = fig1.add_subplot(1, 2, 1)
#     ax2 = fig1.add_subplot(1, 2, 2)
#     ax1.plot(np.nanmean(hpm_5bin[:9,:11],0))
#     ax1.plot(np.nanmean(hpm_5bin[9:,:11],0))
#     ax2.plot(np.nanmean(PC_5bin[:9,:11],0))
#     ax2.plot(np.nanmean(PC_5bin[9:,:11],0))
    
    
    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax2 = fig1.add_subplot(2, 2, 2)
    ax3 = fig1.add_subplot(2, 2, 3)
    ax4 = fig1.add_subplot(2, 2, 4)
    for aa, animal in enumerate(animals):
        if animal[:2]=='IT':
            ax1.plot(bins, hpm_smoo[aa,:50], 'r', linewidth=0.5, alpha=0.5)
            ax3.plot(bins, PC_5bin[aa,:50], 'r', linewidth=0.5, alpha=0.5)
        else:
            ax2.plot(bins, hpm_smoo[aa,:50], 'b', linewidth=0.5, alpha=0.5)
            ax4.plot(bins, PC_5bin[aa,:50], 'b', linewidth=0.5, alpha=0.5)
    ax1.plot(bins, np.nanmean(hpm_smoo[:9,:50],0),'r', linewidth=2)
    ax2.plot(bins, np.nanmean(hpm_smoo[9:,:50],0),'b', linewidth=2)
    ax3.plot(bins, np.nanmean(PC_5bin[:9,:50],0),'r', linewidth=2)
    ax4.plot(bins, np.nanmean(PC_5bin[9:,:50],0),'b', linewidth=2)
    ax1.set_ylim([0.3,1.6])
    ax2.set_ylim([0.3,1.6])
    ax3.set_ylim([0.1,0.5])
    ax4.set_ylim([0.1,0.5])
    ax1.set_ylabel('hpm')
    ax2.set_ylabel('hpm')
    ax3.set_ylabel('PC')
    ax4.set_ylabel('PC')
    ax3.set_xlabel('IT')
    ax4.set_xlabel('PT')
    fig1.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
    hpm_ci_IT = stats.norm.interval(0.90, loc=np.nanmean(hpm_smoo[:9,:50],0),scale=np.nanstd(hpm_smoo[:9,:50],0)/np.sqrt(hpm_smoo[:9,:50].shape[0]))
    hpm_ci_PT = stats.norm.interval(0.90, loc=np.nanmean(hpm_smoo[9:,:50],0),scale=np.nanstd(hpm_smoo[9:,:50],0)/np.sqrt(hpm_smoo[9:,:50].shape[0]))
    PC_ci_IT = stats.norm.interval(0.90, loc=np.nanmean(PC_smoo[:9,:50],0),scale=np.nanstd(PC_smoo[:9,:50],0)/np.sqrt(PC_smoo[:9,:50].shape[0]))
    PC_ci_PT = stats.norm.interval(0.90, loc=np.nanmean(PC_smoo[9:,:50],0),scale=np.nanstd(PC_smoo[9:,:50],0)/np.sqrt(PC_smoo[9:,:50].shape[0]))

   #all minutes
    fig2 = plt.figure(figsize=(8,8))
    bx1 = fig2.add_subplot(2, 2, 1)
    bx2 = fig2.add_subplot(2, 2, 2)
    bx3 = fig2.add_subplot(2, 2, 3)
    bx4 = fig2.add_subplot(2, 2, 4)
#     bx1.errorbar(bins[:16]+0.5, np.nanmean(PC_5bin[:9,:16],0), label='IT', yerr=pd.DataFrame(PC_5bin[:9,:16]).sem(0), c='r')
#     bx1.errorbar(bins[:16], np.nanmean(PC_5bin[9:,:16],0), label='PT', yerr=pd.DataFrame(PC_5bin[9:,:16]).sem(0), c='b')
    bx1.plot(bins[:50], np.nanmean(hpm_smoo[:9,:50],0), label='IT', c='r')
    bx1.fill_between(bins[:50], hpm_ci_IT[0][:50], hpm_ci_IT[1][:50], color='r', alpha=0.3)
    bx2.plot(bins[:50], np.nanmean(hpm_smoo[9:,:50],0), label='PT', c='b')
    bx2.fill_between(bins[:50], hpm_ci_PT[0][:50], hpm_ci_PT[1][:50], color='b', alpha=0.3)
    bx1.legend(loc=4)
    bx1.set_ylabel('HPM')
    bx1.set_xlabel('Time (min)')
    bx2.legend(loc=4)
    bx2.set_ylabel('HPM')
    bx2.set_xlabel('Time (min)')
    bx3.plot(bins[:50], np.nanmean(PC_smoo[:9,:50],0), label='IT', c='r')
    bx3.fill_between(bins[:50], PC_ci_IT[0][:50], PC_ci_IT[1][:50], color='r', alpha=0.3)
    bx4.plot(bins[:50], np.nanmean(PC_smoo[9:,:50],0), label='PT', c='b')
    bx4.fill_between(bins[:50], PC_ci_PT[0][:50], PC_ci_PT[1][:50], color='b', alpha=0.3)
    bx3.legend(loc=4)
    bx1.set_ylim([0.4,1.1])
    bx2.set_ylim([0.4,1.1])
    bx3.set_ylim([0.2,0.45])
    bx4.set_ylim([0.2,0.45])
    bx3.set_ylabel('PC')
    bx3.set_xlabel('Time (min)')
    bx4.legend(loc=4)
    bx4.set_ylabel('PC')
    bx4.set_xlabel('Time (min)')
    fig2.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

    # first 10min
    fig2 = plt.figure(figsize=(8,4))
    bx1 = fig2.add_subplot(1, 2, 1)
    bx2 = fig2.add_subplot(1, 2, 2)
    bx1.plot(bins[:10], np.nanmean(hpm_smoo[:9,:10],0), label='IT', c='r')
    bx1.plot(bins[:10]-0.1, hpm_smoo[:9,:10].T, 'r.', alpha=0.3)
    bx1.plot(bins[:10], np.nanmean(hpm_smoo[9:,:10],0), label='PT', c='b')
    bx1.plot(bins[:10]+0.1, hpm_smoo[9:,:10].T, 'b.', alpha=0.3)
    bx1.legend(loc=4)
    bx1.set_ylabel('HPM')
    bx1.set_xlabel('Time (min)')
    bx2.plot(bins[:10], np.nanmean(PC_smoo[:9,:10],0), label='IT', c='r')
    bx2.plot(bins[:10]-0.1, PC_smoo[:9,:10].T, 'r.', alpha=0.3)
    bx2.plot(bins[:10], np.nanmean(PC_smoo[9:,:10],0), label='PT', c='b')
    bx2.plot(bins[:10]+0.1, PC_smoo[9:,:10].T, 'b.', alpha=0.3)
    bx2.legend(loc=4)
    bx1.set_ylim([0.3,1.3])
    bx2.set_ylim([0.1,0.45])
    bx2.set_ylabel('PC')
    bx2.set_xlabel('Time (min)')
    fig2.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
    
    fig3 = plt.figure(figsize=(8,8))
    cx1 = fig3.add_subplot(2, 2, 1)
    cx2 = fig3.add_subplot(2, 2, 2)
    cx3 = fig3.add_subplot(2, 2, 3)
    cx4 = fig3.add_subplot(2, 2, 4)
    for aa, animal in enumerate(animals):
        if animal[:2]=='IT':
            cx1.plot(days, hpm_ses[aa,:], 'r', linewidth=0.5, alpha=0.5)
            cx3.plot(days, PC_ses[aa,:], 'r', linewidth=0.5, alpha=0.5)
        else:
            cx2.plot(days, hpm_ses[aa,:], 'b', linewidth=0.5, alpha=0.5)
            cx4.plot(days, PC_ses[aa,:], 'b', linewidth=0.5, alpha=0.5)
    cx1.plot(days, np.nanmean(hpm_ses[:9,:],0),'r', linewidth=2)
    cx2.plot(days, np.nanmean(hpm_ses[9:,:],0),'b', linewidth=2)
    cx3.plot(days, np.nanmean(PC_ses[:9,:],0),'r', linewidth=2)
    cx4.plot(days, np.nanmean(PC_ses[9:,:],0),'b', linewidth=2)
    cx1.set_ylim([0,4])
    cx2.set_ylim([0,4])
    cx3.set_ylim([0,1])
    cx4.set_ylim([0,1])
    cx1.set_ylabel('hpm')
    cx2.set_ylabel('hpm')
    cx3.set_ylabel('PC')
    cx4.set_ylabel('PC')
    cx3.set_xlabel('IT')
    cx4.set_xlabel('PT')
    fig3.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
    
    fig4 = plt.figure(figsize=(8,4))
    dx1 = fig4.add_subplot(1, 2, 1)
    dx2 = fig4.add_subplot(1, 2, 2)
    dx1.errorbar(days+0.1, np.nanmean(hpm_ses[:9,:],0), label='IT', yerr=pd.DataFrame(hpm_ses[:9,:]).sem(0), c='r')
    dx1.errorbar(days, np.nanmean(hpm_ses[9:,:],0), label='PT', yerr=pd.DataFrame(hpm_ses[9:,:]).sem(0), c='b')
    dx1.legend(loc=2)
    dx1.set_ylabel('HPM')
    dx1.set_xlabel('Time (min)')
    dx2.errorbar(days+0.1, np.nanmean(PC_ses[:9,:],0), label='IT', yerr=pd.DataFrame(PC_ses[:9,:]).sem(0), c='r')
    dx2.errorbar(days, np.nanmean(PC_ses[9:,:],0), label='PT', yerr=pd.DataFrame(PC_ses[9:,:]).sem(0), c='b')
    dx2.legend(loc=2)
#     dx1.set_ylim([0.4,1.1])
#     dx2.set_ylim([0.25,0.45])
    dx2.set_ylabel('PC')
    dx2.set_xlabel('Time (min)')
    fig4.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
    
    ## relative percentage increase of hpm and pc
    # the whole session
    fig5 = plt.figure(figsize=(8,8))
    ex1 = fig5.add_subplot(2, 2, 1)
    ex2 = fig5.add_subplot(2, 2, 2)
    ex3 = fig5.add_subplot(2, 2, 3)
    ex4 = fig5.add_subplot(2, 2, 4)
    for aa, animal in enumerate(animals):
        if animal[:2]=='IT':
            ex1.plot(bins-1, hpm_g[aa,:50], 'r.', alpha=0.1)
            ex3.plot(bins-1, PC_g[aa,:50], 'r.', alpha=0.1)
        else:
            ex2.plot(bins-1, hpm_g[aa,:50], 'b.', alpha=0.1)
            ex4.plot(bins-1, PC_g[aa,:50], 'b.', alpha=0.1)
#     ex1.plot(bins, np.nanmean(hpm_g[:9,:50],0),'r', linewidth=2)
#     ex2.plot(bins, np.nanmean(hpm_g[9:,:50],0),'b', linewidth=2)
#     ex3.plot(bins, np.nanmean(PC_g[:9,:50],0),'r', linewidth=2)
#     ex4.plot(bins, np.nanmean(PC_g[9:,:50],0),'b', linewidth=2)
    ex1.plot(bins, uc.sliding_mean(np.nanmean(hpm_g[:9,1:51],0),2),'r', linewidth=2)
    ex2.plot(bins, uc.sliding_mean(np.nanmean(hpm_g[9:,1:51],0),2),'b', linewidth=2)
    ex3.plot(bins, uc.sliding_mean(np.nanmean(PC_g[:9,1:51],0),2),'r', linewidth=2)
    ex4.plot(bins, uc.sliding_mean(np.nanmean(PC_g[9:,1:51],0),2),'b', linewidth=2)
    ex1.set_ylim([-60,160])
    ex2.set_ylim([-60,160])
    ex3.set_ylim([-100,150])
    ex4.set_ylim([-100,150])
    ex1.set_ylabel('hpm (% increase)')
    ex2.set_ylabel('hpm (% increase)')
    ex3.set_ylabel('PC (% increase)')
    ex4.set_ylabel('PC (% increase)')
    ex3.set_xlabel('IT')
    ex4.set_xlabel('PT')
    fig5.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
    
    fig5 = plt.figure(figsize=(8,4))
    ex1 = fig5.add_subplot(1, 2, 1)
    ex2 = fig5.add_subplot(1, 2, 2)

#     ex1.errorbar(bins[0:10]-0.1, uc.sliding_mean(np.nanmean(hpm_g[:9,1:11],0),2), yerr= pd.DataFrame(hpm_g[:9,1:11]).sem(0),c='r', linewidth=2)
#     ex1.errorbar(bins[0:10]+0.1, uc.sliding_mean(np.nanmean(hpm_g[9:,1:11],0),2), yerr= pd.DataFrame(hpm_g[9:,1:11]).sem(0),c='b', linewidth=2)

#     ex1.plot(bins[0:10], uc.sliding_mean(np.nanmean(hpm_g[:9,1:11],0),2),'r', linewidth=2)
#     ex1.plot(bins[0:10], uc.sliding_mean(np.nanmean(hpm_g[9:,1:11],0),2),'b', linewidth=2)
#     ex2.plot(bins[0:10], uc.sliding_mean(np.nanmean(PC_g[:9,1:11],0),2),'r', linewidth=2)
#     ex2.plot(bins[0:10], uc.sliding_mean(np.nanmean(PC_g[9:,1:11],0),2),'b', linewidth=2)

    ex1.plot(bins[0:10], np.nanmean(hpm_gsmoo[:9,1:11],0),'r', linewidth=2)
    ex1.plot(bins[0:10], np.nanmean(hpm_gsmoo[9:,1:11],0),'b', linewidth=2)
    ex2.plot(bins[0:10], np.nanmean(PC_gsmoo[:9,1:11],0),'r', linewidth=2)
    ex2.plot(bins[0:10], np.nanmean(PC_gsmoo[9:,1:11],0),'b', linewidth=2)
    ex1.plot(bins[0:11]-1.1, hpm_gsmoo[:9,0:11].T, 'r.', alpha=0.3)
    ex1.plot(bins[0:11]-0.9, hpm_gsmoo[9:,0:11].T, 'b.', alpha=0.3)
    ex2.plot(bins[0:11]-1.1, PC_gsmoo[:9,0:11].T, 'r.', alpha=0.3)
    ex2.plot(bins[0:11]-0.9, PC_gsmoo[9:,0:11].T, 'b.', alpha=0.3)
#     ex1.set_ylim([0,75])
#     ex2.set_ylim([0,45])
    ex1.set_ylabel('hpm gain')
    ex2.set_ylabel('PC gain')
    ex1.set_xlabel('IT')
    ex2.set_xlabel('PT')
    fig5.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
        
    fig6 = plt.figure(figsize=(8,4))
    fx1 = fig6.add_subplot(1, 2, 1)
    fx2 = fig6.add_subplot(1, 2, 2)
    fx1.bar([0,1], [np.nanmean(hpm_ses[:9,:]), np.nanmean(hpm_ses[9:,:])])
    fx1.errorbar([0,1], [np.nanmean(hpm_ses[:9,:]), np.nanmean(hpm_ses[9:,:])], \
                       yerr=[pd.DataFrame(np.nanmean(hpm_ses[:9,:],1)).sem(0)[0], \
                             pd.DataFrame(np.nanmean(hpm_ses[9:,:],1)).sem(0)[0]], c='k')
    _, p_value = stats.ttest_ind(np.nanmean(hpm_ses[:9,:],1), np.nanmean(hpm_ses[9:,:],1))
    p = uc.calc_pvalue(p_value)
    fx1.text(0.45, 0.8, p)
    fx1.set_ylabel('HPM')
    fx1.set_xticks([0,1])
    fx1.set_xticklabels(['IT', 'PT'])
    fx1.set_ylim([0,1])

    fx2.bar([0,1], [np.nanmean(PC_ses[:9,:]), np.nanmean(PC_ses[9:,:])])
    fx2.errorbar([0,1], [np.nanmean(PC_ses[:9,:]), np.nanmean(PC_ses[9:,:])], \
                       yerr=[pd.DataFrame(np.nanmean(PC_ses[:9,:],1)).sem(0)[0], \
                             pd.DataFrame(np.nanmean(PC_ses[9:,:],1)).sem(0)[0]], c='k')
    _, p_value = stats.ttest_ind(np.nanmean(PC_ses[:9,:],1), np.nanmean(PC_ses[9:,:],1))
    p = uc.calc_pvalue(p_value)
    fx2.text(0.45, 0.4, p)
    fx2.set_ylabel('PC')
    fx2.set_xticks([0,1])
    fx2.set_xticklabels(['IT', 'PT'])
    fx2.set_ylim([0,0.45])
    fig6.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
    
    fig7 = plt.figure(figsize=(8,4))
    
    

