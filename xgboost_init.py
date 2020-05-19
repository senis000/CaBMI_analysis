
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
import xgboost
import shap
import xgboost_analysis as xga
import analysis_functions as af
import utils_cabmi as uc
from matplotlib import interactive
from scipy import stats
from itertools import combinations
from scipy.stats.mstats import zscore
from scipy import ndimage
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

interactive(True)

folder_main = 'I:/Nuria_data/CaBMI/Layer_project'
out = 'I:/Nuria_data/CaBMI/Layer_project/learning_stats'
file_csv = os.path.join(out, 'learning_stats_summary_bin_5.csv')
file_csv_hpm = os.path.join(out, 'learning_stats_HPM_bin_1.csv')
file_csv_PC = os.path.join(out, 'learning_stats_cumuPC_bin_1.csv')

to_load_pick = os.path.join(folder_main, 'cursor_engagement.p')
to_load_df = os.path.join(folder_main, 'df_all.hdf5')
df = pd.read_hdf(to_load_df)
df_ce = pd.read_pickle(to_load_pick)

#until new df
# df = df.rename(columns={'label': 'ITPTlabel'})
# df['ITPTlabel'] = pd.to_numeric(df['ITPTlabel'])
bts_n=1000
mod_n=10
mod_x=10
error_max=[0.022,0.5,0.018,0.18,0.12,0.23]
size_split_test=0.2
max_iter=20
stability_var=0.7
toplot=False





####
df_hpm = pd.read_csv(file_csv_hpm)
df_PC = pd.read_csv(file_csv_PC)