
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
# import analysis_functions as af
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
to_load_e2 = os.path.join(folder_main, 'e2_neurs.p')
to_load_df = os.path.join(folder_main, 'df_all.hdf5')
df = pd.read_hdf(to_load_df)
df_ce = pd.read_pickle(to_load_pick)
df_e2 = pd.read_pickle(to_load_e2)

#until new df
# df = df.rename(columns={'label': 'ITPTlabel'})
# df['ITPTlabel'] = pd.to_numeric(df['ITPTlabel'])
bts_n=1000
mod_n=10000
mod_x=100

size_split_test=0.2
max_iter=40
stability_var=0.5
toplot=True
classif=False
if classif:
    error_bstmax=[0.03] #[0.022,1,0.02,0.2,0.8,2.5]
    error_msemax=[0.05] #[0.035,0.8,0.03,0.3,1.4,4]
elif synthetic:
    error_bstmax = [0.021]  # [0.022,1,0.02,0.2,0.8,2.5]
    error_msemax = [0.021]
else:
    error_bstmax=[0.02]
    error_msemax=[0.03] #[0.035,0.8,0.03,0.3,1.4,4]

### xgboost
to_load_xgboost = os.path.join(folder_main, 'XGShap_model.h5py')
f = h5py.File(to_load_xgboost, 'r')
labels_to_study = f.attrs['labels_to_study']
all_shap = np.asarray(f['all_shap'])
all_y_pred = np.asarray(f['all_y_pred'])
all_mse = np.asarray(f['all_mse'])
shap_correlations = np.asarray(f['shap_correlations'])
explainer_val = np.asarray(f['explainer_val'])
all_df = np.asarray(f['all_df'])
all_shap_reshape = np.asarray(f['all_shap_reshape'])
all_df_reshape = np.asarray(f['all_df_reshape'])
shap_experiment_mean = np.asarray(f['shap_experiment_mean'])
shap_experiment_std = np.asarray(f['shap_experiment_std'])
shap_experiment_sem = np.asarray(f['shap_experiment_sem'])
spread = np.asarray(f['spread'])
bins_zscore = np.arange(-2,2,0.1)

f.close()
columns = df.columns.tolist()
columns_ler = [columns[6]]


#####
df_hpm = pd.read_csv(file_csv_hpm)
df_PC = pd.read_csv(file_csv_PC)
df['PC_gain'][np.isinf(df['PC_gain'])]=np.nan
df['HPM_gain'][np.isinf(df['HPM_gain'])]=np.nan
