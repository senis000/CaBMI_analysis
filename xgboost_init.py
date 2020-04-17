
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
to_load_df = os.path.join(folder_main, 'df_all.hdf5')
df = pd.read_hdf(to_load_df)

#until new df
df = df.rename(columns={'label': 'ITPTlabel'})
df['ITPTlabel'] = pd.to_numeric(df['ITPTlabel'])
bts_n=1000
mod_n=10
mod_x=10
error_max=[0.02,0.5,0.02,0.15,0.04,0.15]
size_split_test=0.2
max_iter=20
stability_var=0.7
toplot=True