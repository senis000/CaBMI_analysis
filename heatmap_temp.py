import pandas as pd
import numpy as np;
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
from utils_bursting import df_cv_validate

file = "/Volumes/ALBERTSHD/BMI/mats/df_window_IT3_IT4_IT5_IT6_PT6_PT7_PT9_PT12_theta_cwt_std_t2_windowNone.csv"
dfw = pd.read_csv(file)
dfw_valid = df_cv_validate(dfw, 0)


def plot_heatmap(df, groups, data, cblabel, title):
    x0, y0 = df[groups[0]].unique(), df[groups[1]].unique()
    X, Y = np.meshgrid(x0, y0)
    Z = df.groupby(groups).mean()[data].values.reshape(X.shape, order='F')
    plt.imshow(Z, cmap=cm.coolwarm)
    plt.ylabel(groups[1])
    plt.xlabel(groups[0])
    plt.yticks(range(len(y0)), y0)
    plt.xticks(range(len(x0)), x0)
    plt.colorbar(label=cblabel)
    plt.title(title)


def plot_3D(df, groups, data, title):
    x0, y0 = df[groups[0]].unique(), df[groups[1]].unique()
    X, Y = np.meshgrid(x0, y0)
    Z = df.groupby(groups).mean()[data].values.reshape(X.shape, order='F')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(groups[0])
    ax.set_ylabel(groups[1])
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.title(title)


m = 'cv_ub'
for rt in ('D', 'IR', 'IG', 'E', 'E1', 'E2'):
    df_rt = dfw_valid[(dfw_valid.roi_type == rt)]
    IT_sub =  df_rt[df_rt.group == 'IT']
    PT_sub = df_rt[df_rt.group == 'PT']
    plot_heatmap(IT_sub, ['session', 'window'], m, 'average '+m, m+" for IT animals evolution")
