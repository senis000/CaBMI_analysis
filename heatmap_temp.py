import pandas as pd
import numpy as np;
import os
import seaborn as sns
import matplotlib.pyplot as plt
from utils_bursting import df_cv_validate

file = "/Volumes/ALBERTSHD/BMI/mats/df_window_IT3_IT4_IT5_IT6_PT6_PT7_PT9_PT12_theta_cwt_std_t2_windowNone.csv"
dfw = pd.DataFrame.from_csv(file)
dfw_valid = df_cv_validate(dfw, 0)





for rt in ('D', 'IR', 'IG', 'E', 'E1', 'E2'):
    IT_sub = dfw_valid
    PT_sub = dfw_valid

l = kdf.groupby(["window", 'session', 'group'])
l.get_group[[0, 0, 'IT']]
l.get_group([0, 0, 'IT'])
l.get_group((0, 0, 'IT'))
l.get_group(0)

l = kdf.groupby(["window", 'session', 'group']).mean()
l = kdf.groupby(["window", 'session', 'group'])
l
itl = kdf[kdf.group=='IT'].groupby(['window', 'session'])
ptl = kdf[kdf.group=='PT'].groupby(['window', 'session'])
ptl
itl.mean()
kdf = kdf[kdf.session<=17]
itl = kdf[kdf.group=='IT'].groupby(['window', 'session'])
ptl = kdf[kdf.group=='PT'].groupby(['window', 'session'])
itl.mean()
X = np.arange(5)
Y = np.arange(1, 18)
X, Y = np.meshgrid(X, Y)
cv_ub = itl.mean().cv_ub.values
cv_ub
X.shape
X
itl.mean()
cv_ub
cv_ub.reshape((17, 5))
cv_ub.reshape((5, 17))
Z = cv_ub.reshape((5, 17)).T
Z.shape
X.shape
X[3][3]
Y[3][3]
Z[3][3]
itl.mean().iloc[15+3]
Z[0][10]
Z[10][0]
window[10][0]
X[10][0]
Y[10][0]
X.shape
Y.shape
Z.shape
fig = plt.figure()
ax = fig.gca(projection='3d')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
ax.set_xlabel('window');ax.set_ylabel("session")
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('window');ax.set_ylabel("session")
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
Z
plt.imshow(Z); plt.xlabel('window'); plt.ylabel("session");plt.show()
plt.imshow(Z); plt.xlabel('window'); plt.ylabel("session");plt.colorbar();plt.show()
plt.imshow(Z, cmap=cm.coolwarm); plt.xlabel('window'); plt.ylabel("session");plt.colorbar();plt.show()
plt.imshow(Z, cmap=cm.coolwarm); plt.xlabel('window'); plt.ylabel("session");plt.colorbar(label="average unbiased cv");plt.show()
plt.imshow(Z, cmap=cm.coolwarm); plt.xlabel('window'); plt.ylabel("session");plt.xticks(np.arange(5), np.arange(5));plt.colorbar(label="average unbiased cv");plt.show()
plt.imshow(Z, cmap=cm.coolwarm); plt.xlabel('window'); plt.ylabel("session");plt.xticks(np.arange(5), np.arange(5));plt.yticks(np.arange(17), np.arange(1, 18));plt.colorbar(label="average unbiased cv");plt.show()
plt.imshow(Z.T, cmap=cm.coolwarm); plt.ylabel('window'); plt.xlabel("session");plt.yticks(np.arange(5), np.arange(5));plt.xticks(np.arange(17), np.arange(1, 18));plt.colorbar(label="average unbiased cv");plt.show()
plt.imshow(Z.T, cmap=cm.coolwarm); plt.ylabel('window'); plt.xlabel("session");plt.yticks(np.arange(5), np.arange(5));plt.xticks(np.arange(17), np.arange(1, 18));plt.colorbar(label="average unbiased cv");plt.title("unbiased CV for IT animals evolution");plt.show()
%hist
