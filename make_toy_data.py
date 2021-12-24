import os
import h5py
import numpy as np
import pandas as pd


def create_synthetic_data(folder_main, N=60, noise_std=0.1, independent=True):
    '''
    User-defined parameters
    N: Number of sessions (we have 286 sessions and 43 features)
    noise_std: Standard deviation of outcome noise
    '''

    #
    low, high = (0.25, 0.75)  # The different scales of coefficients

    to_save_df = os.path.join(folder_main, 'df_all.hdf5')

    # Construct X matrix by sampling each feature uniformly from [0,1]
    # Features {0,1,2} increase y; {3,4,5} decrease y; {6,7,8} are irrelevant
    N = 32
    if independent:
        X = np.zeros((N, 5))
        X[:16, 0] = 1
        X[:8, 1] = 1
        X[16:24, 1] = 1
        X[:4, 2] = 1
        X[8:12, 2] = 1
        X[16:20, 2] = 1
        X[24:28, 2] = 1
        X[::2, 3] = 1
        X[:, 0:4] -= 0.5

        y = high * X[:, 0] - high * X[:, 1] + low * X[:, 2] - low * X[:, 3]
    else:
        noise_std = 0.003
        X = np.random.random((N, 5))
        y = high * X[:, 0] - high * X[:, 1] + low * X[:, 2] - low * X[:, 3]
        y += np.random.normal(0, noise_std, size=(y.shape))

    y = 0.3*(y - np.min(y)) / np.ptp(y)

    # Save X/Y matrix to hdf5 file
    columns_basic = ['PC_fake', 'UP_high', 'Down_high', 'up_low', 'down_low', 'Not']
    df = pd.DataFrame(columns=columns_basic)
    df[columns_basic[0]] = y
    for cc, column_to_create in enumerate(columns_basic[1:]):
        df[column_to_create] = X[:, cc]

    df.to_hdf(to_save_df, key='df', mode='w')
