import os
import h5py
import numpy as np
import pandas as pd


def create_synthetic_data(folder_main, N=60, noise_std=0.1):
    '''
    User-defined parameters
    N: Number of sessions (we have 286 sessions and 43 features)
    noise_std: Standard deviation of outcome noise
    '''

    #
    low, med, high = (0.25, 0.5, 0.75)  # The different scales of coefficients

    to_save_df = os.path.join(folder_main, 'df_all.hdf5')

    # Construct X matrix by sampling each feature uniformly from [0,1]
    # Features {0,1,2} increase y; {3,4,5} decrease y; {6,7,8} are irrelevant
    X = np.random.uniform(0, 1, size=(N, 9))

    # Generate Y matrix with some noise
    y = low * X[:, 0] + med * X[:, 1] + high * X[:, 2] - low * X[:, 3] - med * X[:, 4] - high * X[:, 5]
    y += np.random.normal(0, noise_std, size=(y.shape))
    y = .6*(y - np.min(y)) / np.ptp(y)

    # Save X/Y matrix to hdf5 file
    columns_basic = ['PC_fake', 'feature_up_low', 'feature_up_med', 'feature_up_high',
                     'feature_down_low', 'feature_down_med', 'feature_down_high',
                                                             'feature_not_1', 'feature_not_2', 'feature_not_3']
    df = pd.DataFrame(columns=columns_basic)
    df[columns_basic[0]] = y
    for cc, column_to_create in enumerate(columns_basic[1:]):
        df[column_to_create] = X[:, cc]

    df.to_hdf(to_save_df, key='df', mode='w')
