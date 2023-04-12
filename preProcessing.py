import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import train_test_split
window = 5

def features(segment):

    # creates dictionary of features for a segment
    feature = {"maximum": np.max(segment),
               "minimum": np.min(segment),
               "range": np.max(segment) - np.min(segment),
               "mean": np.mean(segment),
               "median": np.median(segment),
               "variance": np.var(segment),
               "skewness": skew(segment),
               "std": np.std(segment),
               "kurtosis": kurtosis(segment),
               "entropy": entropy(segment)}
    return feature


with h5py.File('./project_data.h5', 'r') as hdf:
    walk_train = hdf['Dataset/Train/Walking'][:, :, 1:] # drop time, as it is doesn't matter
    jump_train = hdf['Dataset/Train/Jumping'][:, :, 1:]
    walk_test = hdf['Dataset/Test/Walking'][:, :, 1:]
    jump_test = hdf['Dataset/Test/Jumping'][:, :, 1:]

walk_window_count = walk_train.shape[0]
jump_window_count = jump_train.shape[0]

walking = np.zeros((walk_train.shape[0], walk_train.shape[1]-window+1, walk_train.shape[2]))
walk_features_list = [[dict() for x in range(4)] for y in range(walk_window_count)]
# walk_features_list = [] # np.zeros((walk_window_count, 4))
# for i in range(walk_window_count):
 #   walk_features_list[i] = [dict(), 0]

for i in range(walk_window_count):
    x_acceleration = pd.DataFrame(walk_train[i, :, 0])
    x_acceleration = x_acceleration.rolling(window).mean().values.ravel()
    x_acceleration = x_acceleration[window - 1:]

    y_acceleration = pd.DataFrame(walk_train[i, :, 1])
    y_acceleration = y_acceleration.rolling(window).mean().values.ravel()
    y_acceleration = y_acceleration[window - 1:]

    z_acceleration = pd.DataFrame(walk_train[i, :, 2])
    z_acceleration = z_acceleration.rolling(window).mean().values.ravel()
    z_acceleration = z_acceleration[window - 1:]

    abs_acceleration = pd.DataFrame(walk_train[i, :, 3])
    abs_acceleration = abs_acceleration.rolling(window).mean().values.ravel()
    abs_acceleration = abs_acceleration[window - 1:]

    walking[i, :, 0] = x_acceleration
    walking[i, :, 1] = y_acceleration
    walking[i, :, 2] = z_acceleration
    walking[i, :, 3] = abs_acceleration

    for j in range(4):
        walk_features_list[i][j] = features(walking[i, :, j])

jumping = np.zeros((jump_train.shape[0], jump_train.shape[1] - window + 1, jump_train.shape[2]))
jump_features_list = [[dict() for x in range(4)] for y in range(jump_window_count)]

for i in range(jump_window_count):
    x_acceleration = pd.DataFrame(jump_train[i, :, 0])
    x_acceleration = x_acceleration.rolling(window).mean().values.ravel()
    x_acceleration = x_acceleration[window - 1:]

    y_acceleration = pd.DataFrame(jump_train[i, :, 1])
    y_acceleration = y_acceleration.rolling(window).mean().values.ravel()
    y_acceleration = y_acceleration[window - 1:]

    z_acceleration = pd.DataFrame(jump_train[i, :, 2])
    z_acceleration = z_acceleration.rolling(window).mean().values.ravel()
    z_acceleration = z_acceleration[window - 1:]

    abs_acceleration = pd.DataFrame(jump_train[i, :, 3])
    abs_acceleration = abs_acceleration.rolling(window).mean().values.ravel()
    abs_acceleration = abs_acceleration[window - 1:]

    walking[i, :, 0] = x_acceleration
    walking[i, :, 1] = y_acceleration
    walking[i, :, 2] = z_acceleration
    walking[i, :, 3] = abs_acceleration

    for j in range(4):
        jump_features_list[i][j] = features(jumping[i, :, j])

# normalize!!
scaler = StandardScaler()
walking_x = scaler.fit_transform(walking[:, :, 0])
walking_y = scaler.fit_transform(walking[:, :, 1])
walking_z = scaler.fit_transform(walking[:, :, 2])
walking_abs = scaler.fit_transform(walking[:, :, 3])

jumping_x = scaler.fit_transform(jumping[:, :, 0])
jumping_y = scaler.fit_transform(jumping[:, :, 1])
jumping_z = scaler.fit_transform(jumping[:, :, 2])
jumping_abs = scaler.fit_transform(jumping[:, :, 3])
