import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, mode
from scipy.fft import fft
from sklearn.model_selection import train_test_split
window = 5

with h5py.File('./project_data.h5', 'r') as hdf:
    walk_train = hdf['Dataset/Train/Walking'][:, :, 1:] # drop time, as it is doesn't matter
    jump_train = hdf['Dataset/Train/Jumping'][:, :, 1:]
    walk_test = hdf['Dataset/Test/Walking'][:, :, 1:]
    jump_test = hdf['Dataset/Test/Jumping'][:, :, 1:]

walk_window_count = walk_train.shape[0]
jump_window_count = jump_train.shape[0]

walk_test_window_count = walk_test.shape[0]
jump_test_window_count = jump_test.shape[0]

# ---------------------------------------
# WALKING TRAIN

walking = np.zeros((walk_train.shape[0], walk_train.shape[1]-window+1, walk_train.shape[2]))
walk_features_list = np.zeros((walk_window_count, 10, 4))

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
        #print("walking shape " + str((walking[i, :, j]).shape))
        #print("walk features list shape " + str(walk_features_list[i, :, j].shape))
        #walk_features_list[i, :, j] = features(walking[i, :, j])
        walk_features_list[i, 0, j] = np.max(walking[i, :, j])
        walk_features_list[i, 1, j] = np.min(walking[i, :, j])
        walk_features_list[i, 2, j] = np.ptp(walking[i, :, j])
        walk_features_list[i, 3, j] = np.mean(walking[i, :, j])
        walk_features_list[i, 4, j] = np.median(walking[i, :, j])
        walk_features_list[i, 5, j] = np.var(walking[i, :, j])
        walk_features_list[i, 6, j] = skew(walking[i, :, j])
        walk_features_list[i, 7, j] = np.std(walking[i, :, j])
        walk_features_list[i, 8, j] = kurtosis(walking[i, :, j])
        walk_features_list[i, 9, j] = np.sqrt(np.mean(walking[i, :, j]) ** 2)

walk_features = np.concatenate((pd.DataFrame(walk_features_list[:, :, 0]),
                                pd.DataFrame(walk_features_list[:, :, 1]),
                                pd.DataFrame(walk_features_list[:, :, 2]),
                                pd.DataFrame(walk_features_list[:, :, 3])),
                               axis=0)

column_names = ["max", "min", "range", "mean", "median", "var", "skew", "std", "kurtosis", "rms"]
x_accel_array = np.array([np.array(['x_accel' for _ in range(walk_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(walk_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(walk_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(walk_window_count)], dtype=object)])

#x_accel_array = np.empty((walk_window_count, 10))
#for j in range(walk_window_count):
 #   for k in range(10):
 #       x_accel_array[j, k]

label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
   # axis=1  # stack vertically
)

#print(np.rot90(label_column).shape)

walk_features = np.hstack((walk_features, np.rot90(label_column)))
print("walk features: " + str(walk_features.shape))

# ---------------------------------------
# JUMPING TRAIN

jumping = np.zeros((jump_train.shape[0], jump_train.shape[1] - window + 1, jump_train.shape[2]))
jump_features_list = np.zeros((jump_window_count, 10, 4))

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

    jumping[i, :, 0] = x_acceleration
    jumping[i, :, 1] = y_acceleration
    jumping[i, :, 2] = z_acceleration
    jumping[i, :, 3] = abs_acceleration

    for j in range(4):
        jump_features_list[i, 0, j] = np.max(jumping[i, :, j])
        jump_features_list[i, 1, j] = np.min(jumping[i, :, j])
        jump_features_list[i, 2, j] = np.ptp(jumping[i, :, j])
        jump_features_list[i, 3, j] = np.mean(jumping[i, :, j])
        jump_features_list[i, 4, j] = np.median(jumping[i, :, j])
        jump_features_list[i, 5, j] = np.var(jumping[i, :, j])
        jump_features_list[i, 6, j] = skew(jumping[i, :, j])
        jump_features_list[i, 7, j] = np.std(jumping[i, :, j])
        jump_features_list[i, 8, j] = kurtosis(jumping[i, :, j])
        jump_features_list[i, 9, j] = np.sqrt(np.mean(jumping[i, :, j]) ** 2)

jump_features = np.concatenate((pd.DataFrame(jump_features_list[:, :, 0]),
                                pd.DataFrame(jump_features_list[:, :, 1]),
                                pd.DataFrame(jump_features_list[:, :, 2]),
                                pd.DataFrame(jump_features_list[:, :, 3])),
                               axis=0)

column_names = ["max", "min", "range", "mean", "median", "var", "skew", "std", "kurtosis", "rms"]
x_accel_array = np.array([np.array(['x_accel' for _ in range(jump_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(jump_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(jump_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(jump_window_count)], dtype=object)])

label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
)

jump_features = np.hstack((jump_features, np.rot90(label_column)))
print("jump features: " + str(jump_features.shape))

# ---------------------------------------
# WALKING TEST

walk_test_features_list = np.zeros((walk_test_window_count, 10, 4))

for i in range(walk_test_window_count):
    for j in range(4):
        #print("walking shape " + str((walking[i, :, j]).shape))
        #print("walk features list shape " + str(walk_features_list[i, :, j].shape))
        #walk_features_list[i, :, j] = features(walking[i, :, j])
        walk_test_features_list[i, 0, j] = np.max(walk_test[i, :, j])
        walk_test_features_list[i, 1, j] = np.min(walk_test[i, :, j])
        walk_test_features_list[i, 2, j] = np.ptp(walk_test[i, :, j])
        walk_test_features_list[i, 3, j] = np.mean(walk_test[i, :, j])
        walk_test_features_list[i, 4, j] = np.median(walk_test[i, :, j])
        walk_test_features_list[i, 5, j] = np.var(walk_test[i, :, j])
        walk_test_features_list[i, 6, j] = skew(walk_test[i, :, j])
        walk_test_features_list[i, 7, j] = np.std(walk_test[i, :, j])
        walk_test_features_list[i, 8, j] = kurtosis(walk_test[i, :, j])
        walk_test_features_list[i, 9, j] = np.sqrt(np.mean(walk_test[i, :, j]) ** 2)

walk_test_features = np.concatenate((pd.DataFrame(walk_test_features_list[:, :, 0]),
                                pd.DataFrame(walk_test_features_list[:, :, 1]),
                                pd.DataFrame(walk_test_features_list[:, :, 2]),
                                pd.DataFrame(walk_test_features_list[:, :, 3])),
                               axis=0)

column_names = ["max", "min", "range", "mean", "median", "var", "skew", "std", "kurtosis", "rms"]
x_accel_array = np.array([np.array(['x_accel' for _ in range(walk_test_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(walk_test_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(walk_test_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(walk_test_window_count)], dtype=object)])

#x_accel_array = np.empty((walk_window_count, 10))
#for j in range(walk_window_count):
 #   for k in range(10):
 #       x_accel_array[j, k]

label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
   # axis=1  # stack vertically
)

walk_test_features = np.hstack((walk_test_features, np.rot90(label_column)))

# ---------------------------------------
# JUMPING TEST

jump_test_features_list = np.zeros((jump_test_window_count, 10, 4))

for i in range(jump_test_window_count):
    for j in range(4):
        jump_test_features_list[i, 0, j] = np.max(jump_test[i, :, j])
        jump_test_features_list[i, 1, j] = np.min(jump_test[i, :, j])
        jump_test_features_list[i, 2, j] = np.ptp(jump_test[i, :, j])
        jump_test_features_list[i, 3, j] = np.mean(jump_test[i, :, j])
        jump_test_features_list[i, 4, j] = np.median(jump_test[i, :, j])
        jump_test_features_list[i, 5, j] = np.var(jump_test[i, :, j])
        jump_test_features_list[i, 6, j] = skew(jump_test[i, :, j])
        jump_test_features_list[i, 7, j] = np.std(jump_test[i, :, j])
        jump_test_features_list[i, 8, j] = kurtosis(jump_test[i, :, j])
        jump_test_features_list[i, 9, j] = np.sqrt(np.mean(jump_test[i, :, j]) ** 2)

jump_test_features = np.concatenate((pd.DataFrame(jump_test_features_list[:, :, 0]),
                                     pd.DataFrame(jump_test_features_list[:, :, 1]),
                                     pd.DataFrame(jump_test_features_list[:, :, 2]),
                                     pd.DataFrame(jump_test_features_list[:, :, 3])),
                                     axis=0)

column_names = ["max", "min", "range", "mean", "median", "var", "skew", "std", "kurtosis", "rms"]
x_accel_array = np.array([np.array(['x_accel' for _ in range(jump_test_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(jump_test_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(jump_test_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(jump_test_window_count)], dtype=object)])

#x_accel_array = np.empty((walk_window_count, 10))
#for j in range(walk_window_count):
 #   for k in range(10):
 #       x_accel_array[j, k]

label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
   # axis=1  # stack vertically
)

jump_test_features = np.hstack((jump_test_features, np.rot90(label_column)))

# FEATURES DONE!!!

column_names = np.array(["max", "min", "range", "mean", "median", "var", "skew", "std", "kurtosis", "rms", "measurement", "activity"])

train_features = np.concatenate((walk_features, jump_features))
train_labels = np.concatenate((np.zeros((walk_features.shape[0], 1)), np.ones((jump_features.shape[0], 1))))
training = pd.DataFrame(np.hstack((train_features, train_labels)), columns=column_names)

test_features = np.concatenate((walk_test_features, jump_test_features))
test_labels = np.concatenate((np.zeros((walk_test_features.shape[0], 1)), np.ones((jump_test_features.shape[0], 1))))
testing = pd.DataFrame(np.hstack((test_features, test_labels)), columns=column_names)

print("train shape " + str(training.shape))
print(training.iloc[:, -1])

