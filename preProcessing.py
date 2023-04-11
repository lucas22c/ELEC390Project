import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from scipy.stats import skew
import joblib


def preprocessing(windows, window_size):
    filtered = np.zeros((windows.shape[0], windows.shape[1] - window_size + 1, windows.shape[2]))
    for i in range(windows.shape[0]):
        x_frame = pd.DataFrame(windows[i, :, 0])
        y_frame = pd.DataFrame(windows[i, :, 1])
        z_frame = pd.DataFrame(windows[i, :, 2])
        total = windows[i, :, 3]

        # apply MA

        x_ma = x_frame.rolling(window_size).mean().values.ravel()
        y_ma = y_frame.rolling(window_size).mean().values.ravel()
        z_ma = z_frame.rolling(window_size).mean().values.ravel()

        # discard NaN

        x_ma = x_ma[window_size - 1:]
        y_ma = y_ma[window_size - 1:]
        z_ma = z_ma[window_size - 1:]
        total_ma = total[window_size - 1:]

        filtered[i, :, 0] = x_ma
        filtered[i, :, 1] = y_ma
        filtered[i, :, 2] = z_ma
        filtered[i, :, 3] = total_ma

    feature_data = train_feature_extract(filtered)

    x_frame = pd.DataFrame(feature_data[i, :, 0])
    y_frame = pd.DataFrame(feature_data[i, :, 1])
    z_frame = pd.DataFrame(feature_data[i, :, 2])
    total_frame = pd.DataFrame(feature_data[i, :, 3])

    for frame in [x_frame, y_frame, z_frame, total_frame]:
        for i in range(frame.shape[0]):
            column =


def train_feature_extract(windows):
    train_features = np.zeros((windows.shape[0], 10, 4))
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            window_data = windows[j, :, i]

            max_val = np.max(window_data)
            min_val = np.min(window_data)
            range_val = max_val - min_val
            mean_val = np.mean(window_data)
            median_val = np.median(window_data)
            var_val = np.var(window_data)
            skew_val = skew(window_data)
            rms_val = np.sqrt(np.mean(window_data ** 2))
            kurt_val = np.mean((window_data - np.mean(window_data)) ** 4) / (np.var(window_data) ** 2)
            std_val = np.std(window_data)

            train_features[j, :, i] = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val,
                                       rms_val, kurt_val, std_val)

    return train_features


def test_feature_extract(windows):

    # Create an empty array to hold the feature vectors

    test_features = np.zeros((windows.shape[0], 10, 4))

    # Iterate over each time window and extract the features
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            # Extract the data from the window
            window_data = windows[j, :, i]

            # Compute the features
            max_val = np.max(window_data)
            min_val = np.min(window_data)
            range_val = max_val - min_val
            mean_val = np.mean(window_data)
            median_val = np.median(window_data)
            var_val = np.var(window_data)
            skew_val = skew(window_data)
            rms_val = np.sqrt(np.mean(window_data ** 2))
            kurt_val = np.mean((window_data - np.mean(window_data)) ** 4) / (np.var(window_data) ** 2)
            std_val = np.std(window_data)

            # Store the features in the features array
            test_features[j, :, i] = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val,
                                 kurt_val, std_val)

    x_feature = test_features[:, :, 0]
    y_feature = test_features[:, :, 1]
    z_feature = test_features[:, :, 2]
    total_feature = test_features[:, :, 3]

    # Concatenate the feature arrays
    all_features = np.concatenate((x_feature, y_feature, z_feature, total_feature), axis=0)

    # Create a column of labels
    labels = np.concatenate((np.ones((x_feature.shape[0], 1)),
                             2 * np.ones((y_feature.shape[0], 1)),
                             3 * np.ones((z_feature.shape[0], 1)),
                             4 * np.ones((total_feature.shape[0], 1))), axis=0)

    # Add the labels column to the feature array
    all_features = np.hstack((all_features, labels))

    return all_features
