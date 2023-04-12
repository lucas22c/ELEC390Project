import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, entropy


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


with h5py.File("./project_data.h5", "r") as f:
    walk_train = pd.DataFrame(f["/Dataset/Train/Walking"])
    walk_test = pd.DataFrame(f["/Dataset/Test/Walking"])
    jump_train = pd.DataFrame(f["/Dataset/Train/Jumping"])
    jump_test = pd.DataFrame(f["/Dataset/Test/Jumping"])

    walk_train.columns = ["time", "x acceleration", "y acceleration", "z acceleration", "abs acceleration"]
    walk_test.columns = ["time", "x acceleration", "y acceleration", "z acceleration", "abs acceleration"]
    jump_train.columns = ["time", "x acceleration", "y acceleration", "z acceleration", "abs acceleration"]
    jump_test.columns = ["time", "x acceleration", "y acceleration", "z acceleration", "abs acceleration"]

    f.close()

walk_train.drop(["time"])
walk_train.fillna(walk_train.mean(), inplace=True)
walk_train.rolling(5, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(walk_train, linewidth=5)
plt.show()

# dataframes are structured as [rows: segments, cols: slot of segment, z-dim: type of acceleration]

jump_train.drop(["time"])
jump_train.fillna(walk_train.mean(), inplace=True)
jump_train.rolling(5, min_periods=1).mean()

# normalize!!
scaler = StandardScaler()
walk_train = scaler.fit_transform(walk_train)
jump_train = scaler.fit_transform(jump_train)
