import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

# for preprocessing and feature extraction
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

# for classifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LearningCurveDisplay, learning_curve
import joblib

# allows for easy testing by changing window size here
window = 5


with h5py.File('../project_data.h5', 'r') as hdf:
    walk_train = hdf['Dataset/Train/Walking'][:, :, 1:]  # drop time, as it is doesn't matter
    jump_train = hdf['Dataset/Train/Jumping'][:, :, 1:]
    walk_test = hdf['Dataset/Test/Walking'][:, :, 1:]
    jump_test = hdf['Dataset/Test/Jumping'][:, :, 1:]

walk_window_count = walk_train.shape[0]
jump_window_count = jump_train.shape[0]
walk_test_window_count = walk_test.shape[0]
jump_test_window_count = jump_test.shape[0]

# ---------------------------------------
# WALKING TRAIN

walking = np.zeros((walk_window_count, walk_train.shape[1] - window + 1, walk_train.shape[2]))
walk_features_list = np.zeros((walk_window_count, 10, 4))  # stores features for classifier

fig, ax = plt.subplots()
ax.plot(walk_train[1, :, 3], color = 'green', label = 'Walking Before')

for i in range(walk_window_count):
    # x acceleration moving average
    x_acceleration = pd.DataFrame(walk_train[i, :, 0])
    x_acceleration = x_acceleration.rolling(window).mean().values.ravel()
    x_acceleration = x_acceleration[window - 1:]

    # y acceleration moving average
    y_acceleration = pd.DataFrame(walk_train[i, :, 1])
    y_acceleration = y_acceleration.rolling(window).mean().values.ravel()
    y_acceleration = y_acceleration[window - 1:]

    # z acceleration moving average
    z_acceleration = pd.DataFrame(walk_train[i, :, 2])
    z_acceleration = z_acceleration.rolling(window).mean().values.ravel()
    z_acceleration = z_acceleration[window - 1:]

    # absolute acceleration moving average
    abs_acceleration = pd.DataFrame(walk_train[i, :, 3])
    abs_acceleration = abs_acceleration.rolling(window).mean().values.ravel()
    abs_acceleration = abs_acceleration[window - 1:]

    # complete acceleration array
    walking[i, :, 0] = x_acceleration
    walking[i, :, 1] = y_acceleration
    walking[i, :, 2] = z_acceleration
    walking[i, :, 3] = abs_acceleration

    # compute features
    for j in range(4):
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

# plot comparison
ax.plot(walking[1, :, 3], color = 'yellow', label = 'Walking After')
ax.set_title('Acceleration Magnitude vs Time For One Segment')
ax.set_ylabel('Acceleration Magnitude (in m/s^2)')
ax.set_xlabel('Time (in 1/100 s)')
ax.legend()
plt.show()

# concatenate all features
walk_features = np.concatenate((pd.DataFrame(walk_features_list[:, :, 0]),
                                pd.DataFrame(walk_features_list[:, :, 1]),
                                pd.DataFrame(walk_features_list[:, :, 2]),
                                pd.DataFrame(walk_features_list[:, :, 3])),
                               axis=0)

# labels for acceleration type for readability
x_accel_array = np.array([np.array(['x_accel' for _ in range(walk_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(walk_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(walk_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(walk_window_count)], dtype=object)])
label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
)

# all features
# we rotate because I couldn't figure out how to make the data properly aligned otherwise
walk_features = np.hstack((walk_features, np.rot90(label_column)))

# ---------------------------------------
# JUMPING TRAIN

# SAME PROCESS AS WALKING, SO I'LL LEAVE OUT COMMENTS

# jumping will store the moving average result
jumping = np.zeros((jump_window_count, jump_train.shape[1] - window + 1, jump_train.shape[2]))  # must subtract window+1
jump_features_list = np.zeros((jump_window_count, 10, 4))  # stores features for classifier

for i in range(jump_window_count):
    # x acceleration moving average
    x_acceleration = pd.DataFrame(jump_train[i, :, 0])
    x_acceleration = x_acceleration.rolling(window).mean().values.ravel()
    x_acceleration = x_acceleration[window - 1:]

    # y acceleration moving average
    y_acceleration = pd.DataFrame(jump_train[i, :, 1])
    y_acceleration = y_acceleration.rolling(window).mean().values.ravel()
    y_acceleration = y_acceleration[window - 1:]

    # z acceleration moving average
    z_acceleration = pd.DataFrame(jump_train[i, :, 2])
    z_acceleration = z_acceleration.rolling(window).mean().values.ravel()
    z_acceleration = z_acceleration[window - 1:]

    # absolute acceleration moving average
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

x_accel_array = np.array([np.array(['x_accel' for _ in range(jump_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(jump_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(jump_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(jump_window_count)], dtype=object)])
label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
)

jump_features = np.hstack((jump_features, np.rot90(label_column)))

# ---------------------------------------
# WALKING TEST

walk_test_features_list = np.zeros((walk_test_window_count, 10, 4))  # stores features for classifier

# don't bother doing moving average, we want to compare rough data to see how well it works without normalization
for i in range(walk_test_window_count):
    # compute features for all windows
    for j in range(4):
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

x_accel_array = np.array([np.array(['x_accel' for _ in range(walk_test_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(walk_test_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(walk_test_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(walk_test_window_count)], dtype=object)])
label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
)

walk_test_features = np.hstack((walk_test_features, np.rot90(label_column)))

# ---------------------------------------
# JUMPING TEST

# SAME PROCESS AS WALKING, SO NO COMMENTS AGAIN

jump_test_features_list = np.zeros((jump_test_window_count, 10, 4))  # stores features for classifier

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

x_accel_array = np.array([np.array(['x_accel' for _ in range(jump_test_window_count)], dtype=object)])
y_accel_array = np.array([np.array(['y_accel' for _ in range(jump_test_window_count)], dtype=object)])
z_accel_array = np.array([np.array(['z_accel' for _ in range(jump_test_window_count)], dtype=object)])
abs_accel_array = np.array([np.array(['abs_accel' for _ in range(jump_test_window_count)], dtype=object)])
label_column = np.hstack(
    (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
)

jump_test_features = np.hstack((jump_test_features, np.rot90(label_column)))

# -------------------------------------------
# columns for readability!!
column_names = np.array(["max", "min", "range", "mean", "median", "var", "skew", "std", "kurtosis", "rms", "type", "walk/jump"])

# features, labels, and concatenations
train_features = np.concatenate((walk_features, jump_features))
test_features = np.concatenate((walk_test_features, jump_test_features))
train_labels = np.concatenate((np.zeros((walk_features.shape[0], 1)), np.ones((jump_features.shape[0], 1))))
test_labels = np.concatenate((np.zeros((walk_test_features.shape[0], 1)), np.ones((jump_test_features.shape[0], 1))))
training = pd.DataFrame(np.hstack((train_features, train_labels)), columns=column_names)
testing = pd.DataFrame(np.hstack((test_features, test_labels)), columns=column_names)

# FEATURES DONE!!!
# ----------------------------------------------

# normalize!!
scaler = StandardScaler()

# classifier
X_train = training.iloc[:, 0:-2]
y_train = training.iloc[:, -1]
y_train = y_train.astype('int')

X_test = testing.iloc[:, 0:-2]
y_test = testing.iloc[:, -1]
y_test = y_test.astype('int')

# effectively identical to lab 6 below

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(scaler, l_reg)

clf.fit(X_train, y_train)

LearningCurveDisplay.from_estimator(clf, X_train, y_train)
plt.ylim(0, 1)
plt.show()

joblib.dump(clf, 'classifier.joblib')

predicted = clf.predict(X_test)
predicted_proba = clf.predict_proba(X_test)

print("y_pred is:", predicted)
print("y_predicted_proba is:", predicted_proba)

acc = accuracy_score(y_test, predicted)
print("accuracy is:", acc)

recall = recall_score(y_test, predicted)
print("recall is:", recall)

cm = confusion_matrix(y_test, predicted)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

fpr, tpr, _ = roc_curve(y_test, predicted_proba[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

auc = roc_auc_score(y_test, predicted_proba[:, 1])
print("the AUC is:", auc)

TP = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
F1 = (2*TP)/(2*TP+FP+FN)
print('F1 Score: ', F1)