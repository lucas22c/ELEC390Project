import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from scipy.stats import skew, kurtosis, mode

# Create GUI Frame
window = tk.Tk()
window.title("CSV File Reader")
# Define a function to read the CSV file
def readCSV():
    # Open a file dialog to select the CSV file
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    # Checks to see if the file path exists
    if file_path == "":
        return

    # Train Model
    model = joblib.load('classifier.joblib')

    # Read CSV
    fileData = pd.read_csv(file_path)

    cols_to_keep = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
                    'Linear Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']

    # Drop all other columns
    fileData = fileData[cols_to_keep]

    # Split the data into 5 second intervals
    split_fileData = [fileData[(i * 100): (i * 100 + 500)] for i in range((len(fileData) - 500) // 100 + 1)]

    split_fileData = np.array(split_fileData)

    file_window_count = split_fileData.shape[0]

    # Extract the features from the inputted data
    fileData_features_list = np.zeros((file_window_count, 10, 4))

    for i in range(file_window_count):
        for j in range(4):
            fileData_features_list[i, 0, j] = np.max(split_fileData[i, :, j])
            fileData_features_list[i, 1, j] = np.min(split_fileData[i, :, j])
            fileData_features_list[i, 2, j] = np.ptp(split_fileData[i, :, j])
            fileData_features_list[i, 3, j] = np.mean(split_fileData[i, :, j])
            fileData_features_list[i, 4, j] = np.median(split_fileData[i, :, j])
            fileData_features_list[i, 5, j] = np.var(split_fileData[i, :, j])
            fileData_features_list[i, 6, j] = skew(split_fileData[i, :, j])
            fileData_features_list[i, 7, j] = np.std(split_fileData[i, :, j])
            fileData_features_list[i, 8, j] = kurtosis(split_fileData[i, :, j])
            fileData_features_list[i, 9, j] = np.sqrt(np.mean(split_fileData[i, :, j]) ** 2)

    walk_test_features = np.concatenate((pd.DataFrame(fileData_features_list[:, :, 0]),
                                         pd.DataFrame(fileData_features_list[:, :, 1]),
                                         pd.DataFrame(fileData_features_list[:, :, 2]),
                                         pd.DataFrame(fileData_features_list[:, :, 3])),
                                        axis=0)

    column_names = ["max", "min", "range", "mean", "median", "var", "skew", "std", "kurtosis", "rms"]
    x_accel_array = np.array([np.array(['x_accel' for _ in range(file_window_count)], dtype=object)])
    y_accel_array = np.array([np.array(['y_accel' for _ in range(file_window_count)], dtype=object)])
    z_accel_array = np.array([np.array(['z_accel' for _ in range(file_window_count)], dtype=object)])
    abs_accel_array = np.array([np.array(['abs_accel' for _ in range(file_window_count)], dtype=object)])

    label_column = np.hstack(
        (x_accel_array, y_accel_array, z_accel_array, abs_accel_array),
    )

    fileData_features = np.hstack((walk_test_features, np.rot90(label_column)))

    X_data = fileData_features[:, 0:-1]

    # Predict Input
    prediction_results = model.predict(X_data)

    # Graph the data
    walking_pred = np.count_nonzero(prediction_results)
    jumping_pred = len(prediction_results) - walking_pred

    prediction = ('Walking', 'Jumping')
    values = (walking_pred, jumping_pred)

    print(prediction)
    print(values)

    fig = plt.bar(values, prediction, color='blue', width=0.4)
    plt.xlabel = 'Prediction Options'
    plt.ylabel = 'No. of Segments'
    plt.title('Output Prediction Results')

    plot_button = tk.Button(window, fig, text='Plot')
    plot_button.pack()

    window.update()
    # Output CSV
    prediction_csv = pd.DataFrame(columns=["Time (s)", "Prediction"])
    prediction_csv['Time (s)'] = [(str(0 + i * 5) + " - " + str(5 + i * 5) + " seconds") for i in
                                  range(0, len(prediction_results))]

    prediction_labels = ["walking" if i == 0 else "jumping" for i in prediction_results]
    prediction_csv["Prediction"] = prediction_labels

    # Save dataframe and use filedialog box to get the name and filepath from user

    file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV file', '*.csv')])

    # Save the dataframe
    prediction_csv.to_csv(file_path, index=False)


# Create a button to open the file dialog and read the CSV file

openCSVFileButton = tk.Button(window, text="Open CSV File", command=readCSV)
openCSVFileButton.pack()


# Create a widget to display text
text = tk.Text(window)
text.pack()

# Start GUI
window.mainloop()



