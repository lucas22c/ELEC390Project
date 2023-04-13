import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Part456
import Part2

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
    model = get_model()

    # Read CSV
    fileData = pd.readCSV(file_path)

    # Drop selected columns
    cols_to_keep = ['Time (s)', 'Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)',
                    'Linear Acceleration z (m/s^2)']

    # Drop all other columns
    fileData = fileData[cols_to_keep]

    # Split the data into 5 second intervals
    split_fileData = np.array_split(fileData, len(fileData) / 500, axis=0)

    # Extract the features from the inputted data
    fileData_features = get_features(split_fileData)

    # Predict Input
    prediction_results = model.predict(fileData_features.to_numpy())

    # Graph the data
    fig, ax = plt.subplots()
    ax.plot(fileData.iloc[:, 0], fileData.iloc[:, 1:4])

    # Visualization of the data

    # Set X Label
    ax.set_xlabel("Time (s)")

    # Set Y Label
    ax.set_ylabel("Acceleration (m/s^2)")

    # Set Title
    ax.set_title("Acceleration vs Time")

    # Set Legend
    ax.legend(["X-acceleration", "Y-acceleration", "Z-acceleration"], loc="upper right")

    y_min = plt.ylim()[0] * 0.9

    print(len(prediction_results))
    for i in range(0, len(prediction_results)):
        ax.text(2.5 + i * 5, y_min, 'walking' if prediction_results[i] == 1 else 'jumping', ha='center', va='center',
                fontsize=9)

    ax.grid(axis='x')
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])

    # Output CSV
    prediction_csv = pd.DataFrame(columns=["Time (s)", "Prediction"])
    prediction_csv['Time (s)'] = [(str(0 + i * 5) + " - " + str(5 + i * 5) + " seconds") for i in
                                  range(0, len(prediction_results))]

    prediction_labels = ["walking" if i == 1 else "jumping" for i in prediction_results]
    prediction_csv["Prediction"] = prediction_labels

    # Save dataframe and use filedialog box to get the name and filepath from user

    file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV file', '*.csv')])

    # Save the dataframe
    prediction_csv.to_csv(file_path, index=False)

    plt.show()


# Create a button to open the file dialog and read the CSV file
openCSVFileButton = tk.Button(window, text="Open CSV File", command=readCSV)
openCSVFileButton.pack()

# Create a widget to display text
text = tk.Text(window)
text.pack()

# Start GUI
window.mainloop()
