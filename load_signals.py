import numpy as np
import pandas as pd

DATASET_PATH = "."

# The 9 raw inertial signals we’ll load
SIGNALS = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

def load_signals(split="train"):
    signal_data = []
    for signal in SIGNALS:
        filename = f"{DATASET_PATH}/{split}/Inertial Signals/{signal}{split}.txt"
        data = pd.read_csv(filename, sep=r"\s+", header=None).to_numpy()
        signal_data.append(data)
    return np.transpose(np.array(signal_data), (1, 2, 0))  # (n_samples, 128, 9)

def load_labels(split="train"):
    filename = f"{DATASET_PATH}/{split}/y_{split}.txt"
    labels = pd.read_csv(filename, sep=r"\s+", header=None).to_numpy().flatten()
    return labels - 1  # shift labels from 1–6 to 0–5

# 🔹 Always load data when imported
X_train = load_signals("train")
X_test = load_signals("test")
y_train = load_labels("train")
y_test = load_labels("test")

if __name__ == "__main__":
    print("Train set:", X_train.shape, y_train.shape)
    print("Test set:", X_test.shape, y_test.shape)
    print("Sample labels:", y_train[:10])
