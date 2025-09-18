import numpy as np
import pandas as pd
import os

# Paths to your test data
signals_path = "test/Inertial Signals"
y_test_path = "test/y_test.txt"

# 9 signals (same order used in HAR dataset)
signal_files = [
    "body_acc_x_test.txt",
    "body_acc_y_test.txt",
    "body_acc_z_test.txt",
    "body_gyro_x_test.txt",
    "body_gyro_y_test.txt",
    "body_gyro_z_test.txt",
    "total_acc_x_test.txt",
    "total_acc_y_test.txt",
    "total_acc_z_test.txt"
]

# Load signals (each has shape: [samples, 128])
signals = [np.loadtxt(os.path.join(signals_path, f)) for f in signal_files]

# Stack into shape: (samples, 128, 9)
X_test = np.transpose(np.array(signals), (1, 2, 0))

# Flatten each sample to 1152 columns
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Load labels
y_test = np.loadtxt(y_test_path).astype(int)

# Save to CSV (1152 features + label column)
df = pd.DataFrame(X_test_flat)
df["label"] = y_test
df.to_csv("reshaped_test.csv", index=False)

print(f"✅ Saved reshaped_test.csv with shape: {df.shape}")
