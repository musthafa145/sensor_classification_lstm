import numpy as np
import pandas as pd

# Paths to your dataset (update if needed)
DATASET_PATH = r"C:\Users\91977\Desktop\SA_2025\archive (3)\UCI-HAR Dataset"

# Load features and activity labels
features = pd.read_csv(f"{DATASET_PATH}/features.txt", delim_whitespace=True, header=None)
activity_labels = pd.read_csv(f"{DATASET_PATH}/activity_labels.txt", delim_whitespace=True, header=None, index_col=0)

# Function to load a dataset split (train or test)
def load_split(split="train"):
    X = pd.read_csv(f"{DATASET_PATH}/{split}/X_{split}.txt", delim_whitespace=True, header=None)
    y = pd.read_csv(f"{DATASET_PATH}/{split}/y_{split}.txt", delim_whitespace=True, header=None)
    return X.values, y.values.ravel()

# Load train and test sets
X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# Example: show first few activity labels mapped to text
print("Sample activities:", [activity_labels.loc[i].values[0] for i in y_train[:10]])
