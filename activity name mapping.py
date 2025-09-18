import pandas as pd

# Load activity labels
labels_path = "activity_labels.txt"
labels_df = pd.read_csv(labels_path, delim_whitespace=True, header=None, names=["id", "activity"])

# Convert to dictionary
activity_map = dict(zip(labels_df.id, labels_df.activity))

print("Activity mapping:")
print(activity_map)
