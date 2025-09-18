import os
import pandas as pd

root = "."

print("\n=== First-level folders and files ===")
for item in os.listdir(root):
    print(item)

def preview_file(path, nrows=5):
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=None, nrows=nrows)
        shape = pd.read_csv(path, delim_whitespace=True, header=None).shape
        print(f"\n{path}: shape={shape}")
        print(df.head())
    except Exception as e:
        print(f"\n{path}: Could not load ({e})")

print("\n=== Preview Key Files ===")
for fname in ["train/X_train.txt", "train/y_train.txt", "test/X_test.txt", "test/y_test.txt"]:
    fpath = os.path.join(root, fname)
    if os.path.exists(fpath):
        preview_file(fpath)

# Show activity labels
fpath = os.path.join(root, "activity_labels.txt")
if os.path.exists(fpath):
    labels = pd.read_csv(fpath, delim_whitespace=True, header=None)
    print("\n=== Activity Labels ===")
    print(labels)
