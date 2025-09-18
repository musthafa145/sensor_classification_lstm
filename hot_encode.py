from load_signals import X_train, y_train, X_test, y_test
from tensorflow.keras.utils import to_categorical

# One-hot encode labels (6 activities → 6 classes)
y_train_cat = to_categorical(y_train, num_classes=6)
y_test_cat = to_categorical(y_test, num_classes=6)

print("y_train_cat shape:", y_train_cat.shape)
print("y_test_cat shape:", y_test_cat.shape)

# Expose for import in train_model.py
__all__ = ["X_train", "y_train", "X_test", "y_test", "y_train_cat", "y_test_cat"]
