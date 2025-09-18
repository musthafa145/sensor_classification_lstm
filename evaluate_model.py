import numpy as np
from tensorflow.keras.models import load_model
from load_signals import load_signals, load_labels
from tensorflow.keras.utils import to_categorical

# Load test data
X_test = load_signals("test")
y_test = load_labels("test")
y_test_cat = to_categorical(y_test, num_classes=6)

# Load trained model
model = load_model("har_lstm.h5")

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test Loss: {loss:.4f}")

# Predict a few samples
y_pred = model.predict(X_test[:5])
y_pred_classes = np.argmax(y_pred, axis=1)

print("🔍 True labels:     ", y_test[:5])
print("🤖 Predicted labels:", y_pred_classes)
