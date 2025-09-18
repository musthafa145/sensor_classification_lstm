import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from load_signals import load_signals, load_labels
from hot_encode import y_test_cat

# Load test data again
X_test = load_signals("test")
y_test = load_labels("test")

# Load trained model
model = load_model("har_lstm.h5")

# Predict classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
                         "SITTING","STANDING","LAYING"],
            yticklabels=["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
                         "SITTING","STANDING","LAYING"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - HAR LSTM Model")
plt.show()

# Classification report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred_classes,
      target_names=["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
                    "SITTING","STANDING","LAYING"]))
