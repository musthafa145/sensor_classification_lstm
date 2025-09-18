
import numpy as np
from tensorflow.keras.models import load_model
from load_signals import X_train, X_test, y_train, y_test
from hot_encode import y_train_cat, y_test_cat
from lstm_model import model

# Train the model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=10,
    batch_size=64,
    verbose=1
)

# Save the trained model
model.save("har_lstm.h5")

print("✅ Model trained and saved as har_lstm.h5")
