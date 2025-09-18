from tensorflow.keras.models import load_model

model = load_model("har_lstm.h5")
print(model.summary())
