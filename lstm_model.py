from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Input shape: 128 timesteps, 9 features
timesteps = 128
features = 9
num_classes = 6

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(64),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
