# print_model_input.py
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("har_lstm.h5")

# Full summary
model.summary()

# Helpful explicit prints
print("\n=== Model input info ===")
try:
    print("model.input_shape:", model.input_shape)
except Exception:
    print("model.inputs:", model.inputs)

print("\n=== Layer-by-layer input/output shapes ===")
for i, layer in enumerate(model.layers):
    try:
        print(f"{i:02d} | {layer.name:20s} | input_shape={layer.input_shape} | output_shape={layer.output_shape}")
    except Exception:
        # some layers may not expose shapes
        print(f"{i:02d} | {layer.name:20s} | (shape info not available)")
