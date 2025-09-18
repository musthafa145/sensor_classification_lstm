import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("har_lstm.h5")

# Load activity labels
labels_df = pd.read_csv("activity_labels.txt", sep="\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(labels_df.id, labels_df.activity))

st.title("📱 Human Activity Recognition (HAR) Demo")

st.write("""
Upload a CSV file where:
- Each row = one sample
- Each sample has **1152 values** (128 timesteps × 9 features)
- Optionally, include a column named `label` (1–6) for the true activity
""")

uploaded_file = st.file_uploader("Upload reshaped CSV file", type="csv")

if uploaded_file:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data (first 5 rows)")
    st.write(df.head())

    # Check for true labels
    true_labels = None
    if "label" in df.columns:
        true_labels = df["label"].values
        df = df.drop(columns=["label"])

    # Ensure correct feature size
    if df.shape[1] != 128 * 9:
        st.error(f"Expected 1152 features (128×9), but got {df.shape[1]}. Please upload correct data.")
    else:
        # Reshape into (samples, 128, 9)
        X = df.values.reshape((-1, 128, 9))

        # Predict
        predictions = model.predict(X)
        pred_labels = np.argmax(predictions, axis=1) + 1  # labels are 1–6
        pred_activities = [activity_map[i] for i in pred_labels]

        st.write("### Predictions")
        for i, act in enumerate(pred_activities):
            if true_labels is not None:
                true_act = activity_map[true_labels[i]]
                st.write(f"Sample {i+1}: Predicted → **{act}**, True → {true_act}")
            else:
                st.write(f"Sample {i+1}: Predicted → **{act}**")
