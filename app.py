import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model
with open("svm_image_classifier_model.pkl", "rb") as f:
    model = joblib.load(f)

# Mapping from prediction to class name
class_dict = {0: "Apple", 1: "Orange"}

# UI
st.title("🍎🍊 Fruit Classifier")
st.write("Upload an image of a fruit (Apple or Orange) to classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and flatten the image
    image = image.resize((100, 100))
    image_array = np.array(image)

    # Handle images with alpha channel (RGBA)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]  # Drop alpha channel

    # Flatten the image to 1D
    image_array = image_array.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(image_array)[0]
    prediction_name = class_dict.get(prediction, "Unknown")

    # Show result
    st.write(f"### 🧠 Prediction: **{prediction_name}**")
