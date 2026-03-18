import os
import json
from PIL import Image

import numpy as np

import tensorflow as tf
import streamlit as st


import gdown

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction.h5"

# Download model from Google Drive if not here
if not os.path.exists(model_path):
    with st.spinner("Downloading model... please wait"):
        gdown.download(
            "https://drive.google.com/file/d/1BELKvWfR8-WqumPknQq1CZPdITHJNiIk/view?usp=drive_link",
            model_path,
            quiet=False
        )

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Loading class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to numpy array
    img_array = np.array(img)
    # Add batch dimension  
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array = img_array.astype('float32') / 255.0
    return img_array


# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):  # ✅ Fixed: calss_indices → class_indices
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction)) * 100
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence


# Streamlit app
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title('🌿 Plant Disease Classifier')
st.write("Upload a plant leaf image to detect whether it is healthy or diseased.")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        resized_img = image.resize((300, 300))
        st.image(resized_img)

    with col2:
        st.subheader("Prediction")
        if st.button('Classify'):
            with st.spinner("Analyzing leaf..."):
                # ✅ Fixed: uploases_image → uploaded_image, classify_indices → class_indices
                prediction, confidence = predict_image_class(model, uploaded_image, class_indices)

            # ✅ Fixed: st.sucess → st.success, removed 'str' inside f-string
            st.success(f'Prediction: **{prediction}**')
            st.info(f'Confidence: {confidence:.2f}%')

            # Show if healthy or diseased
            if "healthy" in prediction.lower():
                st.balloons()
                st.success("✅ This plant appears to be healthy!")
            else:
                st.warning("⚠️ Disease detected! Consider consulting an agronomist.")
