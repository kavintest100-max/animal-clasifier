import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

st.title("🐾 Animal Image Identifier")
st.write("Drop an image of an animal below, and the AI will try to identify it!")

uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🧠 AI is thinking...")

    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    best_match = decoded_predictions[0]
    animal_name = best_match[1].replace("_", " ").title()
    confidence = best_match[2] * 100

    st.subheader("Prediction Result:")
    st.metric(label="Identified Animal", value=f"{animal_name}")
    st.write(f"Confidence Level: **{confidence:.2f}%**")
