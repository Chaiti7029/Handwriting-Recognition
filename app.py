import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("Handwritten Digit Recognizer")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image = ImageOps.invert(image)  # invert to match MNIST format
    image = image.resize((28, 28))  # resize to 28x28

    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image)
    img_array = img_array / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"### Predicted Digit: {predicted_class}")
