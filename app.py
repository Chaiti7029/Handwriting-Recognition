import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("✍ Handwritten Digit Recognizer")

st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Input Type:", ("Upload an Image", "Draw a Digit"))

def preprocess_image(image: Image.Image):
    image = image.convert("L")  # grayscale
    image = ImageOps.invert(image)  # MNIST expects white on black
    image = image.resize((28, 28))  # resize
    img_array = np.array(image) / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # add batch & channel dims
    return img_array

def predict_digit(image_array):
    if np.sum(image_array) == 0:
        st.warning("The image is empty or too faint. Please try again.")
    else:
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        st.success(f"### Predicted Digit: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

# Input Method 1: Upload an Image
if input_method == "Upload an Image":
    uploaded_file = st.file_uploader("Upload a digit image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        processed = preprocess_image(image)
        predict_digit(processed)

# Input Method 2: Draw a Digit
else:
    st.write("Draw a digit in the box below:")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
        st.image(image, caption="Your Drawing", width=150)
        processed = preprocess_image(image)
        predict_digit(processed)