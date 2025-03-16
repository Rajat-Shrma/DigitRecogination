import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the pre-trained model
model = load_model("model.h5")

# Set up the Streamlit app
st.title("Digit Recognition App")
st.write("Draw a digit in the box below and click 'Predict' to see the result.")

# Streamlit drawable canvas component
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="black",  # Background color
    stroke_width=20,     # Pen thickness
    stroke_color="white",  # Pen color
    background_color="white",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Process the input and predict
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Preprocess the image
        img = cv2.resize(img, (28, 28))  # Resize to 28x28 for MNIST model
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(1, 28, 28, 3)  # Add batch and channel dimensions
        
        # Predict the digit
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        
        # Display the result
        st.write(f"Predicted Digit: {digit}")
    else:
        st.write("Please draw a digit in the box above.")
