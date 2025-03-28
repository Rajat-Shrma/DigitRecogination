import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2



# Set up the Streamlit app
st.title("Digit Recognition App")
st.write("Draw a digit in the box below and click 'Predict' to see the result.")

# Streamlit drawable canvas component
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    stroke_width=20,     # Pen thickness
    stroke_color="white",  # Pen color
    background_color="black",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Process the input and predict
try:
    model = load_model("model.h5")
except Exception as e:
    model = load_model("model.h5")

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Preprocess the image
        
        img=canvas_result.image_data
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))  # Resize to 28x28 for MNIST model
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
        
        # Predict the digit
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        
        # Display the result
        st.write(f"Predicted Digit: {digit}")
    else:
        st.write("Please draw a digit in the box above.")
# Load the pre-trained model
