import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Processing Playground")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image_np, caption="Original Image", use_container_width=True)

    # Example: Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    st.image(gray, caption="Grayscale", use_container_width=True, channels="GRAY")