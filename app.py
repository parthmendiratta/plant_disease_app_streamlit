import streamlit as st
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_image, decode_prediction
from model import build_model

st.set_page_config(layout="wide",page_title="Plant Disease Detector")

st.markdown("""
    <style>
    html, body, .stApp {
        background: #70e1f5;  /* fallback for old browsers */
        background: -webkit-linear-gradient(to right, #ffd194, #70e1f5);  /* Chrome 10-25, Safari 5.1-6 */
        background: linear-gradient(to right, #ffd194, #70e1f5); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
        background-size: cover;
        background-attachment: fixed;
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #6a1b9a;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #8e24aa;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Plant Disease Classifier")
st.write("Upload the leaf image and we'll predict the disease class with our trained EfficientNet model.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()

class_labels_dict = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3,
    'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20,
    'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32,
    'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36,
    'Tomato___healthy': 37
}

uploaded_file=st.file_uploader("Upload an image of a plant leaf",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analyzing...."):
        img_array,org_image=preprocess_image(uploaded_file)
        print("Final image array shape:", img_array.shape)  # ✅ Add this line
        prediction=model.predict(img_array)
        pred_index=np.argmax(prediction)
        confidence=float(np.max(prediction))

        human_label=decode_prediction(pred_index,class_labels_dict)

        
    st.image(org_image,caption="Uploaded Image",use_container_width=True)
    st.markdown(f"""
        <div style='
            background-color: #f0fff4;
            padding: 1.2rem;
            border-radius: 10px;
            border-left: 8px solid #38a169;
        '>
        <h3 style='color: #2f855a;'>✅ Diagnosis Result</h3>
        <p style='font-size: 1.1rem;'><strong>🧠 Prediction:</strong> <span style='color:#2c5282;'>{human_label}</span></p>
        <p style='font-size: 1.1rem;'><strong>🔍 Confidence:</strong> <span style='color:#2c5282;'>{confidence * 100:.2f}%</span></p>
    </div>
    """, unsafe_allow_html=True)
    


st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; margin-top: 40px; font-size: 0.85rem; color: #444; background-color: #f9f9f9; padding: 10px; border-radius: 8px;'>
        📝 **Note**:
        This model supports 38 different diseases across plants like:- 🍎 Apple, 🍇 Grape, 🌽 Corn, 🍅 Tomato, 🥔 Potato, 🍑 Peach, 🫑 Bell Pepper, 🍊 Orange, 🍓 Strawberry, 🫐 Blueberry, 🌿 Soybean, and more.
    </div>
    """,
    unsafe_allow_html=True
)