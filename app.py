import streamlit as st
import tensorflow as tf

st.title("🧪 Model Load Test")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras", compile=False)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
