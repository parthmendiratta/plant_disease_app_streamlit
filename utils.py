import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
def preprocess_image(uploaded_file,target_size=(224,224)):
    img=Image.open(uploaded_file).convert("RGB")
    img=img.resize(target_size)
    img_array=tf.keras.utils.img_to_array(img)
    img_array=img_array/255.0
    img_array=np.expand_dims(img_array,axis=0)
    print("Image mode:", img.mode)
    st.write("Image mode:", img.mode)
    st.write("Image shape before expand:", img_array.shape)

    return img_array,img 

def decode_prediction(pred_index,class_labels_dict):
    index_to_label= {v: k for k, v in class_labels_dict.items()}
    raw_label=index_to_label.get(pred_index,"Unknown")
    readable=raw_label.replace("___"," - ").replace("_"," ")
    return readable