import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = Image.open(uploaded_file)
    print("Original mode:", img.mode)
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)  # ✅ fix resize rounding
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print("Final image array shape:", img_array.shape)

    if img_array.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, but got {img_array.shape[-1]}")

    return img_array, img

def decode_prediction(pred_index,class_labels_dict):
    index_to_label= {v: k for k, v in class_labels_dict.items()}
    raw_label=index_to_label.get(pred_index,"Unknown")
    readable=raw_label.replace("___"," - ").replace("_"," ")
    return readable