from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(224,224,3),num_classes=38):
    base_model=EfficientNetB0(include_top=False,weights='imagenet',input_shape=input_shape)
    base_model.trainable=True

    fine_tune_at = 100  

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.3)(x)
    outputs=Dense(num_classes,activation='softmax')(x)

    model=Model(inputs=base_model.input,outputs=outputs)
    return model