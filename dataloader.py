import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(img_size=(224,224),batch_size=32):

    base_path=os.path.join("data","New Plant Diseases Dataset(Augmented)","New Plant Diseases Dataset(Augmented)")
    train_path=os.path.join(base_path,"train")
    valid_path=os.path.join(base_path,"valid")

    train_datagen=ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    valid_datagen=ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    valid_generator=valid_datagen.flow_from_directory(
        valid_path,
        batch_size=batch_size,
        target_size=img_size,
        class_mode="categorical"
    )

    return train_generator,valid_generator