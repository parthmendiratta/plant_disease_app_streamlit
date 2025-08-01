import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Paths
test_dir=os.path.join("data","test","test")
model_path="best_model.h5"

# Image Parameter
img_size=(224,224)
batch_size=16

# Load Trained model
model=load_model(model_path)

# Date Preprocessing
test_datagen=ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)



# Evaluate
loss,acc=model.evaluate(test_generator)
print(f"Test Accucracy: {acc}")
print(f"Test loss: {loss}")

y_pred=model.predict(test_generator) # this is the precition of each image , each mahe as 38 diferent probabilities for each class
y_pred_classes=np.argmax(y_pred,axis=1) # np.argmax choose the max probability from each image and notes its index and create a list of it
y_true=test_generator.classes # this is the true list of indices

class_labels=list(test_generator.class_indices.keys()) # we will get all the label names assigned to photos

print("Classification Report: \n")
print(classification_report(y_true,y_pred_classes,target_names=class_labels))

print("Confusion Matrix:\n")
print(confusion_matrix(y_true,y_pred_classes))