from model import build_model
import tensorflow as tf

model = build_model(input_shape=(224, 224, 3), num_classes=38)
model.load_weights("best_model.h5")

# Save the model with both architecture and weights
model.save("best_model.keras", save_format="keras", include_optimizer=False)
