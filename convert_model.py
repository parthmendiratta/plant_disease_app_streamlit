from model import build_model
model = build_model(input_shape=(224, 224, 3), num_classes=38)
model.load_weights("best_model.h5")
model.save("trained_model.keras")
