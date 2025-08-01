from dataloader import get_data_generators
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

img_size=(224,224)
batch_size=16
epochs=10
num_classes=38

train_gen,valid_gen=get_data_generators(img_size=img_size,batch_size=batch_size)

model=build_model(input_shape=(224,224,3),num_classes=num_classes)

model.compile(optimizer=Adam(learning_rate=0.0005),loss="categorical_crossentropy",metrics=['accuracy'])

checkpoint_cb=ModelCheckpoint("best_model.h5",save_best_only=True)
earlystop_cb=EarlyStopping(patience=3,restore_best_weights=True,monitor='val_loss')

history=model.fit( # history stores los and acuracy of each epochs for potting or showing
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    callbacks=[checkpoint_cb,earlystop_cb]
)

model.save("final_model.h5")