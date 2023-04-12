# -*- coding: utf-8 -*-
"""
Spyder Editor
Python version 3.9 
TensorFlow 2.12.*

"""
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.applications import EfficientNetB0
#Datasets to train the ML model, in this case two datasets are used.
datasets=["dada","tomato1000"]
#Define some parameters for the loader:
batch_size = 32
img_height = 224
img_width = 224

train_set_path = "/"
test_set_path = "/"
def split_tratin_test_set (dataset):
    train_dir=train_set_path + dataset +"/" + "train"
    test_dir=test_set_path + dataset +"/" + "test"
    # Import data from directories and turn it into batches
    train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                  seed=123,
                                                                  label_mode="categorical",
                                                                  batch_size=batch_size, # number of images to process at a time 
                                                                  image_size=(img_height, img_width)) # convert all images to be 224 x 224

    test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                seed=123,
                                                                label_mode="categorical",
                                                                batch_size=batch_size, # number of images to process at a time 
                                                                image_size=(img_height, img_width)) # convert all images to be 224 x 224    
    return train_data, test_data
    

img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def build_model(num_classes,aprov_pre):
    inputs = tf.keras.layers.Input(shape=(img_height,img_width,3))
    if aprov_pre==True:
        x = img_augmentation(inputs)
        model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
        print("preprocessing:",aprov_pre)
    else:
        model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
        print("preprocessing:",aprov_pre)
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    model = build_model(num_classes=10,aprov_pre=False)
    unfreeze_model(model)
    start=datetime.now() 
    hist_fine = model.fit(train_data,
            epochs=10, # max epochs
            steps_per_epoch=len(train_data),
            validation_data=test_data,
            # Go through less of the validation data so epochs are faster (we want faster experiments!)
            validation_steps=int(0.15 * len(test_data)),
           callbacks=[callback],
            verbose=1 )
            # Save the model
    model.save("ffficient_b0" + "_" + str(j))
