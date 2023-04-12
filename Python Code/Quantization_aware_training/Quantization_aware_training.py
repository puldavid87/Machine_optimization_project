# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:25:33 2023

@author: paur
"""

import tensorflow_model_optimization as tfmot
import tensorflow as tf

#Define some parameters for the loader:
batch_size = 32
img_height = 224
img_width = 224

def split_tratin_test_set (dataset):
    train_dir="C:/Users/paur/Documents/Tomato_leaves_illness_detection/DATASET_1" +"/" + dataset +"/" + "train"
    test_dir="C:/Users/paur/Documents/Tomato_leaves_illness_detection/DATASET_1" + "/" + dataset +"/" + "test"
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
    
train_data,test_data=split_tratin_test_set("tomato100")

quantize_model = tfmot.quantization.keras.quantize_model
model=tf.keras.models.load_model("MOBILE_tomato1000")

supported_layers = [tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.ReLU]

def apply_quantization_to_dense(layer):
  for supported_layer in supported_layers:
    if isinstance(layer, tf.keras.layers.Dense):
      return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer

annotated_model = tf.keras.models.clone_model(model,
    clone_function=apply_quantization_to_dense,
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
quant_aware_model.summary()

# `quantize_model` requires a recompile.
quant_aware_model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

quant_aware_model.fit(train_data,
        epochs=5,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        # Go through less of the validation data so epochs are faster (we want faster experiments!)
        validation_steps=int(0.15 * len(test_data)),
       callbacks=[callback],
        verbose=1 )
quant_aware_model.save("MOBILE_AWARE_Q")