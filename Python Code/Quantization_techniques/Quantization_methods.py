# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:41:41 2023

@author: paur
"""

import tensorflow as tf
import os
import numpy as np
import pathlib
path= "path/"
dataset="tomato500"
#Define some parameters for the loader:
batch_size = 32
img_height = 224
img_width = 224
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()


def split_tratin_test_set ():
    train_dir= + dataset + "/" + "train"
    test_dir= + dataset +"/" + "test"
    # Import data from directories and turn it into batches
    train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(img_height, img_width),
                                               batch_size=batch_size,
                                               class_mode="categorical")# convert all images to be 224 x 224

    test_data = train_datagen.flow_from_directory(test_dir,
                                              target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              class_mode="categorical")    
    return train_data, test_data



train_images,test_images=split_tratin_test_set()
model="model_name"

#INFERENCES#
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("Path_to_save_tflite_models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"your_model_name.tflite"
tflite_model_file.write_bytes(tflite_model)


###Post-training float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"your_model_name_quant_f16.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)

#Post-training dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"your_model_name_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)

#Post-training integer quantization

def representative_data_gen():
  for input_value,_ in train_images.take(100):
    input_value=np.expand_dims(input_value[0], axis=0).astype(np.float32)
    yield [input_value]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_model_quant_float = converter.convert()
tflite_model_quant_float_file = tflite_models_dir/"your_model_name_quant_float.tflite"
tflite_model_quant_float_file.write_bytes(tflite_model_quant_float)

#Convert using integer-only quantization


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant_int = converter.convert()
tflite_model_quant_int_file = tflite_models_dir/"your_model_name_quant_int.tflite"
tflite_model_quant_int_file.write_bytes(tflite_model_quant_int)

#Load the model into the interpreters

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
interpreter_fp16.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()

interpreter_quant_float = tf.lite.Interpreter(model_path=str(tflite_model_quant_float_file))
interpreter_quant_float.allocate_tensors()

interpreter_quant_int = tf.lite.Interpreter(model_path=str(tflite_model_quant_int_file))
interpreter_quant_int.allocate_tensors()

      
# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    cont=0
    accurate_count = 0
    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    input_details = interpreter.get_input_details()[0]
    for test_image,test_label in test_images:
        test_label=np.expand_dims(test_label[0], axis=0).astype(np.float32)
        if input_details['dtype'] == np.uint8:
            test_image=np.expand_dims(test_image[0], axis=0).astype(np.uint8)
        else:
            test_image=np.expand_dims(test_image[0], axis=0).astype(np.float32)

        interpreter.set_tensor(input_index, test_image)
          # Check if the input type is quantized, then rescale input data to uint8
          # Run inference.
        interpreter.invoke()
    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
        test_label = np.argmax(test_label)
  # Compare prediction results with ground truth labels to calculate accuracy.
        cont += 1
        if digit == test_label:
            accurate_count += 1    
    accuracy = accurate_count * 1.0 / cont

    return accuracy

models=[interpreter, interpreter_fp16, interpreter_quant,interpreter_quant_float,interpreter_quant_int]
for model in models:
    score = evaluate_model(model)
    print(score)
    
print("######################################")
size_tf = os.path.getsize(tflite_model_file)
size_fp16_tflite = os.path.getsize(tflite_model_fp16_file)
size_quant_tflite = os.path.getsize(tflite_model_quant_file)
size_quant_float_tflite = os.path.getsize(tflite_model_quant_float_file)
size_quant_int_tflite = os.path.getsize(tflite_model_quant_int_file)
print ( f"{size_tf} bytes")
print ( f"{size_fp16_tflite} bytes")
print ( f"{size_quant_tflite} bytes")
print ( f"{size_quant_float_tflite} bytes")
print ( f"{size_quant_int_tflite} bytes")
print("######################################")
