from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time
import os
import random
data_folder = "/home/paur/Quantization/birds/"
models = ["efficient"]
figure_path = data_folder + "birds_test/test/"

labels=[]
for i in (os.listdir(figure_path)):
          labels.append(i)
          print(i)
          
number_of_samples=len(os.listdir(figure_path))
print(number_of_samples)
t1=time.time()
model_path = data_folder+"vgg16_test2_data100_quant_int.tflite"
#label_path = data_folder + "labels_mobilenet_quant_v1_224.txt"
interpreter = Interpreter(model_path)
print("Model Loaded Successfully:", i)
interpreter.allocate_tensors()
# Get input and output tensors.
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")
# Test the model on random input data.
aux=0
datos=[0]
t1=time.time()
for i in labels:
     file_names_train = os.listdir(figure_path + i)
     images_batch = random.sample(file_names_train,5)
     for j in images_batch:
        image = Image.open(figure_path + i +"/" + j).convert('RGB').resize((width, height))
        image = np.array(image)
        processed_image = np.expand_dims(image, axis=0).astype(np.uint8)
        interpreter.set_tensor(input_index, processed_image)
        t2=time.time()
        interpreter.invoke()
        t3=time.time()
        time_taken=(t3-t2)*1000 #milliseconds
        print("time taken for Inference: ",str(time_taken), "ms")
        # Obtain results
        predictions =  interpreter.tensor(output_index)
        predictions = np.argmax(predictions()[0])
        result=labels[predictions]
        print("predict: ", result, "label: ", i)
        if result== i:
            aux+=1
t4=time.time()
time_taken=(t4-t1)*1000 #milliseconds
print("test set Inference: ",str(time_taken), "ms")
print(aux)
print(aux/(number_of_samples*5))
   
 
