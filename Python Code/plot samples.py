# -*- coding: utf-8 -*-
"""
Visualize 3 images per label
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import random
import shutil
#Set the working directory
working_directory= "your_directory_path"
os.chdir(working_directory) # change directory from working dir to dir with files
train_dir="your_trainingset_path"

# Rugh way to get labels by hand:
labels=[
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Leaf_Mold",
"Tomato___Late_blight",
"Tomato___Target_Spot",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Septoria_leaf_spot",
"Tomato___healthy",
"Tomato___Tomato_mosaic_virus"
]

# Nice way to get labels:
labels=[]
for i in (os.listdir(train_dir)):
          labels.append(i)

def view_images(target_dir, target_class, num):
  target_path=target_dir + "/" + target_class
  file_names = os.listdir(target_path)
  target_images = random.sample(file_names, num)
  plt.figure(figsize=(15, 6))
  for i, img in enumerate(target_images):
    img_path = target_path + "/" + img
    plt.subplot(1, num, i+1)
    plt.imshow(mpimg.imread(img_path))
    plt.title(target_class)
    plt.axis("off")

# Print images
for i in labels:
  view_images(target_dir=train_dir ,target_class=i,num=3)

 
# Check one random image 
random_image = random.sample(os.listdir(train_dir+"/"+"Tomato___Septoria_leaf_spot/"),1)
 # Read in the image and plot it using matplotlib
img = mpimg.imread(train_dir +"/"+"Tomato___Septoria_leaf_spot/"  + random_image[0])
print(f"Image shape: {img.shape}")

# check one specific image
img = mpimg.imread("image_path")
print(f"Image shape: {img.shape}")
