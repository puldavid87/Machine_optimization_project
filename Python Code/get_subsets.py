# -*- coding: utf-8 -*-
"""

@author: Paul Rosero
"""
import os
import random
import shutil

train_dir="trainset_path"
test_dir="testset_path"
path="main_folder_path"         

# Nice way to get labels:
labels=[]
for i in (os.listdir(train_dir)):
          labels.append(i)
          
          
def create_folders(train_dir,test_dir, labels, num): 
  number_of_samples=len(os.listdir(train_dir))
  print(number_of_samples)
  path_dest=path+"data"+str(num)
  os.makedirs(path_dest)
  train_dest=path_dest + "/" + "train"
  test_dest=path_dest + "/" + "test"
  os.makedirs(train_dest)
  os.makedirs(test_dest)
  for i in labels:
    label_train_path=train_dest + "/" + str(i)
    os.makedirs(label_train_path)
    label_test_path=test_dest + "/" + str(i)
    os.makedirs(label_test_path)


def copy_images_percentages (train_dir,test_dir, labels, num,):
  train_dest = path+"/data" + str(num) + "/train"
  test_dest = path+"data" + str(num) + "/test"
  for i in labels:
    train_path=train_dest + "/" + i 
    test_path=test_dest + "/" + i 
    file_names_train = os.listdir(train_dir+"/" + i)
    file_names_test = os.listdir(test_dir + "/" + i )
    number_train = len(file_names_train)
    number_test = len(file_names_test)
    target_images_train = random.sample(file_names_train,int((num*number_train)/100))
    target_images_test = random.sample(file_names_test, number_test)
    for j, img in enumerate(target_images_train):
      shutil.copy(train_dir + "/" + str(i) + "/" + str(img),train_path + "/" + img)
    for j, img in enumerate(target_images_test):
      shutil.copy(test_dir + "/" + str(i) + "/" + str(img),test_path + "/" + img)

# subsets by selecting the percentage
samples=[1,10,25,50,75]   

# Get new subsets
for i in samples:
    create_folders(train_dir,test_dir,labels,i) 
    copy_images_percentages(train_dir,test_dir,labels,i)
