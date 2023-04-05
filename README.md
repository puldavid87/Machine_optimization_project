# Machine Optimization Project

## Dataset structure:

* main_folder
  * train
    * Label 1
    * Label 2
    * Label n
  * Validation
    * Label 1
    * Label 2
    * Label n 
  * test
    * Label 1
    * Label 2
    * Label n

## Explore Data

```Linux

#check folders 
!ls main_folder

#check labels
!ls main_folder/train/

#check samples insite of the directory/labels
!ls main_folder/train/Label 1/
     

## Explore Data in python
```bash
pip install os pathlib 
```

```python
import os

# Walk through pizza_steak directory and list number of files
for dirpath, dirnames, filenames in os.walk("main_folder"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
  
# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("main_folder/train/steak"))
num_steak_images_train

# Get the class names (programmatically, this is much more helpful with a longer list of classes)
import pathlib
import numpy as np
data_dir = pathlib.Path("main_folder/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)

```
z
Datasets:

* BIRDS CLASSIFICATION: 
* TOMATO LEAVES ILLNESS DETECTION:
* FRUTS CLASIFICATION:
* WASTE CLASSIFICATION
* SOLAR PANELS FAILURE DETECTION:

