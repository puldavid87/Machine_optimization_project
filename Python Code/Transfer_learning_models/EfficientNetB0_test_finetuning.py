# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras_flops import get_flops


import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.applications import EfficientNetB0
import psutil

datasets=["tomato100","tomato200","tomato310","tomato400","tomato500","tomato600","tomato1000"]
epochs_vector=[5,10,15]



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

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def unfreeze_model(model,num):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[num:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()    



for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    for i in epochs_vector:
        model = build_model(num_classes=10,aprov_pre=False)
        start=datetime.now() 
        start_cpu=psutil.cpu_percent(interval=1)
        hist_m = model.fit(train_data,
                    epochs=i,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    # Go through less of the validation data so epochs are faster (we want faster experiments!)
                    validation_steps=int(0.25 * len(test_data)),
                    verbose=1 )
        end=datetime.now()
        end_cpu=psutil.cpu_percent(interval=None)
        # find difference loop start and end time and display
        td= (end-start)
        td_cpu=(end_cpu-start_cpu)
        flops = get_flops(model, batch_size=batch_size)
        print("----------------- MODEL----------------------------")
        print("Dataset:",j,"Epochs:",i)
        print("CPU utilization: ", td_cpu)
        print(f"The time of execution of above program is : {td}ms")
        # Calling psutil.cpu_precent()for 4 seconds
        cpu_percent_cores = psutil.cpu_percent(interval=2, percpu=True)
        avg = sum(cpu_percent_cores)/len(cpu_percent_cores)
        cpu_percent_total_str = ('%.2f' % avg) + '%'
        cpu_percent_cores_str = [('%.2f' % x) + '%' for x in cpu_percent_cores]
        print('Total: {}'.format(cpu_percent_total_str))
        print('Individual CPUs: {}'.format('  '.join(cpu_percent_cores_str)))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):',psutil.virtual_memory()[3]/1000000000)
        print('RAM active (GB):',psutil.virtual_memory()[4]/1000000000)
        print(f"FLOPS: {flops / 10 ** 9:.03} G")
        print("---------------------------------------------------")
        plot_hist(hist_m)

for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    for i in epochs_vector:
        model = build_model(num_classes=10,aprov_pre=True)
        start=datetime.now() 
        hist_m_p = model.fit(train_data,
                    epochs=i,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    # Go through less of the validation data so epochs are faster (we want faster experiments!)
                    validation_steps=int(0.25 * len(test_data)),
                    verbose=1 )
        end=datetime.now()
        # find difference loop start and end time and display
        td= (end-start)
        flops = get_flops(model, batch_size=batch_size)
        print("----------------- MODEL+PRE------------------------")
        print("Dataset:",j,"Epochs:",i)
        print(f"The time of execution of above program is : {td}ms")
        # Calling psutil.cpu_precent()for 4 seconds
        print('The CPU usage is: ', psutil.cpu_times_percent(5))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):',psutil.virtual_memory()[3]/1000000000)
        print(f"FLOPS: {flops / 10 ** 9:.03} G")
        print("---------------------------------------------------")
        plot_hist(hist_m_p)


unfreeze_layers=[-20]
for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    for i in epochs_vector:
        for l in unfreeze_layers:
            model = build_model(num_classes=10,aprov_pre=False)
            unfreeze_model(model,l)
            start=datetime.now() 
            hist_fine = model.fit(train_data,
                    epochs=i+5,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    # Go through less of the validation data so epochs are faster (we want faster experiments!)
                    validation_steps=int(0.25 * len(test_data)),
                    verbose=1 )
            end=datetime.now()
            # find difference loop start and end time and display
            td= (end-start)
            print("----------------- MODEL+FINE T----------------------")
            print(f"The time of execution of above program is : {td}ms")
            # Calling psutil.cpu_precent()for 4 seconds
            print('The CPU usage is: ', psutil.cpu_times_percent(5))
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print('RAM Used (GB):',psutil.virtual_memory()[3]/1000000000)
            print("---------------------------------------------------")
            plot_hist(hist_fine)


unfreeze_layers=[-20]
for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    for i in epochs_vector:
        for l in unfreeze_layers:
            model = build_model(num_classes=10,aprov_pre=True)
            unfreeze_model(model,l)
            start=datetime.now() 
            hist_fine_p = model.fit(train_data,
                    epochs=i+5,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    # Go through less of the validation data so epochs are faster (we want faster experiments!)
                    validation_steps=int(0.25 * len(test_data)),
                    verbose=1 )
            end=datetime.now()
            # find difference loop start and end time and display
            td= (end-start)
            print("----------------- MODEL+PRE+FINE T-----------------")
            print(f"The time of execution of above program is : {td}ms")
            # Calling psutil.cpu_precent()for 4 seconds
            print('The CPU usage is: ', psutil.cpu_times_percent(5))
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print('RAM Used (GB):',psutil.virtual_memory()[3]/1000000000)
            print("---------------------------------------------------")
            plot_hist(hist_fine_p)
