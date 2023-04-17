# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:05:12 2023
@author: paur
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn import metrics
import psutil
import seaborn as sns
import pandas as pd
import pathlib
from datetime import datetime


path_destination = "your_path/"
path_data_source="your_path" 
test_dir = "your_path/test" 

df=[{'dataset':1, 'epochs':1, 'DA':1, 'layers':1, 'train':1, 'test':1, 'exec.time':1,'ram':1, 'cpu':1}]
df=pd.DataFrame(data=df)
datasets=["data1","data10","data25","data50","data75","data_con1","data_con10", "data_con25", "data_con50", "data_con75", "data100"]
classes=10
epochs_vector=[10]
unfreeze_layers=[-20]

#Define some parameters for the loader:
batch_size = 32
img_height = 150
img_width = 150



def split_tratin_test_set (dataset):
    train_dir=path_data_source + dataset +"/" + "train"
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
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.20),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #tf.keras.layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def build_model(num_classes,aprov_pre):
    inputs = tf.keras.layers.Input(shape=(img_height,img_width,3))
    inputs_re = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(inputs)
    if aprov_pre==True:
        y = img_augmentation(inputs_re)
        model_input = InceptionV3(include_top=False, input_tensor=y, weights="imagenet")
        print("preprocessing:",aprov_pre)
    else:
        model_input = InceptionV3(include_top=False, input_tensor=inputs_re, weights="imagenet")
        print("preprocessing:",aprov_pre)
    # Freeze the pretrained weights
    model_input.trainable = False
    x = model_input.output
    x = tf.keras.layers.Flatten()(x)
    predictions = tf.keras.layers.Dense(num_classes,activation = "softmax")(x)
    model=tf.keras.models.Model(inputs,predictions)
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def unfreeze_model(model,num):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[num:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

def results (model,test_data,dataset,name):
    y_pred=[]
    results=model.predict(test_data)
    for i in results:
        y_pred.append(np.argmax(i))

    y_test=[]
    for test_image,test_label in test_data:
        for t in test_label:
            print(np.array(t))
            y_test.append(np.argmax(t))
    print("")
    print("Precision: {}%".format(100*metrics.precision_score(y_test,y_pred, average="weighted")))
    print("Recall: {}%".format(100*metrics.recall_score(y_test,y_pred, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(y_test,y_pred, average="weighted")))
    print("Error: {}%".format(metrics.mean_absolute_error(y_test,y_pred)))
    report=classification_report(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred)
    print('\Report\n')
    print(report)

    matrix=pd.DataFrame(matrix)
    matrix.to_csv(path_destination+"cf_matrix" + str(name)+ str(dataset)+ ".csv", index=False)
    ax = plt.plot()
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    plt.savefig(path_destination+"cf_matrix_fig_inception_"+str(dataset)+"_"+str(name)+".png")
    ## Display the visualization of the Confusion Matrix.
    plt.show()



def training_plots(datasets,name):
    fig, ax = plt.subplots()
    for  j in range (len (datasets)):
        ax.plot(models[j].history["val_accuracy"], label=datasets[j])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        ax.legend(loc='best')
        #plt.legend(["train", "validation"], loc="upper left")
        plt.savefig(path_destination+"_val_accuracy_inception" + str(j)+"_" + str(name)+ ".png")
        plt.show()


def training_results_fine (datasets,name):
    fig, ax = plt.subplots()
    for  j in range (len (datasets)):
        ax.plot(models[j].history["val_accuracy"], label=datasets[j])
        ax.plot([10-1, 10-1],
        plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        ax.legend(loc='best')
        #plt.legend(["train", "validation"], loc="upper left")
        plt.show()
        fig.savefig(path_destination+"val_accuracy_fine_inception_" + str(j)+ ".png")

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

models=[]

for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    for i in epochs_vector:
        model = build_model(num_classes=classes,aprov_pre=False)
        start=datetime.now()
        hist = model.fit(train_data,
                    epochs=i,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    # Go through less of the validation data so epochs are faster (we want faster experiments!)
                    validation_steps=int(0.25 * len(test_data)),
                    verbose=1 )
        end=datetime.now()
        td= (end-start)
        print("----------------- MODEL----------------------------")
        cpu_percent_cores = psutil.cpu_percent(interval=2, percpu=True)
        avg = sum(cpu_percent_cores)/len(cpu_percent_cores)
        cpu_percent_total_str = ('%.2f' % avg)
        cpu_percent_cores_str = [('%.2f' % x) + '%' for x in cpu_percent_cores]
        df1=[{'dataset':j, 'epochs':i,'DA':0, 'layers':0, 'train':round(hist.history['accuracy'][-1],4), 'test':round(hist.history['val_accuracy'][-1],4),
              'exec.time':td,'ram':psutil.virtual_memory()[3]/1000000000, 'cpu': cpu_percent_total_str}]
        df1=pd.DataFrame(data=df1)
        df=pd.concat([df,df1])
        print('Total: {}'.format(cpu_percent_total_str))
        print('Individual CPUs: {}'.format('  '.join(cpu_percent_cores_str)))
        print("Dataset:",j,"Epochs:",i)
        #print("CPU utilization: ", td_cpu)
        print(f"The time of execution of above program is : {td}ms")
        # Calling psutil.cpu_precent()for 4 seconds
        print('The CPU usage is: ', psutil.cpu_times())
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        print("---------------------------------------------------")
        models.append(hist)
        results (model,test_data,j,"test1")
        model.save(path_destination+"inception" + "_test1_" + str(j))
        model.save(path_destination+"inception_h" + "_test1_"  + str(j) + ".h5")



training_plots(datasets,"test1")

models=[]
for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    for i in epochs_vector:
        for l in unfreeze_layers:
            model = build_model(num_classes=classes,aprov_pre=False)
            unfreeze_model(model,l)
            start=datetime.now()
            hist= model.fit(train_data,
                    epochs=i,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    # Go through less of the validation data so epochs are faster (we want faster experiments!)
                    validation_steps=int(0.25 * len(test_data)),
                     # callbacks=[callback],
                    verbose=1 )
            end=datetime.now()
            #end_cpu=psutil.cpu_percent(interval=None)
            # find difference loop start and end time and display
            td= (end-start)
            #td_cpu=(end_cpu-start_cpu)
            print("----------------- MODEL+FINE T----------------------")
            cpu_percent_cores = psutil.cpu_percent(interval=2, percpu=True)
            avg = sum(cpu_percent_cores)/len(cpu_percent_cores)
            cpu_percent_total_str = ('%.2f' % avg)
            cpu_percent_cores_str = [('%.2f' % x) + '%' for x in cpu_percent_cores]
            df1=[{'dataset':j, 'epochs':i,'DA':0, 'layers':l, 'train':round(hist.history['accuracy'][-1],4), 'test':round(hist.history['val_accuracy'][-1],4),
              'exec.time':td,'ram':psutil.virtual_memory()[3]/1000000000, 'cpu': cpu_percent_total_str}]
            df1=pd.DataFrame(data=df1)
            df=pd.concat([df,df1])
            print('Total: {}'.format(cpu_percent_total_str))
            print('Individual CPUs: {}'.format('  '.join(cpu_percent_cores_str)))
            #print("CPU utilization: ", td_cpu)
            print("Dataset: ",j,"epochs: ",i,"layers: ",l)
            print(f"The time of execution of above program is : {td}ms")
            # Calling psutil.cpu_precent()for 4 seconds
            print('The CPU usage is: ', psutil.cpu_times())
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print("---------------------------------------------------")
            models.append(hist)
            results (model,test_data,j, "test2")
            model.save(path_destination+"inception" + "_test2_" + str(j))
            model.save(path_destination+"inception_h" + "_test2_"  + str(j) + ".h5")


training_results_fine(datasets,"test2")
models=[]

for j in datasets:
    train_data,test_data=split_tratin_test_set(j)
    for i in epochs_vector:
        for l in unfreeze_layers:
            model = build_model(num_classes=10,aprov_pre=True)
            unfreeze_model(model,l)
            start=datetime.now()
            #start_cpu=psutil.cpu_percent(interval=1)
            hist = model.fit(train_data,
                    epochs=i+5,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    # Go through less of the validation data so epochs are faster (we want faster experiments!)
                    validation_steps=int(0.25 * len(test_data)),
                      callbacks=[callback],
                    verbose=1 )
            end=datetime.now()
            td= (end-start)
            #td_cpu=(end_cpu-start_cpu)
            print("----------------- MODEL+PRE+FINE T-----------------")
            cpu_percent_cores = psutil.cpu_percent(interval=2, percpu=True)
            avg = sum(cpu_percent_cores)/len(cpu_percent_cores)
            cpu_percent_total_str = ('%.2f' % avg)
            cpu_percent_cores_str = [('%.2f' % x) + '%' for x in cpu_percent_cores]
            df1=[{'dataset':j, 'epochs':i,'DA':1, 'layers':l, 'train':round(hist.history['accuracy'][-1],4), 'test':round(hist.history['val_accuracy'][-1],4),
              'exec.time':td,'ram':psutil.virtual_memory()[3]/1000000000, 'cpu': cpu_percent_total_str}]
            df1=pd.DataFrame(data=df1)
            df=pd.concat([df,df1])
            print('Total: {}'.format(cpu_percent_total_str))
            print('Individual CPUs: {}'.format('  '.join(cpu_percent_cores_str)))
            print("Dataset: ",j,"epochs: ",i,"layers: ",l)
            #print("CPU utilization: ", td_cpu)
            print(f"The time of execution of above program is : {td}ms")
            # Calling psutil.cpu_times()for 4 seconds
            print('The CPU usage is: ', psutil.cpu_times())
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print("---------------------------------------------------")
            models.append(hist)
            results (model,test_data,j, "test3")
            model.save(path_destination+"inception" + "_test3_" + str(j))
            model.save(path_destination+"inception_h" + "_test3_"  + str(j) + ".h5")

training_results_fine(datasets,"test3")
df.to_csv(path_destination+"inception.csv", index=False)





