#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import sklearn
#import pandas as pd
import os
import sys
import time
import tensorflow as tf
import pathlib
import random
import glob
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.metrics import classification_report
import load_dataset_tif
#import load_dataset
import pandas as pd
from arguments import *

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

start = time.time()
#epoch_no = 30

print(tf.__version__)

#print(sys.version_info)
#for module in mpl, np, pd, sklearn, tf, keras:
    #print(module.__name__, module.__version__)

#tf.enable_eager_execution()

#tf.test.is_gpu_available()

# Check if the folder exists; if a folder exists, clear all its data. If no folder exists, create a new one.
def check(path):  
    paths = path.split('/')
    folder_name = paths[-1]
    path = path.replace('/' + folder_name,'')
        
    #print('hello world!!')
    if not path.endswith('/'):
        path = path + '/'
    
    try:
        os.mkdir(path + folder_name)
        print('new folder!!')
    except FileExistsError:
        files = os.listdir(path + folder_name)
        print("folder exist")
        for file in files:
            print('del ' + file)
            os.remove(path + folder_name + '/' + file)

### Generate Results

output_Path = "./Results/outcome_pictures/" + args.band_set
check(output_Path)

### Generate Results
classification_report_out = "./Results/outcome_pictures/" + args.band_set + "/2022_tree_VGG19_Original.txt"
f = open(classification_report_out, 'a')


print('!!!Data preparation!!!')
##　WARNING:tensorflow:Expected a shuffled dataset but input dataset `x` is not shuffled. Please invoke `shuffle()` on input dataset.
epoch_no = 30
BATCH_SIZE=args.batch_size # 4
train_dataset, test_dataset, val_dataset, train_count, test_count, val_count = load_dataset_tif.call()
#train_dataset, test_dataset, val_dataset, train_count, test_count, val_count = load_dataset.call()
#print(train_dataset.as_numpy_iterator())

print('Batch_size: ',args.batch_size)

print('train start!!!')
train_dataset=train_dataset.shuffle(buffer_size=train_count).repeat().batch(BATCH_SIZE)
val_dataset=val_dataset.batch(BATCH_SIZE)
test_dataset=test_dataset.batch(BATCH_SIZE)


keras=tf.keras
layers=tf.keras.layers
conv_base=keras.applications.VGG19(weights='imagenet',include_top=False) # original version


# modify version2
conv_base=keras.applications.VGG19(weights=None,include_top=False, input_shape=(64, 64, args.band_num)) # original version 256, 256
conv_base.summary()


# Afterward, you need to flatten this network.
model=keras.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8,activation='softmax'))

# I want the weights in the network to stay put and the convolutional product to be untrainable.
conv_base.trainable=False

#model = keras.Model(inputs=[input0, input1, input2])

model.summary()


model.compile(#optimizer='adam',
                        optimizer=keras.optimizers.Adam(lr=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['acc']                   
)
steps_per_epoch=train_count//BATCH_SIZE
validation_steps=val_count//BATCH_SIZE


# Early end callback function, listener val_acc
early_stopping = EarlyStopping(
    monitor='val_acc',
    min_delta=0.0001,
    patience=10
)


start_time = time.time()

history=model.fit(train_dataset,
                            epochs=epoch_no,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset,
                            validation_steps=validation_steps,
                            callbacks=[early_stopping])


end_time = time.time()
mins = (end_time - start_time) // 60
secs = (end_time - start_time) % 60
print("VGG19 Training Time: {}:{:.2f}".format(mins, secs))

modelPath = './Results/outcome_pictures/' + args.band_set + '/VGG19_model.h5'

model.save(modelPath)  # creates a HDF5 file 'VGG19_model.h5'

f.write("VGG19 Training Time: {}:{:.2f}".format(mins, secs)+"\n")

# plot the training loss and accuracy
def plot_learning_curves(history):
	# plot the training loss and accuracy
    plt.figure(figsize=(8,5))
    titleName = "Training / Validation Loss and Accuracy on VGG19 for "+args.band_set # The title of the graph.
    plt.title(titleName ,y=1.05)
    plt.grid(True)
    plt.gca().set_ylim(0,2)
    plt.gca().set_xlim(0,epoch_no)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    #plt.xticks(np.linspace(0, epoch_no, 7))
    plt.plot(history.history["loss"], label="training_loss")
    plt.plot(history.history["val_loss"], label="validation_loss")
    plt.plot(history.history["acc"], label="training_accuracy")
    plt.plot(history.history["val_acc"], label="validation_accuracy")
    plt.legend(loc="upper right",fontsize=10)

    #curves_img_path = './Results/outcome_pictures/' + args.band_set + '/VGG19_Original_7_curves.png'
    curves_img_path = './Results/outcome_pictures/' + args.band_set + '/VGG19_Original_7_curves'
    i = 0
    while os.path.exists('{}{:d}.png'.format(curves_img_path, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(curves_img_path, i))
    #plt.savefig('./Results/outcome_pictures/' + args.band_set + '/VGG19_Original_7_curves.png')
    #plt.show()
	
plot_learning_curves(history)

start_time1=time.time()
# You must include the batch step, and you can only compute the loss and accuracy once.
model.evaluate(test_dataset,verbose=0)

end_time1 = time.time()
mins = (end_time1 - start_time1) // 60
secs = (end_time1 - start_time1) % 60
print("VGG19 Testing Time: {}:{:.2f}".format(mins, secs),(model.evaluate(test_dataset,verbose=0)))

f.write("VGG19 Testing Time: {}:{:.2f}".format(mins, secs)+"\n")

from sklearn.metrics import confusion_matrix
x, y_true = [], []
i = 0
for element in test_dataset:
    i += 1
    _x, _y = element	
    x.append(_x.numpy())
    y_true.append(_y.numpy())
    if i==test_count//BATCH_SIZE:
        break

x = np.concatenate(x, axis=0)
y_true = np.concatenate(y_true)
y_pred = model.predict(x, verbose=0)
y_pred = np.argmax(y_pred, axis=-1)
confmatrix = confusion_matrix(y_true, y_pred)

###############################################################################################################

target_names = os.listdir("../../" + args.band_set + "/train_set/")

###############################################################################################################

### 1. classification_report
#target_names = ["type01","type02","type03","type04","type05","type06","type07","type08"]
print(classification_report(y_true, y_pred, labels=range(len(target_names)), target_names=target_names))

f.write(classification_report(y_true, y_pred, labels=range(len(target_names)), target_names=target_names))
f.close()

### 2. ConfusionMatrixPlot
csv_path = './Results/outcome_pictures/' + args.band_set + '/2022_VGG19_Original_confusion_matrix.csv'
pd.DataFrame(confmatrix).to_csv(csv_path, mode = 'a')

def ConfusionMatrixPlot(confmatrix_Input):    
    #pd.DataFrame(confmatrix_Input).to_csv('confusion_matrix.csv')
    clsnames = np.arange(0, len(target_names))
    tick_marks = np.arange(len(clsnames))
    plt.figure(figsize=(len(target_names) + 1, len(target_names) + 1))
    titleName = "Confusion matrix of VGG19 for "+args.band_set # The title of the graph.
    plt.title(titleName,fontsize=15,pad=10)
    iters = np.reshape([[[i, j] for j in range(len(clsnames))] for i in range(len(clsnames))], (confmatrix_Input.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confmatrix_Input[i, j]), fontsize=15, va='center', ha='center')  # Display the corresponding numbers.

    plt.gca().set_xticks(tick_marks + 0.5, minor=True)
    plt.gca().set_yticks(tick_marks + 0.5, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')

    plt.imshow(confmatrix_Input, interpolation='nearest', cmap='cool')  # Display the matrix as pixels.
    plt.xticks(tick_marks,target_names) # 9-class with category names on the horizontal and vertical axes.
    plt.yticks(tick_marks,target_names)
   
    plt.ylabel('Actual Species',labelpad=-5,fontsize=15)
    plt.xlabel('Predicted Species',labelpad=10,fontsize=15)
    plt.ylim([len(clsnames) - 0.5, -0.5])
    plt.tight_layout()

    cb=plt. colorbar()# heatmap
   #cb.set_label('Numbers of Predict',fontsize = 15)
    Confusion_img_path = './Results/outcome_pictures/' + args.band_set + '/VGG19_Original_7'
    i = 0
    while os.path.exists('{}{:d}.png'.format(Confusion_img_path, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(Confusion_img_path, i))
    #plt.savefig('./Results/outcome_pictures/' + args.band_set + '/VGG19_Original_7.png')
    
matrixInput = np.array(confmatrix)

PercentageInput = (matrixInput.T / matrixInput.astype(np.float).sum(axis=1)).T

#print (PercentageInput)

AroundPercentageInput = np.around(PercentageInput, decimals=3)

print(AroundPercentageInput)

###############################################################
#'''
new = list(AroundPercentageInput)
AroundPercentageInput_new = []
for ns in new:
    #print(ns)
    #print(list(ns))

    count = len(list(ns))
    #print(count, type(count))

    ns = list(ns)
    ns = ns + [0.0] * (len(target_names)-count)
    #print(ns)

    ns = np.array(ns)
    #print(ns)

    AroundPercentageInput_new.append(ns)
    #print(AroundPercentageInput_new)

col_count = len(target_names) - len(AroundPercentageInput_new)
for i in range(col_count):
    ns = [0.0] * (len(target_names))
    AroundPercentageInput_new.append(ns)
#print(AroundPercentageInput_new)

#print(AroundPercentageInput_new)
AroundPercentageInput_new = np.array(AroundPercentageInput_new)
#print(AroundPercentageInput_new)

AroundPercentageInput_new = AroundPercentageInput_new.reshape((len(target_names), len(target_names)))
ConfusionMatrixPlot(AroundPercentageInput_new)

#################################################################

#ConfusionMatrixPlot(AroundPercentageInput)

end = time.time()

print(end - start)
