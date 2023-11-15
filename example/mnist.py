"""test_core_layers.py
This file is part of the test suite for keras2c
Implements tests for core layers
"""


#!/usr/bin/env python3


import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.abspath(cwd+"/tests"))
sys.path.append(os.path.abspath(cwd+"/keras2C"))
sys.path.append(os.path.abspath(cwd))


import unittest
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Conv2D, ConvLSTM2D, Dot, Add, Multiply, Concatenate, Reshape, Permute, ZeroPadding1D, Cropping1D
from tensorflow.keras.models import Model
# ----------------------------------------------------------------------------------
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
# %matplotlib inline # matplotlib is not a python code
from keras.optimizers import Adam ,RMSprop
from keras import  backend as K
from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
# ----------------------------------------------------------------------------------
import numpy as np
from keras2c import keras2c_main
import subprocess
import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


CC = 'gcc'

def data_preparation():
    # load dataset
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # count the number of unique train labels
    unique, counts = np.unique(y_train, return_counts=True)
    print("Unique Train labels: ", dict(zip(unique, counts)))

    # count the number of unique test labels
    unique, counts = np.unique(y_test, return_counts=True)
    print("\nUnique Test labels: ", dict(zip(unique, counts)))
    
    # ---- Visualize
    # sample 25 mnist digits from train dataset
    # indexes = np.random.randint(0, x_train.shape[0], size=25)
    # images = x_train[indexes]
    # labels = y_train[indexes]
     
    # # plot the 25 mnist digits
    # plt.figure(figsize=(5,5))
    # for i in range(len(indexes)):
    #     plt.subplot(5, 5, i + 1)
    #     image = images[i]
    #     plt.imshow(image, cmap='gray')
    #     plt.axis('off')
        
    # plt.show()
    # plt.savefig("mnist-samples.png")
    # plt.close('all') 


    # compute the number of labels
    num_labels = len(np.unique(y_train))


    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # image dimensions (assumed square)
    image_size = x_train.shape[1]
    input_size = image_size * image_size
    print('input size:', input_size)
    

    # resize and normalize
    x_train = np.reshape(x_train, [-1, input_size])
    x_train = x_train.astype('float32') / 255
    x_test = np.reshape(x_test, [-1, input_size])
    x_test = x_test.astype('float32') / 255

    return [input_size, num_labels, x_train, y_train, x_test, y_test]
def mnist_training(input_size, num_labels, x_train, y_train):
    
    # network parameters
    batch_size = 128
    hidden_units = 256
    dropout = 0.45

    # ---- Model description
    a = keras.layers.Input(input_size)
    b = keras.layers.Dense(hidden_units, activation='relu', use_bias=False, name='dense_1')(a)
    c = keras.layers.Dropout(dropout, name='dropout_1')(b)
    d = keras.layers.Dense(hidden_units, activation='relu', use_bias=False, name='dense_2')(c)
    e = keras.layers.Dropout(dropout, name='dropout_2')(d)
    f = keras.layers.Dense(num_labels, use_bias=False, name='dense_3')(e)
    g = keras.layers.Activation('softmax', name='activation_1')(f)
    model = keras.models.Model(inputs=a, outputs=g)



    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    


    model.fit(x_train, y_train, epochs=2, batch_size=batch_size)

    model.save("mnist.h5")


    
def build_and_run(name, return_output=False):

    cwd = os.getcwd()
    os.chdir(os.path.abspath('./include/'))
    lib_code = subprocess.run(['make']).returncode
    os.chdir(os.path.abspath(cwd))
    if lib_code != 0:
        return 'lib build failed'

    if os.environ.get('CI'):
        ccflags = '-g -Og -std=c99 --coverage -I./include/'
    else:
        ccflags = '-Ofast -std=c99 -I./include/'

    cc = CC + ' ' + ccflags + ' -o ' + name + ' ' + name + '.c ' + \
        name + '_test_suite.c -L./include/ -l:libkeras2c.a -lm'
    build_code = subprocess.run(cc.split()).returncode
    if build_code != 0:
        return 'build failed'
    proc_output = subprocess.run(['./' + name])
    rcode = proc_output.returncode
    if rcode == 0:
        if not os.environ.get('CI'):
            return (rcode, proc_output.stdout) if return_output else rcode
    return rcode
# -----------------------------------
def MNIST_dense():
    
    name = 'mnist_dense'
    # Remove former generated files
    subprocess.run('rm ' + name + '*', shell=True)
    keras2c_main.k2c('mnist.h5', name)
    build_and_run(name)
    return
def MNIST_test(x_test, y_test):
  batch_size = 128
  model = keras.models.load_model('mnist.h5')
  loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
  print("\nTest accuracy: %.1f%%" % (100.0 * acc))

# -----------------------------------
if __name__ == "__main__":
    [input_size, num_labels, x_train, y_train, x_test, y_test] = data_preparation()
    mnist_training(input_size, num_labels, x_train, y_train)
    MNIST_test(x_test, y_test)
    # MNIST_dense()
