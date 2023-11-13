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
import numpy as np
from keras2c import keras2c_main
import subprocess
import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


CC = 'gcc'


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

def test_CIFAR_10_CNN():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(8, (3, 3), padding='same',
                                      input_shape=(32, 32, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(8, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Conv2D(8, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(8, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(20))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation('softmax'))
    name = 'test___CIFAR_10_CNN'
    # Remove former generated files
    subprocess.run('rm ' + name + '*', shell=True)
    keras2c_main.k2c(model, name)
    build_and_run(name)
    return

if __name__ == "__main__":
    test_CIFAR_10_CNN()
