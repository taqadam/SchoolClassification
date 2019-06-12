import sys
import argparse
import cv2
import glob
import numpy as np
import re
import h5py
import skimage

from generator import DataGenerator
import tensorflow as tf
from keras import optimizers
from keras.optimizers import Adam
import keras
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras.utils.vis_utils import plot_model

from keras import backend as K

# Custom keras functions and layers for mobilenetv2

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def relu6(x):
    return K.relu(x, max_value=6.0)

def _conv_block(inputs, filters, kernel, strides):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x

# Model Bundles

class Resnet50Model():

    def __init__(self):
        model = ResNet50(weights='imagenet', include_top=False)

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.7)(x)
        predictions = Dense(2, activation= 'softmax')(x)
        adjusted_model = Model(inputs = model.input, outputs = predictions)
        self.model = adjusted_model

    def modelOptimizer(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.summary()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

class ModelBundleMobilenetV2():
    def __init__(self, imageSize=(224,224,3), k = 2, alpha=1.0, lr=0.001):
        self.lr = lr
        inputs = Input(shape=imageSize)

        first_filters = _make_divisible(32 * alpha, 8)
        x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

        x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
        x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
        x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
        x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=2)
        x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=1)

        if alpha > 1.0:
            last_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_filters = 1280

        x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, last_filters))(x)
        x = Dropout(0.3, name='Dropout')(x)
        x = Conv2D(k, (1, 1), padding='same')(x)

        x = Activation('softmax', name='softmax')(x)
        output = Reshape((k,))(x)

        model = Model(inputs, output)

        self.model = model

    def modelOptimizer(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(lr=self.lr)

        self.model.summary()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

class ModelBundle224():
    def __init__(self,lr=0.001, imageSize=(224,224,3)):
        self.lr = lr

        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=imageSize))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(96,activation='relu'))
        model.add(Dense(2, activation='softmax'))
        self.model = model

    def modelOptimizer(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
            #optimizer = optimizers.SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=True) # -- slower convergence

        self.model.summary()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

class ModelBundle518():
    def __init__(self,lr=0.001, imageSize=(518,518,3)):
        self.lr = lr

        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=imageSize))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(DepthwiseConv2D(3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(96,activation='relu'))
        model.add(Dense(2, activation='softmax'))
        self.model = model

    def modelOptimizer(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
            #optimizer = optimizers.SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=True) # -- slower convergence

        self.model.summary()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
