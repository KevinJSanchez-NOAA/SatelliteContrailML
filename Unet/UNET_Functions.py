# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script file called by Contrail_Unet.ipynb
"""
import tensorflow as tf
import os
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import imageio
#import matplotlib.pyplot as plt

from termcolor import colored
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D






def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """


    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size 
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = BatchNormalization()(conv)
    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:

        conv = Dropout(dropout_prob)(conv)

         
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:

        next_layer = MaxPooling2D(pool_size = 2)(conv)

        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """


    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides=2,
                 padding='same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)

    conv = BatchNormalization()(conv)
    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)

    conv = BatchNormalization()(conv)

    
    return conv
def upsampling_blockv2(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """


    up = UpSampling2D()(expansive_input)
    up = Conv2D(n_filters, 2,  padding='same')(up)

    #Conv2DTranspose(
    #             n_filters,    # number of filters
    #             3,    # Kernel size
    #             strides=2,
    #             padding='same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)

    conv = BatchNormalization()(conv)
    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)

    conv = BatchNormalization()(conv)

    
    return conv

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=8):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)

    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    cblock1 = conv_block(inputs, n_filters)
    # Chain the first element of the output of each block to be the input of the next conv_block. 
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*2*2)
    cblock4 = conv_block(cblock3[0], n_filters*2*2*2, dropout_prob=0)
    #cblock4 = conv_block(cblock3[0], n_filters*2*2*2, dropout_prob=0.5,max_pooling=None)
    # Include a dropout of 0.5 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(cblock4[0], n_filters*2*2*2*2, dropout_prob=0.5, max_pooling=None)
    
    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8

    ublock6 = upsampling_blockv2(cblock5[0], cblock4[1],  n_filters*8)
    
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
    # At each step, use half the number of filters of the previous block
    #ublock7 = upsampling_blockv2(cblock4[0], cblock3[1],  n_filters*8/2)
    ublock7 = upsampling_blockv2(ublock6, cblock3[1],  n_filters*8/2)
    ublock8 = upsampling_blockv2(ublock7, cblock2[1],  n_filters*8/2/2)
    ublock9 = upsampling_blockv2(ublock8, cblock1[1],  n_filters*8/2/2/2)
    
    conv9 = Conv2D(n_filters,
             3,
             activation='relu',
             padding='same',
             kernel_initializer='he_normal')(ublock9)
    conv9 = BatchNormalization()(conv9)
    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv10 = Conv2D(n_classes, kernel_size=1, padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=tf.keras.activations.sigmoid(tf.squeeze(conv10,3)))


   
    return model

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    return

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
    
# Summary of model
def summary(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
        if (type(layer) == Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if (type(layer) == MaxPooling2D):
            descriptors.append(layer.pool_size)
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
        result.append(descriptors)
    return result
