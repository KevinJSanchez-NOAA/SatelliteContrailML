### -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:26:13 2021

@author: kjsanche


"""

import time
#time.sleep(60*60*3)
from matplotlib import pyplot as plt
#%matplotlib inline
import matplotlib as mpl
from itertools import compress
import numpy as np
import gc
import struct
import os
import glob
import random
from twilio.rest import Client
#from decouple import config
from UNET_Functions import unet_model, summary
from Sat_contrail_read import Extract_RawDef, extract_img, extract_mask, extract_imglist, get_model_memory_usage
import tensorflow as tf
import sys
sys.path.append('/home/kjsanche/Desktop/Projects/loss')
#from loss_function import *
from tensorflow.python.ops.metrics_impl import false_positives, false_negatives
import tensorflow.keras.metrics as tfm
#import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
#from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss

configproto = tf.compat.v1.ConfigProto() 
#configproto.gpu_options.per_process_gpu_memory_fraction = 0.8 # fraction of memory
configproto.gpu_options.visible_device_list = "0"
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

sys_details = tf.sysconfig.get_build_info()
print(sys_details)
cudnn_version = sys_details["cudnn_version"]
cuda_version = sys_details["cuda_version"]

print('cuda version: ', cuda_version)
print('cudNN version: ',cudnn_version)
print('TF version: ', tf.version.VERSION)




BATCH_SIZE = 128
EPOCHS = 30
IMG_W=64
IMG_H=64
N_CHANNELS = 7
N_FILTERS = 64
LEARNING_RATE = 0.000001
AUTO = tf.data.AUTOTUNE # used in tf.data.Dataset API
VERSION = 'T121' # v## or T121 for one case
TFrecord_path ='/home/kjsanche/Desktop/TFrecords/'
Models_path ='/home/kjsanche/Desktop/ExternalSSD/SatContrailData/Models/'

training_filenames=sorted(tf.io.gfile.glob([TFrecord_path + '*' + VERSION + '.tfrecords']))

if N_CHANNELS == 3:
    validation_filenames=sorted(tf.io.gfile.glob([TFrecord_path + '*vVALIDATION.tfrecords']))
elif N_CHANNELS == 7:
    validation_filenames=sorted(tf.io.gfile.glob([TFrecord_path + '*v7CHANNEL_' +str(IMG_W)+ 'Val.tfrecords']))
#test_filenames=sorted(tf.io.gfile.glob([TFrecord_path + '*vTEST.tfrecords']))

random.Random(5).shuffle(training_filenames)
random.Random(5).shuffle(validation_filenames)
training_filenames = training_filenames[:1400]
print(len(training_filenames))
print(len(validation_filenames))


account_sid = 'ACe04b332ed3f99aa10895372bd8ea5034'#config('ACCOUNT_ID',default='')
auth_token = '83a01a63442a4dfb5a3d6ec1bf92abba'#config('AUTHENTICATION_TOKEN',default='')
phone_num = '3019741551'#config('PHONE_NUMBER',default='')
client = Client(account_sid, auth_token)

#message = client.messages .create(
#                    body =  "Testing 1, 2, 3, 4", #Message you send
#                    from_ = '+12184383951',#Provided phone number
#                    to =    phone_num)#Your phone number
#message.sid

class SMSCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%1 == 0:
            FP = logs['val_false_positives']
            TP = logs['val_true_positives']
            FN = logs['val_false_negatives']
            TN = logs['val_true_negatives']
            IOU_val = TP/(FN+TP+FP)
            message = client.messages .create(
                
                body = "For Epoch:{}, "
                        "Training loss={:.2f}, Validation loss ={:.2f} and Validation IOU = {:.2f}"
                .format(epoch, logs["loss"], logs["val_loss"], IOU_val),

                from_ = "+12184383951", to = phone_num)

            print(message.sid)
            
            
def parse_tfr_element(element):
    
    data = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width':tf.io.FixedLenFeature([], tf.int64),
      'depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_label':tf.io.FixedLenFeature([], tf.string),#tf.string = bytestring (not text string)
      'raw_image' : tf.io.FixedLenFeature([], tf.string),#tf.string = bytestring (not text string)
    }


    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    depth = content['depth']
    raw_label = content['raw_label']
    raw_image = content['raw_image']


    #get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(raw_image, out_type=tf.float16)
    feature = tf.reshape(feature, shape=[height,width,depth])
    label = tf.io.parse_tensor(raw_label, out_type=tf.int8)
    label = tf.reshape(label, shape=[height,width])
    return (feature, label)

def get_batched_dataset(filenames, testing = False):
    option_no_order = tf.data.Options()
    if testing:
        option_no_order.experimental_deterministic = True
    else:
        option_no_order.experimental_deterministic = False

    #dataset = tf.data.TFRecordDataset(filenames)
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
    dataset = dataset.map(parse_tfr_element, num_parallel_calls=AUTO)

    #dataset = dataset.cache() # If dataset fits in RAM
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
    if not testing:
        dataset = dataset.shuffle(2048)
    #dataset = dataset.repeat()
    
    dataset = dataset.prefetch(AUTO) #

    return dataset


def get_training_dataset(training_filenames):
    return get_batched_dataset(training_filenames)

def get_validation_dataset(validation_filenames):
    return get_batched_dataset(validation_filenames)

def get_test_dataset(test_filenames):
    return get_batched_dataset(test_filenames, testing = True)

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        #print(i)
        #print(display_list[i].shape)
        if i == 0:
            plt.imshow(np.float32(display_list[i][:,:,7]))#-display_list[i][:,:,1]))
        else:
            plt.imshow(np.float32(1*display_list[i]))
        plt.axis('off')
    plt.show()
    
def FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):
        '''
        ... in the case of α=β=0.5 the Tversky index simplifies to be 
        the same as the Dice coefficient, which is also equal to the F1 
        score. With α=β=1, Equation 2 produces Tanimoto coefficient, and 
        setting α+β=1 produces the set of Fβ scores. Larger βs weigh 
        recall higher than precision (by placing more emphasis on false negatives).
        '''
        targets = tf.cast(targets,tf.float32)
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth) 

        softdice = 2*TP/(K.sum(inputs**2)+K.sum(targets**2)+smooth)
        Tversky = softdice
        
        FocalTversky = K.pow((1 - Tversky), gamma)

        
        
        return FocalTversky
    

#gamma>1 reduces the relative loss for well-classified examples 
#alpha is a weighted term whose value is α for positive(foreground) alpha = 1 does nothing. alpha = 0.25 is best
#class and 1-α for negative(background) class.

unet = unet_model((IMG_W, IMG_H, N_CHANNELS),n_filters=N_FILTERS,n_classes=1)
#loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#loss=tfa.losses.SigmoidFocalCrossEntropy(),
#loss=[BinaryFocalLoss(gamma=2,from_logits=True)],

#Larger βs weigh recall higher than precision (by placing more emphasis on false negatives)
#loss=FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6)
unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=[FocalTverskyLoss],
              metrics=[tfm.Precision(), tfm.Recall(), tfm.FalseNegatives(), tfm.FalsePositives(), tfm.TruePositives(), tfm.TrueNegatives()],
              run_eagerly=True)

unet.summary(line_length = 130)
print('mem usage: ', tf.config.experimental.get_memory_usage("GPU:0"))

print(get_model_memory_usage(BATCH_SIZE, unet))

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()
        
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.01, patience=3, mode ="min", verbose = 2, restore_best_weights=True)
training_data = get_training_dataset(training_filenames)
validation_data = get_validation_dataset(validation_filenames)
#test_data = get_test_dataset(test_filenames)
print(training_data)
model_history = unet.fit(training_data, validation_data=validation_data, epochs=EPOCHS, callbacks = [SMSCallback(), ClearMemory(), callback])


F1 = 2*np.divide(np.multiply(model_history.history['precision'],model_history.history['recall']), np.add(model_history.history['precision'], model_history.history['recall']))
F1_val = 2*np.divide(np.multiply(model_history.history['val_precision'],model_history.history['val_recall']), np.add(model_history.history['val_precision'], model_history.history['val_recall']))
FP = model_history.history['false_positives']
TP = model_history.history['true_positives']
FN = model_history.history['false_negatives']
TN = model_history.history['true_negatives']
IOU = np.divide(TP,np.add(FN,np.add(TP,FP)))
MCC = np.subtract(np.multiply(TP,TN),np.multiply(FP,FN))/np.sqrt(np.multiply(np.multiply(np.add(TP,FP),np.add(TP,FN)), np.multiply(np.add(TN,FP),np.add(TN,FN))))
FP = model_history.history['val_false_positives']
TP = model_history.history['val_true_positives']
FN = model_history.history['val_false_negatives']
TN = model_history.history['val_true_negatives']
MCC_val = np.subtract(np.multiply(TP,TN),np.multiply(FP,FN))/np.sqrt(np.multiply(np.multiply(np.add(TP,FP),np.add(TP,FN)), np.multiply(np.add(TN,FP),np.add(TN,FN))))
IOU_val = np.divide(TP,np.add(FN,np.add(TP,FP)))

axs[1,0].plot(IOU)
axs[1,0].plot(IOU_val)
axs[1,0].set_title(f'model IoU, max = {np.max(IOU_val)}')
axs[1,0].set_ylabel('IoU')
axs[1,0].legend(['train', 'val'], loc='upper right')

axs[0,1].plot(F1)
axs[0,1].plot(F1_val)
axs[0,1].set_title('model F1 score')
axs[0,1].set_ylabel('F1')
axs[0,1].set_xlabel('epoch')

axs[1,1].plot(MCC)
axs[1,1].plot(MCC_val)
axs[1,1].set_title('model MCC')
axs[1,1].set_ylabel('MCC')
axs[1,1].set_xlabel('epoch')


i = 1
while os.path.exists(Models_path+"Model" + VERSION + "run%s.png" % i):
    i += 1
plt.savefig(Models_path+"Model" + VERSION + "run%s.png" % i)

# Save the weights 
unet.save_weights(Models_path+ "Model" + VERSION + "run%s.png" % i)


message = client.messages .create(
                
    body = "Model stopped at Epoch:{}, "
            "max Validation IOU = {:.2f}"
    .format(len(IOU_val), np.max(IOU_val)),

    from_ = "+12184383951", to = phone_num)

print(message.sid)
