import numpy as np
import struct
import glob
import os
from os.path import exists
import tensorflow as tf


def Extract_RawDef(AUX_list):
## obtain dimensions from .AUX files.
    for file_path in AUX_list:
        f = open(file_path, "r")
        for lines in f:
            if 'RawDefinition:' in lines:    
                str_values = lines.split(":")[-1].strip()    
                RawDeftmp = [int(x) for x in str_values.split()]
                if 'RawDef' in locals():
                    RawDef = np.append(RawDef, [RawDeftmp], axis=0)
                else:
                    RawDef = [RawDeftmp]
            
                break
    RawDef = np.delete(RawDef, 2, 1)
    return RawDef


#def extract_img(files,tmp,dim):
## convert binary files to matrix of integers
#    
#    #for idx,files in enumerate(image_list):
#    with open(files, mode='rb') as file: # b is important -> binary
#        fileContent = file.read()
#        x = np.uint32(np.reshape(struct.unpack("H"*dim[0,1]*dim[0,0] , fileContent),(dim[0,1],dim[0,0]))) 
#        tmp[0, 0:x.shape[0], 0:x.shape[1]] += x
#    return tmp

def extract_imglist(image_path, NEWMASK, NCHANNELS):
    
    #Black_list = ['2018MYD/109/A2018109.1540']
    image0065 = glob.glob(image_path + "/**/01__1km.raw", recursive = True)
    image0380 = glob.glob(image_path + "/**/20__1km.raw", recursive = True)
    image0680 = glob.glob(image_path + "/**/27__1km.raw", recursive = True)
    image0850 = glob.glob(image_path + "/**/29__1km.raw", recursive = True)
    image1100 = glob.glob(image_path + "/**/31__1km.raw", recursive = True)
    image1200 = glob.glob(image_path + "/**/32__1km.raw", recursive = True)
    image1330 = glob.glob(image_path + "/**/33__1km.raw", recursive = True)
    AUX_list = glob.glob(image_path + "/**/01__1km.AUX", recursive = True)
    mask_list = glob.glob(image_path + "/**/*.contrail-mask", recursive = True)
    if NEWMASK:
        mask_list = glob.glob(image_path + "/**/*.contrail-maskUpdate", recursive = True)
    mask_list = [ x for x in mask_list if "_sw" not in x ]

    #exclude granuls with missing files
    for x in os.walk(image_path):
        if '/A2018' in x[0]:
            aa = exists(x[0]+'/01__1km.raw')
            bb = exists(x[0]+'/20__1km.raw')
            cc = exists(x[0]+'/27__1km.raw')
            dd = exists(x[0]+'/29__1km.raw')
            ee = exists(x[0]+'/31__1km.raw')
            ff = exists(x[0]+'/32__1km.raw')
            gg = exists(x[0]+'/33__1km.raw')
            hh = exists(x[0]+'/01__1km.AUX')
            masks = glob.glob(x[0]+'/*.contrail-mask')
            if NEWMASK:
                masks = glob.glob(x[0]+'/*.contrail-maskUpdate')
            ii = 0
            ii = [1 for s in masks if "_sw" not in s]
            if not (aa and bb and cc and dd and ee and ff and gg and hh and ii):
                image0065 = [s for s in image0065 if x[0] not in s]
                image0380 = [s for s in image0380 if x[0] not in s]
                image0680 = [s for s in image0680 if x[0] not in s]
                image0850 = [s for s in image0850 if x[0] not in s]
                image1330 = [s for s in image1330 if x[0] not in s]
                image1100 = [s for s in image1100 if x[0] not in s]
                image1200 = [s for s in image1200 if x[0] not in s]
                AUX_list = [s for s in AUX_list if x[0] not in s]
                mask_list = [s for s in mask_list if x[0] not in s]
                


    return image0065, image0380, image0680, image0850, image1100, image1200, image1330, AUX_list, mask_list

def extract_img(files, dim0, dim1):
# convert binary files to matrix of integers
    #for idx,files in enumerate(image_list):
    img = np.zeros((4096,4096,1), dtype = int)
    with open(files, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        x = np.uint32(np.reshape(struct.unpack("H"*dim1*dim0 , fileContent),(dim1,dim0))) 
        if x.shape[0] <= 4096 and x.shape[1] <= 4096:
            img[0:x.shape[0], 0:x.shape[1], 0] += x
        else:
            print(files)
                
    return img


def extract_mask(files,dim0, dim1):
# convert binary files to matrix of integers
    #for idx,files in enumerate(mask_list):
    mask = np.zeros((4096,4096,1), dtype = int)
    with open(files, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        x = np.uint16(np.reshape(struct.unpack("B"*dim1*dim0 , fileContent),(dim1,dim0))) 
        if x.shape[0] <= 4096 and x.shape[1] <= 4096:
            mask[0:x.shape[0], 0:x.shape[1], 0] += x
        else:
            print(files)
        mask = np.int8(1*(mask>0))
    return mask


class DatasetIterator(tf.data.Dataset):
    def _generator(num_samples):
        #open the file
        
        for sample_idx in range(num_samples):
            #read data file
            
            yield (sample_idx,)
    
    def __new__(cls, num_samples = 4):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (1,), dtype = tf.int64),
        args =(num_samples,)
        )
    
def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes