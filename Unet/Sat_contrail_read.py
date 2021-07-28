import numpy as np
import struct


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


def extract_img(image_list,tmp,dim):
# convert binary files to matrix of integers
    
    for idx,files in enumerate(image_list):
        with open(files, mode='rb') as file: # b is important -> binary
            fileContent = file.read()
            x = np.uint32(np.reshape(struct.unpack("H"*dim[idx,1]*dim[idx,0] , fileContent),(dim[idx,1],dim[idx,0]))) 
            tmp[idx, 0:x.shape[0], 0:x.shape[1]] += x
    return tmp




def extract_mask(mask_list,tmp,dim):
# convert binary files to matrix of integers
    for idx,files in enumerate(mask_list):
        with open(files, mode='rb') as file: # b is important -> binary
            fileContent = file.read()
            x = np.uint16(np.reshape(struct.unpack("B"*dim[idx,1]*dim[idx,0] , fileContent),(dim[idx,1],dim[idx,0]))) 
            tmp[idx, 0:x.shape[0], 0:x.shape[1]] += x
    return tmp
