#!/usr/bin/env python
# coding: utf-8

# In[]:


import SimpleITK as sitk
import numpy as np
from scipy.ndimage.interpolation import zoom

def change_labels_order(masks_filenames,path_output,
                        old_labels_numbers,old_labels_names,
                        new_labels_numbers):
    """
        From a list of masks filenames defined with old_labels_
        numbers/names generates new masks according to 
        new_labels_numbers and save them at path_output
        
        return reordered labels names
    """
    assert(len(old_labels_numbers)==len(old_labels_names)==len(new_labels_numbers)), "labels names, number or new labels are not same size"
    
    for filename in masks_filenames:
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage(filename,sitk.sitkVectorUInt8))
        
        new_mask = swap_mask_labels(mask,old_labels_numbers,new_labels_numbers)
        
        new_filename = path_output+'/'+basename(filename)
        sitk.WriteImage(sitk.GetImageFromArray(new_mask),new_filename)
        
        print("save image: %s" % basename(filename))
    
    new_labels_names = [old_labels_names[old_labels_numbers.index(number)] for number in new_labels_numbers]
    
    return new_labels_names

def swap_mask_labels(mask,old_labels_numbers,new_labels_numbers):
    """
        * function called by change_labels_order *
        
        Given a mask, swap labels order according to labels numbers
        before/after
        
        ex:
            old_labels_numbers = [0,1,2,3] #can be different but not recommended
            new_labels_numbers = [3,2,0,1]
            labels_values      = [0,1,2,3] #automatic
            
            # label 0 now has value 2
            # label 1 now has value 3
            # label 2 now has value 1
            # label 3 now has value 0

        By default, Background/Unknow = 0
    """
    new_mask = np.zeros((mask.shape),dtype=mask.dtype)
    
    # label value = value coded on the mask for the concerned ROI
    for label_value in range(1,len(old_labels_numbers)):
        
        value = old_labels_numbers.index(new_labels_numbers[label_value])
        new_mask = np.where(mask==value,label_value,new_mask)

    # check if last ROI correspond to default background
    # a.k.a if this is a semantic segmentation
    condition_1 = mask==old_labels_numbers.index(new_labels_numbers[0])
    condition_2 = new_mask==0
    assert(np.all(condition_1==condition_2)), "new label 0 (Background) is different than old label %s" % old_labels_numbers.index(new_labels_numbers[0])
        
    return new_mask

def resample(input_img, image_size, order, logger=None):
    """
        Resample 3D image to a defined size with ndimage spline interpolation
    """
    
    dim_input = np.shape(input_img)
    dim_output = image_size
    
    zoom_io = (dim_output[0]/dim_input[0],
               dim_output[1]/dim_input[1],
               dim_output[2]/dim_input[2]
              )
    
    output_img = zoom(input_img,zoom=zoom_io,order=order,mode='mirror')
    
    if logger!=None:
        logger.info("Image size input"+str(np.shape(input_img)))
        logger.info("Image size output"+str(np.shape(output_img)))
        
    return output_img

def isometric_resample(input_img,
                       input_pixel_size,
                       output_shape=[448,176,176],
                       output_pixel_size=[4.1,4.1,4.1],
                       interpolation_order=3):
    """
        Isotropic resample:
            - zoom the input img to get the right voxel size
            - centered on the scan, crop img to the final output shape
        
        Mathematically, as we zoom exactly based on pixel size, the shape is an
        approximation, a small shift of the structures might exist
        
        Params:
            - input_img :
            - input_shape :
            - input_shape :
            - output_shape :
            - output_pixel_size :
            - interpolation_order :
    """
    input_dtype = input_img.dtype
    output_img = np.zeros(output_shape,dtype=input_dtype)

    # deform image to the output pixel size
    zoom_io = np.asarray(input_pixel_size)/np.asarray(output_pixel_size)
    input_img = zoom(input_img,
                     zoom=zoom_io,
                     order=interpolation_order,
                     mode='mirror')
    
    input_shape = input_img.shape
    
    # without deformation, incorporate zoom_input_img in output_img
    center_input  = np.asarray(input_shape)//2
    center_output = np.asarray(output_shape)//2
    
    c_min = np.minimum(center_input,center_output)
    
    cimcoi = center_input-c_min
    cipcoi = center_input+c_min
    comcii = center_output-c_min
    copcii = center_output+c_min

    output_img[comcii[0]:copcii[0],
               comcii[1]:copcii[1],
               comcii[2]:copcii[2]] = input_img[cimcoi[0]:cipcoi[0],
                                                cimcoi[1]:cipcoi[1],
                                                cimcoi[2]:cipcoi[2]]
    return output_img
