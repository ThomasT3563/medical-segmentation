#!/usr/bin/env python
# coding: utf-8

###########################  IMPORTS   ############################################# 

import tensorflow as tf
print("Tensorflow version : %s" % tf.__version__)

import numpy as np
import glob
import os
import csv
import time

from class_modalities.modality_PETCT import Modality_PREDICTION_PET_CT


###########################  RULES   ############################################# 

# REQUIRED : .csv file containing all images filenames per patient
#           example : patient n°1 | PET | CT |
#                     patient n°2 | PET | CT |
#                     etc

csv_filenames = "/media/storage/projet_LYSA_TEP_3.5/TEP_CT_prediction_filenames.csv"

# definition of modality
MODALITY = Modality_PREDICTION_PET_CT()

# parameters
trained_model_path = '/media/storage/projet_LYSA_TEP_3.5/RESULTS_PETCT_4/model_09241142.h5'
path_output = '/media/storage/projet_LYSA_TEP_3.5/PREDICTIONS_TEPCT'

# preprocess parameters
IMAGE_SHAPE  = [368,128,128]
PIXEL_SIZE   = [4.8,4.8,4.8]
RESIZE       = True
NORMALIZE    = True

# TIMER
start_tt = time.time()

# prediction
VISUALISATION_MIP = True

# generates folders
if not os.path.exists(path_output):
    os.makedirs(path_output)
    
for folder in ['/PREPROCESSED_DATA','/RAW_CNN_PREDICTIONS','/POSTPROCESSED_PREDICTIONS']:
    directory = path_output+folder
    if not os.path.exists(directory):
        os.makedirs(directory)
        
###########################  PREPROCESSING  ############################################# 

# read filenames from csv file
patients_filenames = []
with open(csv_filenames,"r") as file:
    filereader = csv.reader(file)
    for row in filereader:
        patients_filenames.append(row)
        
# TODO : multi core preprocessing

# preprocessing
preprocessed_filenames = MODALITY.PREPROCESS(data_set_ids=patients_filenames,
                                             path_output=path_output+'/PREPROCESSED_DATA',
                                             output_shape=IMAGE_SHAPE,
                                             pixel_size=PIXEL_SIZE,
                                             resample=RESIZE,
                                             normalize=NORMALIZE)
print('')

###########################  PREDICTION  ############################################# 

# loading trained model
trained_model = tf.keras.models.load_model(trained_model_path,compile=False)

# generating predictions
predictions_ids = MODALITY.PREDICT_MASK(data_set_ids=preprocessed_filenames,
                                        path_predictions=path_output+'/RAW_CNN_PREDICTIONS',
                                        trained_model=trained_model)
print('')

###########################  POSTPROCESSING  ############################################# 

postprocessed_ids = MODALITY.POSTPROCESS(path_output=path_output+'/POSTPROCESSED_PREDICTIONS',
                                         data_set_ids=patients_filenames,
                                         prediction_ids=predictions_ids,
                                         input_pixel_size=PIXEL_SIZE,
                                         resize=RESIZE)
print('')

# TIMER
total_tt = time.time() - start_tt
hours = int(total_tt//3600)
mins = int((total_tt-hours*3600)//60)
sec = int((total_tt-hours*3600-mins*60))
print("Run time = "+str(hours)+':'+str(mins)+':'+str(sec)+' (H:M:S)')

if VISUALISATION_MIP:
    filename = "/MIP_inference_"+time.strftime("%m%d%H%M%S")+".pdf" 
    print("Generating MIP visualisation : %s" % filename)
    MODALITY.GENERATES_MIP_PREDICTION(path_output=path_output,
                                      data_set_ids=preprocessed_filenames,
                                      pred_ids=predictions_ids,
                                      filename=filename)

print('\nDone!')