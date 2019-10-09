#!/usr/bin/env python
# coding: utf-8

# In[]:

###########################  IMPORTS   ############################################# 

import tensorflow as tf
print("Tensorflow version : %s" % tf.__version__)

import csv
import time
import random
import glob
import os
import numpy as np
from math import ceil
import warnings

from deeplearning_tools.loss_functions import Multiclass_DSC_Loss
from deeplearning_tools.loss_functions import Tumoral_DSC
from global_tools.tools import display_learning_curves
from class_modalities.modality_PETCT import Modality_TRAINING_PET_CT

###########################  RULES   ############################################# 

# REQUIRED : .csv file containing all images filenames per patient
#           example : patient n°1 | PET | CT | MASK |
#                     patient n°2 | PET | CT | MASK |
#                     etc

csv_filenames = "/media/storage/projet_LYSA_TEP_3.5/TEP_CT_training_filenames.csv"

# definition of modality
MODALITY = Modality_TRAINING_PET_CT()

# path folders
path_preprocessed = '/media/storage/projet_LYSA_TEP_3.5/PREPROCESS_PETCT_4'
path_results = '/media/storage/projet_LYSA_TEP_3.5/RESULTS_PETCT_4'

# generates folders
if not os.path.exists(path_preprocessed):
    os.makedirs(path_preprocessed)
if not os.path.exists(path_results):
    os.makedirs(path_results)
    
# preprocess parameters
PREPROCESS_DATA = True
visualisation_preprocessed_files = True
IMAGE_SHAPE  = [368,128,128]
PIXEL_SIZE   = [4.8,4.8,4.8]
DATA_AUGMENT = True
RESIZE       = True
NORMALIZE    = True 

# training parameters
trained_model_path = '/media/storage/projet_LYSA_TEP_3.5/RESULTS_PETCT_4/model_09241142.h5' # or None
if trained_model_path is None:
    # CNN that will be generated and trained
    from deeplearning_models.Unet import custom_Unet3D as CNN
    
SHUFFLE = True
labels_names   = MODALITY.labels_names   # example for TEP :['Background','Lymphoma',]
labels_numbers = MODALITY.labels_numbers #                 :[0,1]
ITERATIONS = 50000
BATCH_SIZE = 2 # TODO : increase image size/resolution
GPU_NB=2 # TODO : add dynamic evaluation

# visualisation parameters
PREDICTION_TRAINING_SET   = False #(for development or verification purpose)
PREDICTION_VALIDATION_SET = True
PREDICTION_TEST_SET       = False #(final trained model only)
saving_directives = {
    'Save history'      : True,
    'Save model'        : True
}

###########################  PREPROCESSING   ############################################# 

if PREPROCESS_DATA:

    # read filenames from csv file
    all_patients_filenames = []
    with open(csv_filenames,"r") as file:
        filereader = csv.reader(file)
        for row in filereader:
            all_patients_filenames.append(row)

    # define training / validation / tests sets : 80% / 10% / 10%
    BUFFER_SIZE = len(all_patients_filenames)
    random.shuffle(all_patients_filenames)
    valid_set = all_patients_filenames[slice(0,ceil(0.1*BUFFER_SIZE))]
    test_set  = all_patients_filenames[slice(ceil(0.1*BUFFER_SIZE),ceil(0.2*BUFFER_SIZE))]
    train_set = all_patients_filenames[slice(ceil(0.2*BUFFER_SIZE),BUFFER_SIZE)]

    preprocessed_sets = {'TRAIN_SET':None,'VALID_SET':None,'TEST_SET':None}
    
    # loop overs sets
    for folder,data_set_ids in zip(['TRAIN_SET','VALID_SET','TEST_SET'],[train_set,valid_set,test_set]):

        print("Preprocessing : %s" % folder)
        
        # generates folder
        directory = path_preprocessed+'/'+folder
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # preprocess files
        preprocessed_sets[folder] = MODALITY.PREPROCESS(data_set_ids,
                                                        path_output=directory,
                                                        output_shape=IMAGE_SHAPE,
                                                        pixel_size=PIXEL_SIZE,
                                                        resample=RESIZE,
                                                        normalize=NORMALIZE)     
        print('')

    if DATA_AUGMENT: # perform only on train set
        print("Data augmentation :")

        preprocessed_sets['TRAIN_SET'] += MODALITY.DATA_AUGMENTATION(preprocessed_sets['TRAIN_SET'],
                                                                     augmentation_factor=3)
        print('')

    if visualisation_preprocessed_files:
        print("Generation MIP .pdf :")
        MODALITY.VISUALISATION_MIP_PREPROCESS(path_preprocessed,
                                              preprocessed_sets['TRAIN_SET'],
                                              filename="MIP_preprocessed_training_data.pdf")
        print('') 
        MODALITY.VISUALISATION_MIP_PREPROCESS(path_preprocessed,
                                              preprocessed_sets['VALID_SET'],
                                              filename="MIP_preprocessed_validation_data.pdf")
        print('') 
        MODALITY.VISUALISATION_MIP_PREPROCESS(path_preprocessed,
                                              preprocessed_sets['TEST_SET'],
                                              filename="MIP_preprocessed_test_data.pdf")
        print('') 

else:
    # load previously generated files
     warnings.warn("Not preprocessing datas, be sure to know what you're doing")
    
    preprocessed_sets = {'TRAIN_SET':None,'VALID_SET':None,'TEST_SET':None}
    
    for folder in ['TRAIN_SET','VALID_SET','TEST_SET']:
        
        directory = path_preprocessed+'/'+folder
        
        PET_ids = np.sort(glob.glob(directory+'/*float*.nii'))
        CT_ids = np.sort(glob.glob(directory+'/*ctUh*.nii'))
        MASK_ids = np.sort(glob.glob(directory+'/*pmask*.nii'))
        
        preprocessed_sets[folder] = list(zip(PET_ids,CT_ids,MASK_ids))
    
###########################  TRAINING   #############################################

# shuffle training data
if SHUFFLE:
    random.shuffle(preprocessed_sets['TRAIN_SET'])
    
# preparation of tensorflow DATASETS
train_generator = MODALITY.get_GENERATOR(preprocessed_sets['TRAIN_SET'])
train_dataset = tf.data.Dataset.from_generator(train_generator.call,train_generator.types,train_generator.shapes).batch(BATCH_SIZE).repeat()

valid_generator = MODALITY.get_GENERATOR(preprocessed_sets['VALID_SET'])
valid_dataset = tf.data.Dataset.from_generator(valid_generator.call,valid_generator.types,valid_generator.shapes).batch(BATCH_SIZE).repeat()

# MODEL PREPARATION
epochs = int(ITERATIONS/len(preprocessed_sets['TRAIN_SET']))

strategy = tf.distribute.MirroredStrategy()

if trained_model_path is None:
    # GENERATE NEW MODEL
    number_channels = MODALITY.number_channels
    cnn_img_shape = tuple(IMAGE_SHAPE.copy()+[number_channels])
    with strategy.scope():
        model = CNN(cnn_img_shape,len(labels_names)).get_model()
else:
    # LOAD MODEL FROM .h5 FILE
    with strategy.scope():
        model = tf.keras.models.load_model(trained_model_path,compile=False)

# definition of loss, optimizer and metrics
loss_object = Multiclass_DSC_Loss()
metrics = [Tumoral_DSC(),tf.keras.metrics.SparseCategoricalCrossentropy(name='SCCE'),]
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5,momentum=0.1)

# TODO : generate a learning rate scheduler
with strategy.scope():
    model.compile(loss=loss_object,optimizer=optimizer,metrics=metrics)

# LEARNING PROCEDURE

start_tt = time.time()

history = model.fit(
    x=train_dataset,
    validation_data=valid_dataset,
    validation_steps=len(preprocessed_sets['VALID_SET']),
    epochs=epochs)

# TIMER
total_tt = time.time() - start_tt
hours = int(total_tt//3600)
mins = int((total_tt-hours*3600)//60)
sec = int((total_tt-hours*3600-mins*60))
print("\n\nRun time = "+str(hours)+':'+str(mins)+':'+str(sec)+' (H:M:S)')

###########################  VISUALISATION   #############################################

# plot learning curves and save history
if history:
    
    print("Learning curves :")
    display_learning_curves(history)

    if saving_directives['Save history']:
        filename = path_results+"/history_"+time.strftime("%m%d%H%M")+".dat"
        print("Saving history: %s" % filename)
        with open(filename,'w') as file:
            file.write(str(history.history))
            
# save whole model as .h5 file
if saving_directives['Save model']:
    filename = path_results+"/model_"+time.strftime("%m%d%H%M")+".h5"
    print("Saving model : %s" % filename)
    model.save(filename)
    
if PREDICTION_TRAINING_SET:
    print("Prediction on training set : /!\ use only for development or verification purpose")
   
    n_sample = min(20,len(preprocessed_sets['TRAIN_SET'])) #number of training imgs visualised
    random.shuffle(preprocessed_sets['TRAIN_SET'])
    
    filename = "/RESULTS_train_set_"+time.strftime("%m%d%H%M%S")+".pdf"
    
    print("Generating predictions :")
    train_prediction_ids = MODALITY.PREDICT_MASK(data_set_ids=preprocessed_sets['TRAIN_SET'][:n_sample],
                                                 path_predictions=path_results+'/train_predictions',
                                                 model=model)
    
    print("\nDisplaying stats and MIP : %s" % filename)
    MODALITY.VISUALISATION_MIP_PREDICTION(path_results,
                                          data_set_ids=preprocessed_sets['TRAIN_SET'][:n_sample],
                                          pred_ids=train_prediction_ids,
                                          filename=filename)

if PREDICTION_VALIDATION_SET:
    print("Prediction on validation set :")
    # use to fine tune and evaluate model performances
    
    filename = "/RESULTS_valid_set_"+time.strftime("%m%d%H%M%S")+".pdf"
    
    print("Generating predictions :")
    valid_prediction_ids = MODALITY.PREDICT_MASK(data_set_ids=preprocessed_sets['VALID_SET'],
                                                 path_predictions=path_results+'/valid_predictions',
                                                 model=model)
    
    print("\nDisplaying stats and MIP : %s" % filename)
    MODALITY.VISUALISATION_MIP_PREDICTION(path_results,
                                          data_set_ids=preprocessed_sets['VALID_SET'],
                                          pred_ids=valid_prediction_ids,
                                          filename=filename)

if PREDICTION_TEST_SET:
    print("Prediction on test set : /!\ use only on fully trained model")

    filename = "/RESULTS_test_set_"+time.strftime("%m%d%H%M%S")+".pdf"
    
    print("Generating predictions :")
    test_prediction_ids = MODALITY.PREDICT_MASK(data_set_ids=preprocessed_sets['TEST_SET'],
                                                 path_predictions=path_results+'/test_predictions',
                                                 model=model)
    
    print("\nDisplaying stats and MIP : %s" % filename)
    MODALITY.VISUALISATION_MIP_PREDICTION(path_results,
                                          data_set_ids=preprocessed_sets['TEST_SET'],
                                          pred_ids=test_prediction_ids,
                                          filename=filename)