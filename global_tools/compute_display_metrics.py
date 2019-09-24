#!/usr/bin/env python
# coding: utf-8

# In[]:

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def compute_metrics(ground_truth,prediction,
                    Conf,TruePos,GTsum,PRsum,
                    labels_names,labels_numbers
                   ):
    """
        from a prediction and its ground truth,
        compute accuracy, dice score
        and update board containing important values
            Conf: confusion matrix
            TruePos: board of true positives values
            GTsum: board of ground truth sums
            PRsum: board of prediction sums
    """
    num_class = len(labels_names)
    
    TP_mask = ground_truth==prediction
    patient_confusion_matrix = []

    TP = []
    normX = []
    normY = []
    for label_number in labels_numbers:
        lab_mask = ground_truth==label_number

        TP.append( np.sum(np.logical_and(TP_mask,lab_mask)) )
        normX.append(np.sum(lab_mask))
        normY.append(np.sum(prediction==label_number))

        patient_confusion_raw = []
        for pred_label in labels_numbers:
            y_pred = prediction==pred_label
            patient_confusion_raw.append( np.sum(np.logical_and(lab_mask,y_pred)) )
        patient_confusion_matrix.append(np.array(patient_confusion_raw))

    Conf += np.stack(patient_confusion_matrix)
    TruePos.append(np.array(TP)) 
    GTsum.append(np.array(normX))
    PRsum.append(np.array(normY))
    
    return None

def display_metric_boards(confusion_matrix,TruePos,GT_sum,PR_sum,
                          labels_names,labels_numbers):
    """
        Use with compute stats
        With board returned by compute_stats, plot graphs of different metrics
            - accuracy, balanced accuracy
            - dice score, balanced dice score 
    """
    
    print("\nConfusion matrix:")
    print(confusion_matrix)

    accuracy_TAB = TruePos/GT_sum
    accuracy = np.mean(np.sum(TruePos,axis=1)/np.sum(GT_sum,axis=1))
    balanced_acc = np.mean(accuracy_TAB)
    print("\naccuracy          = %7.7s" % accuracy)
    print("balanced accuracy = %7.7s" % balanced_acc)
    
    #figure accuracy
    plt.figure()
    plt.boxplot(accuracy_TAB[:,:],sym='x',whis=5,labels=labels_names)
    plt.title("Accuracy Boxplot")
    plt.show()
    
    smooth = 1
    dice_TAB = (2*TruePos+smooth) / (GT_sum+PR_sum+smooth)
    dice = (2*np.sum(TruePos)+smooth)/(np.sum(GT_sum)+np.sum(PR_sum)+smooth)
    balanced_dice = np.mean(dice_TAB)
    
    print("\ndice              = %7.7s" % dice)
    print("balanced dice     = %7.7s" % balanced_dice)
    
    #figure dice
    plt.figure()
    plt.boxplot(dice_TAB[:,:],sym='x',whis=5,labels=labels_names)
    plt.title("Dice Boxplot")
    plt.show()
    
    return (accuracy_TAB,accuracy,balanced_acc),(dice_TAB,dice,balanced_dice)

def display_stats(masks_ids,preds_ids,labels_names,labels_numbers):
    """
        Load ground truth and prediction, compute and evaluate stats
        to display it
            - masks_ids : list of filenames of ground truth masks
            - preds_ids : list of filenames of prediction masks
    """
    
    num_class = len(labels_numbers)
    
    # initialize values board
    Conf = np.zeros((num_class,num_class),dtype=np.int32)
    TruePos = []
    GTsum = []
    PRsum = []
    
    for filename_true,filename_pred in zip(masks_ids,preds_ids):
        
        ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(filename_true,sitk.sitkVectorUInt8))
        prediction   = sitk.GetArrayFromImage(sitk.ReadImage(filename_pred,sitk.sitkVectorUInt8))
        
        compute_metrics(ground_truth,prediction,
                        Conf,TruePos,GTsum,PRsum,
                        labels_names=labels_names,
                        labels_numbers=labels_numbers)
    
    # final conversion of board of values
    TruePos = np.stack(TruePos)
    GTsum = np.stack(GTsum)
    PRsum = np.stack(PRsum)

    print("Displaying stats:")
    display_metric_boards(Conf,TruePos,GTsum,PRsum,
                          labels_names=labels_names,
                          labels_numbers=labels_numbers)
    return None


