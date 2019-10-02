#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.metrics import MeanMetricWrapper

################################ generalized Dice loss ################################

def multiclass_dsc_loss(ground_truth,prediction,from_logits=False,name=None):
    """
        compute approximation of dice score balanced on differents classes 
            ground_truth : sparse label
            prediction : softmax output of neural network
    """
    num_classes = prediction.shape[-1]
    smooth = 0.1
    indices = tf.squeeze(tf.cast(ground_truth,dtype=tf.int32))
    preds = tf.squeeze(prediction)
    onehot_labels = tf.one_hot(indices=indices,depth=num_classes,dtype=tf.float32,name='onehot_labels')
    label_sum = tf.reduce_sum(onehot_labels, axis=[0,1,2], name='label_sum')
    pred_sum = tf.reduce_sum(preds, axis=[0,1,2], name='pred_sum')
    intersection = tf.reduce_sum(onehot_labels * preds, axis=[0,1,2],name='intersection')
    return tf.reduce_mean(1.-(2. * intersection + smooth)/(label_sum + pred_sum + smooth))

class Multiclass_DSC_Loss(LossFunctionWrapper):
    """
        Tensorflow compatible wrapper of dice loss function
    """
    def __init__(self,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
                 name='multiclass_dsc_loss'):
        super(Multiclass_DSC_Loss,self).__init__(
            fn=multiclass_dsc_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits) 

################################ TUMORAL dice metric #############################
        
def tumoral_dsc_metric(ground_truth,prediction,from_logits=False,axis=-1,name=None):
    """
        compute dice on tumoral segmentation
        /!\ : needs binary segmentation as 0=Background, 1=Lymphoma
    """
    tumor_prediction = tf.argmax(prediction,axis=axis)
    
    labels = tf.squeeze(tf.cast(ground_truth,dtype=tf.float32))
    preds = tf.squeeze(tf.cast(tumor_prediction,dtype=tf.float32))
    label_sum = tf.reduce_sum(labels, axis=[0,1,2], name='label_sum')
    pred_sum = tf.reduce_sum(preds, axis=[0,1,2], name='pred_sum')
    intersection = tf.reduce_sum(labels * preds, axis=[0,1,2],name='intersection')
    return tf.reduce_mean((2. * intersection + 0.1)/(label_sum + pred_sum + 0.1))

class Tumoral_DSC(MeanMetricWrapper):
    """
        Tensorflow compatible wrapper of dice metric function
    """
    def __init__(self, 
                 name='tumoral_dsc_metric', 
                 dtype=None):
        super(Tumoral_DSC, self).__init__(
            fn=tumoral_dsc_metric, 
            name=name, 
            dtype=dtype)
