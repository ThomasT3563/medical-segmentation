#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


def Simple_Binary_model(input_shape):
    """
        Simple model for 3d segmentation of 1 class
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=1,input_shape=input_shape, kernel_size=(1, 1, 1), padding="same", activation="sigmoid"))
    return model
