#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

class My_Test_Model(object):
    """
        WIP
        doesnt work atm
    """
    def __init__(self,shape_image):
        self.num_classes = 5
        self.shape_image = shape_image

        self.filters = 32 #[16,32,64,128,256]
        self.kernel = (3,3,3)
        self.activation = tf.keras.layers.LeakyReLU()
        self.padding = "same"
    
    def convolution(self,input_):
        layer1 = tf.keras.layers.Conv3D(filters=self.filters,
                                        kernel_size=self.kernel,
                                        padding=self.padding,
                                       )(input_)
        layer2 = tf.keras.layers.BatchNormalization()(layer1)
        layer3 = self.activation(layer2)
        return layer3
    
    def final_convolution(self,input_):
        layer1 = tf.keras.layers.Conv3D(filters=self.num_classes,
                                        kernel_size=(1, 1, 1),
                                        padding=self.padding,
                                       )(input_)
        layer2 = tf.keras.layers.BatchNormalization()(layer1)
        layer3 = self.activation(layer2)
        return layer3
    
    def residual_learning(self,stage_input,conv_output):
        residual_connection = conv_output + tf.pad(stage_input,[[0,0],[0,0],[0,0],[0,0],[0,conv_output.shape[4]-stage_input.shape[4]]])
        return residual_connection
    
    def get_model(self):
        input_ = tf.keras.layers.Input(shape=self.shape_image,dtype=tf.float32,name="MODEL_INPUT")
        
        layer1 = self.convolution(input_)
        layer2 = self.convolution(layer1)
        layer3 = self.convolution(layer2)
        layer4 = self.residual_learning(input_,layer3)

        logits = self.final_convolution(layer4)
        dropout = tf.keras.layers.Dropout(0.5)(logits)
        output_ = tf.keras.layers.Softmax(name='MODEL_OUTPUT')(dropout)
        model = tf.keras.models.Model(input_, output_,name='TestModel')
        
        return model